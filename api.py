"""JSON API for external frontends (the Vercel-hosted web UI in web/).

One call runs the classical, deep-learning and hybrid pipelines on the same image and returns
everything a UI needs: the three outputs, the image diagnosis, plain-language explanations,
no-reference quality checks, per-stage timings, compact intermediate maps and a transparent
per-image recommendation. Registered in app.py with gr.api(..., api_name="restore"); call it with

    POST {BACKEND}/gradio_api/call/restore   {"data": [image_data_url, preset, strength]}
      -> {"event_id": "..."}
    GET  {BACKEND}/gradio_api/call/restore/{event_id}   (server-sent events; "complete" carries the JSON)

No secrets are involved: the endpoint is public, stateless and stores nothing.
"""
import base64
import fnmatch
import os
import time

import cv2
import gradio as gr
import numpy as np

from lowlight import analysis as A
from lowlight import classical as C
from lowlight import metrics as M
from lowlight import pipeline as P

MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))  # processing resolution cap (speed on 2 vCPUs)
MIN_SIDE = 32
MAX_BYTES = 15 * 2 ** 20  # decoded upload limit; the web UI downsizes photos before sending
# Optional origin allow-list, e.g. "https://lumina-t76.vercel.app,https://*-darshchandran.vercel.app".
# Empty = any origin (the API is public, stateless and uses no cookies or credentials).
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

METHODS = {"classical": "Classical CV", "dl": "Deep learning", "hybrid": "Hybrid"}
# Transparent recommendation: lower penalty is better. Weights are a stated design choice, not learned.
WEIGHTS = {"exposure": 0.35, "noise": 0.35, "clipping": 0.2, "colour": 0.1}

VIZ_HELP = {
    "Illumination map T": "How much light reached each area (bright = well lit). Max of R,G,B, smoothed by an edge-aware guided filter.",
    "Reflectance R = I / T": "The scene's own colours and textures with the lighting divided out (Retinex). Grainy where it was dark - hence the restoration step.",
    "Retinex gain T^(gamma-1)": "How strongly each area is brightened: dark corners get a lot, lamps and lit areas almost none, so nothing blows out.",
    "Curve strength |A|": "The deep model's per-pixel tone-curve strength (Zero-DCE++). Curves can only re-tone pixels, never invent content.",
    "Classical candidate (Retinex)": "Candidate 1: classical Retinex re-lighting.",
    "DL candidate (Zero-DCE++)": "Candidate 2: fine-tuned deep curve estimation, exposure picked by the quality check.",
    "Fusion weight (DL share)": "Where the result trusts the deep model (bright) vs Retinex (dark), chosen per pixel from exposure + detail quality.",
    "Noise-level map (to DL)": "Where brightening amplified sensor noise the most. Handed to the restoration network so it knows where to clean hard.",
    "Removed noise / corrections (x4)": "What the restoration network changed, magnified 4x. Should look like grain, not like object edges.",
    "Noise-removal strength": "Edge-preserving denoising strength: strongest where shadows were lifted most.",
}


class APIError(gr.Error):
    """A problem with the request that the user can fix (gr.Error: the message reaches the client)."""


def heat(m):
    """Colour-blind-safe (cividis) heatmap for single-channel maps."""
    return cv2.cvtColor(cv2.applyColorMap(C.to_u8(np.clip(m, 0, 1)), cv2.COLORMAP_CIVIDIS), cv2.COLOR_BGR2RGB)


def origin_allowed(origin):
    return not ALLOWED_ORIGINS or (origin is not None and any(fnmatch.fnmatch(origin, p) for p in ALLOWED_ORIGINS))


def decode_image(data):
    """data URL or bare base64 -> (float RGB image capped at MAX_SIDE, note, original (w, h))."""
    if not isinstance(data, str) or not data.strip():
        raise APIError("No image received. Choose a photo first.")
    b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(b64, validate=True)
    except ValueError as e:
        raise APIError("The upload was corrupted in transit. Please try again.") from e
    if len(raw) > MAX_BYTES:
        raise APIError(f"That image is too large ({len(raw) / 2 ** 20:.0f} MB). Please use one under 15 MB.")
    # IMREAD_COLOR: drops alpha, expands greyscale, applies EXIF orientation (phone photos come out upright)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise APIError("That file isn't a readable image. Please use a JPG, PNG or WebP photo.")
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise APIError(f"That image is only {w}x{h} px - too small to restore (minimum {MIN_SIDE} px per side).")
    note = ""
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        note = f"Processed at {img.shape[1]}x{img.shape[0]} px (resized from {w}x{h}) to stay fast."
    return C.to_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), note, (w, h)


def encode(img, quality=90, max_side=None, fmt="webp"):
    """RGB image (float [0,1] or uint8) -> WebP (default) or JPEG data URL, optionally downsized.
    WebP is ~30% smaller than JPEG at equal quality, which matters on slow mobile links."""
    u8 = C.to_u8(img) if img.dtype != np.uint8 else img
    if max_side and max(u8.shape[:2]) > max_side:
        s = max_side / max(u8.shape[:2])
        u8 = cv2.resize(u8, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    flag = cv2.IMWRITE_WEBP_QUALITY if fmt == "webp" else cv2.IMWRITE_JPEG_QUALITY
    ok, buf = cv2.imencode("." + fmt, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [flag, quality])
    return f"data:image/{fmt};base64," + base64.b64encode(buf.tobytes()).decode()


def quality(img):
    q = M.no_reference(img, with_niqe=False)
    return {"brightness": round(q["brightness"], 3), "noise": round(q["noise"], 2),
            "clipped_pct": round(q["overexposed"], 2), "colour_cast": round(q["colour_cast"], 1)}


def recommend(qualities, target):
    """Pick the output with the lowest weighted no-reference penalty. This is a heuristic (no ground
    truth exists for a user's photo), stated as such in the UI; it can and does pick any method."""
    terms = {m: {"exposure": abs(q["brightness"] - target) / 0.2, "noise": q["noise"] / 5.0,
                 "clipping": q["clipped_pct"] / 1.0, "colour": q["colour_cast"] / 15.0}
             for m, q in qualities.items()}
    scores = {m: round(sum(WEIGHTS[k] * v for k, v in t.items()), 3) for m, t in terms.items()}
    best = min(scores, key=scores.get)
    labels = {"exposure": "closest to the target brightness", "noise": "least visible grain",
              "clipping": "fewest blown-out pixels", "colour": "most neutral colour"}
    reasons = [f"{labels[k]}" for k in WEIGHTS if min(terms, key=lambda m: terms[m][k]) == best]
    return {"method": best, "label": METHODS[best], "scores": scores, "weights": WEIGHTS,
            "target_brightness": round(target, 3),
            "why": (f"{METHODS[best]} has the lowest combined penalty"
                    + (f": {', '.join(reasons)}." if reasons else "."))}


def restore(image: str, preset: str = "Auto (balanced)", strength: float = 1.0,
            request: gr.Request = None) -> dict:
    """Run classical, deep-learning and hybrid restoration on one image (data URL) and return JSON."""
    origin = request.headers.get("origin") if request is not None else None
    if not origin_allowed(origin):
        raise APIError("This site is not allowed to use the Lumina API.")
    preset = preset if preset in A.PRESETS else "Auto (balanced)"
    strength = float(np.clip(float(strength), 0.6, 1.4))
    t0 = time.perf_counter()
    x, note, (w0, h0) = decode_image(image)
    decode_ms = (time.perf_counter() - t0) * 1000

    runs = {}
    for key in METHODS:
        t = time.perf_counter()
        runs[key] = P.run(x, key, preset, strength)
        runs[key]["ms"] = round((time.perf_counter() - t) * 1000)
    hyb = runs["hybrid"]
    d, p = hyb["diagnosis"], hyb["params"]
    seen, did = A.explain(d, p)
    level, why = A.recoverability(d, p)
    qualities = {k: quality(r["output"]) for k, r in runs.items()}
    target = A.TARGET_BRIGHTNESS * A.PRESETS[preset] * strength

    out = {
        "input": {"width": w0, "height": h0, "processed_width": x.shape[1], "processed_height": x.shape[0],
                  "note": note, "quality": quality(x)},
        "diagnosis": A.as_dict(d),
        "params": A.as_dict(p),
        "recoverability": {"level": level, "why": why},
        "noticed": seen,
        "actions": did,
        "grain_uncovered": round(p.denoise_sigma, 1),
        "results": {k: {"label": METHODS[k], "image": encode(r["output"], 90), "ms": r["ms"],
                        "quality": qualities[k]} for k, r in runs.items()},
        "recommendation": recommend(qualities, target),
        "intermediates": [{"name": k, "caption": VIZ_HELP.get(k, ""),
                           "image": encode(heat(v) if v.ndim == 2 else v, 80, max_side=384)}
                          for k, v in hyb["viz"].items()],
        "timings_ms": {"decode": round(decode_ms), **{k: round(v) for k, v in hyb["timings_ms"].items()}},
        "total_ms": round((time.perf_counter() - t0) * 1000),
        "model": "Hybrid: Retinex + fine-tuned Zero-DCE++ + noise-aware restoration U-Net (v3)",
    }
    return out


if __name__ == "__main__":  # self-check: python api.py
    rng = np.random.default_rng(0)
    img = (np.clip(rng.normal(0.04, 0.01, (90, 120, 3)), 0, 1) * 255).astype(np.uint8)
    res = restore(encode(img, 95, fmt="jpg"))
    assert set(res["results"]) == set(METHODS) and res["recommendation"]["method"] in METHODS
    assert all(r["image"].startswith("data:image/webp;base64,") for r in res["results"].values())
    for bad in ("", "not-base64!!", base64.b64encode(b"hello").decode()):
        try:
            restore(bad)
            raise AssertionError("bad input accepted")
        except APIError:
            pass
    print("api self-check ok:", res["recommendation"]["why"], res["total_ms"], "ms")
