"""End-to-end enhancement methods: baselines, classical-only, DL-only and the hybrid (ours).

Hybrid:
  1. classical illumination estimate T + image diagnosis -> adaptive parameters
  2. two brightening candidates: Retinex re-lighting (classical) and Zero-DCE++ (DL) with a
     quality-aware exposure search
  3. per-pixel quality-weighted multi-scale fusion of the candidates + soft highlight guard
  4. adaptive CLAHE + colour correction
  5. DL restoration U-Net conditioned on a noise-level map derived from T (where shadows were
     lifted most, noise was amplified most)
  6. user intent (preset / strength) applied as a final exposure adjustment
Every stage can be switched off with a flag; that is how the ablation study is run.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from . import analysis as A
from . import classical as C
from . import dce, restore

FULL = dict(adaptive=True, fusion=True, dl_branch=True, quality_aware=True, restore=True,
            noise_map=True, denoise=True, gain_aware=True, clahe=True, colour=True,
            details=True, highlights=True, quality=True)

ABLATIONS = {  # name -> flags turned off (evaluated by scripts/evaluate.py)
    "Hybrid (full)": {},
    "w/o adaptive parameters (fixed)": {"adaptive": False},
    "w/o Zero-DCE++ branch (Retinex only)": {"dl_branch": False},
    "w/o Retinex branch (Zero-DCE++ only)": {"fusion": False},
    "w/o quality-aware curve scaling": {"quality_aware": False},
    "w/o DL restoration (classical denoise)": {"restore": False},
    "w/o noise-map conditioning": {"noise_map": False},
    "w/o CLAHE": {"clahe": False},
    "w/o colour correction": {"colour": False},
    "w/o highlight protection": {"highlights": False},
    "w/o detail recovery": {"details": False},
    "w/o quality-aware strategy": {"quality": False},
    "previous pipeline (before quality-aware update)": {"details": False, "quality": False},
}
NOMAP_WEIGHTS = Path(restore.WEIGHTS).with_name("restore_net_nomap.pth")

DEBLOCK_CLAHE = 0.4  # CLAHE share on deblocked JPEGs (validation: compressed proxy)
DENOISE_SCALE = 1.0  # classical denoiser strength knob (tuned on the LOL training split)


class Timer:
    def __init__(self):
        self.t, self.stages = time.perf_counter(), {}

    def __call__(self, name):
        now = time.perf_counter()
        self.stages[name] = round((now - self.t) * 1000, 1)
        self.t = now


def _thumb(x, side=256):
    s = side / max(x.shape[:2])
    return cv2.resize(x, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else x


def choose_dce_scale(img, a, target, grid=np.linspace(0.1, 2.0, 20)):
    """Image-quality-aware exposure control for the DL branch: try several curve strengths on a
    thumbnail and keep the one closest to the target brightness that blows out the fewest
    pixels that weren't already clipped."""
    img_s, a_s = _thumb(img), _thumb(a)
    pre = img_s.max(axis=2) >= 0.98

    def score(s):
        out = dce.apply_curve_np(img_s, a_s, s)
        clip = ((out.max(axis=2) >= 0.98) & ~pre).mean()
        return abs(C.luminance(out).mean() - target) + 2.0 * clip

    return float(min(grid, key=score))


def fusion_weights(cand, src):
    """Per-pixel quality (Mertens-style): well-exposedness x local contrast x amplification trust.
    Contrast is measured on a blurred copy so amplified noise doesn't count as 'detail'; the trust
    term prefers the candidate that needed less gain (less noise blown up) to look well exposed."""
    y = C.luminance(cand)
    contrast = np.abs(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 1.5), cv2.CV_32F)) + 1e-3
    expo = np.exp(-((y - 0.5) ** 2) / (2 * 0.22 ** 2))
    gain = y / np.maximum(C.luminance(src), 1 / 255)
    return contrast * expo / (1 + (gain / 12) ** 2) + 1e-6


def pyramid_blend(a, b, w_b, levels=5):
    """Laplacian-pyramid blend (Burt & Adelson): seamless at every scale, no halo at weight edges."""
    def gauss(x):
        p = [x]
        for _ in range(levels - 1):
            p.append(cv2.pyrDown(p[-1]))
        return p

    def lap(x):
        g = gauss(x)
        return [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(levels - 1)] + [g[-1]]

    out = [la * (1 - w[..., None]) + lb * w[..., None] for la, lb, w in zip(lap(a), lap(b), gauss(w_b))]
    res = out[-1]
    for lv in reversed(out[:-1]):
        res = cv2.pyrUp(res, dstsize=lv.shape[1::-1]) + lv
    return res


def noise_map(img, x):
    """Per-pixel noise std after brightening (0..255 units). The level is measured on the brightened
    image (in the dark input, 8-bit quantisation and JPEG smoothing hide it), on all three channels
    so chroma noise counts, and spread spatially in proportion to how much each area was amplified."""
    gain = cv2.GaussianBlur((C.luminance(x) / np.maximum(C.luminance(img), 1 / 255)).astype(np.float32), (0, 0), 3)
    sigma = float(np.mean([A.estimate_noise(np.ascontiguousarray(x[..., c])) for c in range(3)]))
    return gain / max(float(gain.mean()), 1e-3) * sigma


def classical_denoise(x, local, p, f, viz):
    """Gain-aware edge-preserving denoise: strongest where brightening amplified noise most."""
    if f["adaptive"]:
        p.denoise_sigma = float(np.clip(np.percentile(local, 90) * DENOISE_SCALE, 0.5, 45))
    strong = C.denoise(x, p.denoise_sigma)
    w = np.clip(local / max(p.denoise_sigma, 1e-3), 0, 1) if f["gain_aware"] else np.ones_like(local)
    viz["Noise-removal strength"] = w
    return strong * w[..., None] + x * (1 - w[..., None])


def prepare(img, f, target_mult=1.0, tm=None, viz=None):
    """Steps 1-4 of the hybrid (everything before DL restoration). Also used to build the
    restoration network's training inputs, so train and inference see identical data."""
    tm = tm or Timer()
    viz = {} if viz is None else viz
    t = C.estimate_illumination(img)
    tm("illumination estimation")
    d = A.diagnose(img, t)
    p = A.choose_params(d, target_mult, f["adaptive"])
    q = f.get("quality", False)
    cap = p.gain_cap if q else C.GAIN_CAP
    src = img
    # denoise BEFORE brightening multiplies the noise / JPEG blocks. With DL restoration only for
    # blocky JPEGs: the restoration net was trained on raw sensor noise (validation: pre-denoising
    # clean-but-noisy PNGs before it lowered SSIM 0.824 -> 0.812).
    p.pre_denoise = bool(q and (p.deblock or (p.pre_denoise and not f["restore"])))
    if p.pre_denoise:
        src = C.pre_denoise(img, d.noise_sigma, deblock=p.deblock)
    if f["adaptive"]:
        p.gamma = C.solve_gamma(_thumb(C.luminance(src)), _thumb(t), p.target, cap=cap)
    tm("diagnosis + adaptive params")
    viz["Illumination map T"] = t
    viz["Reflectance R = I / T"] = np.clip(img / t[..., None], 0, 1)

    cl, gain = C.retinex_enhance(src, t, p.gamma, cap)
    lg = np.log2(gain)
    viz["Retinex gain T^(gamma-1)"] = lg / max(float(lg.max()), 1e-3)
    tm("Retinex")
    x = cl
    if f["dl_branch"]:
        a = dce.curve_params(dce.load("dce"), src)
        if f["quality_aware"]:
            p.dce_scale = choose_dce_scale(src, a, p.target)
        dl = dce.apply_curve_np(src, a, p.dce_scale)
        viz["Curve strength |A|"] = np.abs(a).mean(axis=2)
        tm("Zero-DCE++ (deep learning)")
        if f["fusion"]:
            cl_c = np.clip(C.protect_highlights(cl, p.knee), 0, 1).astype(np.float32)
            w_cl, w_dl = fusion_weights(cl_c, src), fusion_weights(dl, src)
            w = cv2.GaussianBlur((w_dl / (w_cl + w_dl)).astype(np.float32), (0, 0), 2)
            x = pyramid_blend(cl, dl, w)
            viz["Classical candidate (Retinex)"], viz["DL candidate (Zero-DCE++)"] = cl_c, dl
            viz["Fusion weight (DL share)"] = w
            tm("quality-weighted fusion")
        else:
            x = dl
    if f["highlights"]:
        x = C.protect_highlights(x, p.knee)
    x = np.clip(x, 0, 1).astype(np.float32)
    tm("highlight guard")
    local = noise_map(src, x)
    p.denoise_sigma = float(np.percentile(local, 90))
    tm("noise-level map")
    return x, local, d, p, viz, tm


def contrast_colour(x, p, f, tm):
    q = f.get("quality", False)
    if f["clahe"]:
        if q and p.deblock:
            p.clahe_weight *= DEBLOCK_CLAHE  # local contrast would sharpen JPEG block edges
        x = C.clahe(x, p.clahe_clip, p.clahe_weight)
        tm("CLAHE")
    if f["colour"]:
        x = np.clip(C.colour_correct(x, p.wb_strength, p.saturation), 0, 1).astype(np.float32)
        if q:
            x = C.limit_saturation(x, p.max_colour).astype(np.float32)
        tm("colour correction")
    return x


def detail_stage(x, ref, ref_sigma, d, p, viz, tm):
    """Adaptive detail recovery after noise removal (decided from the cleaned image)."""
    res_noise = A.estimate_noise(C.luminance(x).astype(np.float32))
    sharp = A.sharpness_index(x)
    A.detail_strengths(p, d, res_noise, sharp, A.recoverability(d, p)[0])
    p.residual_noise, p.sharp_restored = res_noise, sharp
    before = x
    x = C.recover_detail(x, ref, res_noise, p.detail_reinject, p.detail_fine, p.detail_mid,
                         ref_sigma).astype(np.float32)
    viz["Recovered detail (x4)"] = np.clip((x - before) * 4 + 0.5, 0, 1)
    tm("detail recovery")
    return x


def run(img, method="hybrid", preset="Auto (balanced)", strength=1.0, **flags):
    """img: float32 RGB [0,1]. Returns dict(output, diagnosis, params, viz, timings_ms)."""
    f = {**FULL, **flags}
    img = np.clip(img, 0, 1).astype(np.float32)  # e.g. bicubic resizes overshoot slightly below 0
    mult = A.PRESETS.get(preset, 1.0) * strength
    tm, viz = Timer(), {}
    if method == "he":
        out = C.histogram_equalize(img)
        tm("histogram equalisation")
        return dict(output=out, viz=viz, timings_ms=tm.stages)
    if method == "clahe":
        out = C.clahe(img, 2.0)
        tm("CLAHE")
        return dict(output=out, viz=viz, timings_ms=tm.stages)
    if method in ("zero_dce_official", "dl"):
        a = dce.curve_params(dce.load("official" if method == "zero_dce_official" else "dce"), img)
        out = C.adjust_exposure(dce.apply_curve_np(img, a), mult)
        tm("Zero-DCE++")
        viz["Curve strength |A|"] = np.abs(a).mean(axis=2)
        return dict(output=out, viz=viz, timings_ms=tm.stages)

    if method == "classical":
        f.update(dl_branch=False, restore=False)
        x, local, d, p, viz, tm = prepare(img, f, mult, tm, viz)
        pre = x
        if f["denoise"]:
            x = classical_denoise(x, local, p, f, viz)
            tm("gain-aware denoise")
        x = contrast_colour(x, p, f, tm)
        if f["details"] and f["quality"]:
            x = detail_stage(x, pre, local, d, p, viz, tm)
        elif f["details"]:
            x = C.enhance_details(x, p.detail_beta)
            tm("detail enhancement")
        return dict(output=np.clip(x, 0, 1), diagnosis=d, params=p, viz=viz, timings_ms=tm.stages)

    # hybrid: intent is applied at the end so the restoration net always sees its training range
    x, local, d, p, viz, tm = prepare(img, f, 1.0, tm, viz)
    x = contrast_colour(x, p, f, tm)
    if f["restore"]:
        nmap = np.clip(local / restore.NOISE_NORM, 0, 1).astype(np.float32)
        viz["Noise-level map (to DL)"] = nmap
        before = x
        x = restore.restore(x, nmap, restore.WEIGHTS if f["noise_map"] else NOMAP_WEIGHTS)
        viz["Removed noise / corrections (x4)"] = np.clip((before - x) * 4 + 0.5, 0, 1)
        tm("DL restoration (noise-aware U-Net)")
    else:
        before = x
        x = classical_denoise(x, local, p, f, viz)
        tm("gain-aware denoise")
    if f["details"]:
        x = detail_stage(x, before, local, d, p, viz, tm)
    if abs(mult - 1) > 1e-3:
        x = C.protect_highlights(C.adjust_exposure(x, mult), p.knee)
        tm("user intent")
    return dict(output=np.clip(x, 0, 1).astype(np.float32), diagnosis=d, params=p, viz=viz,
                timings_ms=tm.stages)


PROCESS_SIDE = 1280   # longest side the pipeline itself runs at (CPU speed); outputs keep the input size
TRANSFER = 1.0        # share of the input's own above-noise fine detail transferred to the full-size output


def tile_stats(x, tile=32):
    """Per-tile robust noise std (1.4826 x median |x|) and structure share, interpolated to full size.
    Pure noise has RMS ~= its robust std; real edges and texture are sparse and heavy-tailed, so
    their RMS is well above it. Textured tiles read a higher noise floor, so the transfer is
    conservative there (it never invents detail)."""
    h, w = x.shape
    H, W = -(-h // tile) * tile, -(-w // tile) * tile
    t = np.pad(x, ((0, H - h), (0, W - w)), mode="reflect").reshape(H // tile, tile, W // tile, tile)
    t = t.transpose(0, 2, 1, 3).reshape(H // tile, W // tile, -1)
    sd = np.median(np.abs(t), axis=2) * 1.4826 + 1e-6
    structure = np.interp(np.sqrt((t * t).mean(axis=2)) / sd, [1.3, 1.8], [0.0, 1.0])
    up = lambda m: cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[:h, :w]
    return up(sd), up(structure)


def run_full(img, method="hybrid", preset="Auto (balanced)", strength=1.0, process_side=PROCESS_SIDE, **flags):
    """Resolution-preserving run(): enhance a copy reduced to <= process_side, then return the
    result at the INPUT size. Plain upscaling would look soft, so the fine-detail band the
    reduction removed is taken from the input photo itself (input minus its own low-pass),
    multiplied by the local brightening gain and cored at the equally amplified noise floor:
    only detail that is really in the photo, and stronger than its grain, comes back.
    Adds processed_size / output_size (w, h) to the result."""
    img = np.clip(img, 0, 1).astype(np.float32)
    h, w = img.shape[:2]
    s = process_side / max(h, w)
    if s >= 1:
        r = run(img, method, preset, strength, **flags)
        r["processed_size"] = r["output_size"] = (w, h)
        return r
    sw, sh = max(round(w * s), 1), max(round(h * s), 1)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    r = run(small, method, preset, strength, **flags)
    t0 = time.perf_counter()
    up = lambda m, k=cv2.INTER_CUBIC: cv2.resize(m, (w, h), interpolation=k)
    y = C.luminance(img).astype(np.float32)
    band = y - up(cv2.resize(y, (sw, sh), interpolation=cv2.INTER_AREA))
    gain = C.luminance(r["output"]) / np.maximum(C.luminance(small), 1 / 255)
    gain = up(cv2.GaussianBlur(np.clip(gain, 0, 32).astype(np.float32), (0, 0), 2), cv2.INTER_LINEAR)
    amp = band * gain
    sd, structure = tile_stats(amp)
    floor = 2.5 * np.maximum(sd, max(A.estimate_noise(y), 0.5) / 255 * gain)
    detail = TRANSFER * structure * C.core(amp, floor)
    r["output"] = np.clip(up(r["output"]) + detail[..., None], 0, 1).astype(np.float32)
    r["timings_ms"]["full-size detail transfer"] = round((time.perf_counter() - t0) * 1000, 1)
    r["processed_size"], r["output_size"] = (sw, sh), (w, h)
    return r


METHODS = {  # display name -> method key (order used in all tables)
    "Histogram equalisation (baseline)": "he",
    "CLAHE only (baseline)": "clahe",
    "Zero-DCE++ official (baseline)": "zero_dce_official",
    "Classical only (ours)": "classical",
    "Deep learning only (ours, fine-tuned)": "dl",
    "Hybrid (ours)": "hybrid",
}
