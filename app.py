"""Lumina - adaptive low-light image restoration (Gradio web app).

Run:  python app.py   ->  http://localhost:7860
"""
import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from lowlight import analysis as A
from lowlight import classical as C
from lowlight import metrics as M
from lowlight import pipeline as P

try:  # Hugging Face ZeroGPU hosting (free tier) requires at least one @spaces.GPU function to exist.
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_placeholder():
        # ponytail: never called - both networks (~155k params) run on CPU in well under a second,
        # which avoids GPU queueing and daily-quota failures during a live demo.
        return True
except ImportError:  # local / Docker runs
    pass

ROOT = Path(__file__).resolve().parent
MAX_SIDE = int(os.environ.get("MAX_SIDE", 1280))  # keeps every request well under ~10 s on 2 vCPUs
MIN_SIDE = 32
OUT_DIR = Path(tempfile.gettempdir()) / "lumina"
OUT_DIR.mkdir(exist_ok=True)
EXAMPLES = sorted(str(p) for p in (ROOT / "examples").glob("*.jpg"))

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


def heat(m):
    """Colour-blind-safe (cividis) heatmap for single-channel maps."""
    return cv2.cvtColor(cv2.applyColorMap(C.to_u8(np.clip(m, 0, 1)), cv2.COLORMAP_CIVIDIS), cv2.COLOR_BGR2RGB)


def load_input(img):
    """Validate and normalise whatever the browser sent; friendly errors for bad input."""
    if img is None:
        raise gr.Error("Please add a photo first - upload, paste, use the camera or tap an example.")
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise gr.Error("That file doesn't look like a normal colour photo. Try a JPG or PNG.")
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = (img / 257).astype(np.uint8) if img.max() > 255 else np.clip(img * (255 if img.max() <= 1 else 1), 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise gr.Error(f"That image is only {w}x{h} px - too small to restore. Please use one at least {MIN_SIDE} px per side.")
    note = ""
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        note = f"Resized from {w}x{h} to {img.shape[1]}x{img.shape[0]} px to keep it fast."
    return img, C.to_float(img), note


def quality_table(before, after, grain_uncovered):
    b, a = M.no_reference(before, with_niqe=False), M.no_reference(after, with_niqe=False)
    rows = [("Brightness", f"{b['brightness'] * 100:.0f}%", f"{a['brightness'] * 100:.0f}%"),
            ("Grain (noise sigma, /255)", f"{grain_uncovered:.1f}*", f"{a['noise']:.1f}"),
            ("Blown-out pixels", f"{b['overexposed']:.2f}%", f"{a['overexposed']:.2f}%"),
            ("Colour cast (0 = neutral)", f"{b['colour_cast']:.1f}", f"{a['colour_cast']:.1f}")]
    md = "| Check | Before | After |\n|---|---|---|\n" + "\n".join(f"| {r[0]} | {r[1]} | {r[2]} |" for r in rows)
    md += "\n\n<small>* grain uncovered by brightening, measured before noise removal (90th percentile).</small>"
    warn = []
    if a["overexposed"] > 1:
        warn.append(f"{a['overexposed']:.1f}% of pixels are at full white - usually light sources, which can't be recovered.")
    if a["noise"] > 6:
        warn.append("Some grain remains: the darkest areas had very little real signal.")
    if a["colour_cast"] > 10:
        warn.append("A colour tint remains - often intentional (coloured lighting).")
    return md, warn


def enhance(img, preset, strength):
    u8, x, note = load_input(img)
    t0 = time.perf_counter()
    try:
        r = P.run(x, "hybrid", preset, strength)
    except Exception as e:  # never show a stack trace to a user
        raise gr.Error(f"Sorry - processing failed ({type(e).__name__}). Please try another photo.") from e
    secs = time.perf_counter() - t0
    out, d, p = r["output"], r["diagnosis"], r["params"]
    out_u8 = C.to_u8(out)
    seen, did = A.explain(d, p)
    level, why = A.recoverability(d, p)
    qmd, warn = quality_table(x, out, p.denoise_sigma)
    icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[level]
    summary = (f"### {icon} Recoverability: {level}\n{why}\n\n"
               f"Done in **{secs:.1f} s** on CPU ({out.shape[1]}x{out.shape[0]} px). {note}")
    if warn:
        summary += "\n\n" + "\n".join(f"⚠️ {w}" for w in warn)

    gallery = [(u8, "Input"), (out_u8, "Final result")]
    for k, v in r["viz"].items():
        gallery.append((heat(v) if v.ndim == 2 else C.to_u8(v), f"{k} - {VIZ_HELP.get(k, '')}"))
    clip = out.max(axis=2) >= 0.98
    overlay = out_u8.copy()
    overlay[clip] = (255, 0, 255)
    gallery.append((overlay, f"Blown-out check - magenta marks pure-white pixels ({clip.mean() * 100:.2f}%)."))
    timings = "| Stage | ms |\n|---|---|\n" + "\n".join(f"| {k} | {v:.0f} |" for k, v in r["timings_ms"].items())

    for old in OUT_DIR.glob("*.png"):  # privacy: results are kept at most an hour
        if time.time() - old.stat().st_mtime > 3600:
            old.unlink(missing_ok=True)
    path = OUT_DIR / f"lumina_{time.time_ns()}.png"
    cv2.imwrite(str(path), cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))
    return ((u8, out_u8), summary, "\n".join(f"- {s}" for s in seen), "\n".join(f"- {s}" for s in did),
            qmd, gallery, timings, str(path))


def compare(img, preset, strength):
    u8, x, note = load_input(img)
    items, rows = [(u8, "Input")], []
    for name, key in P.METHODS.items():
        t0 = time.perf_counter()
        out = P.run(x, key, preset, strength)["output"]
        ms = (time.perf_counter() - t0) * 1000
        q = M.no_reference(out, with_niqe=False)
        items.append((C.to_u8(out), f"{name} - {ms:.0f} ms"))
        rows.append([name, round(ms), round(q["brightness"] * 100), round(q["noise"], 1),
                     round(q["overexposed"], 2), round(q["colour_cast"], 1)])
    return items, rows


def benchmarks_md():
    t = ROOT / "results" / "tables.md"
    return t.read_text(encoding="utf-8") if t.exists() else "Run `python scripts/evaluate.py` to generate benchmark tables."


ABOUT = (ROOT / "docs" / "about_app.md").read_text(encoding="utf-8") if (ROOT / "docs" / "about_app.md").exists() else ""
CSS = """
.gradio-container {max-width: 1200px !important; margin: auto;}
#enhance-btn {min-height: 52px; font-size: 1.1rem;}
"""

with gr.Blocks(title="Lumina - low-light restoration", delete_cache=(1800, 3600)) as demo:
    gr.Markdown("# 🌙 Lumina\n**See what the dark hid - without inventing what wasn't there.** "
                "Adaptive low-light restoration: classical illumination modelling + deep learning, "
                "with every decision explained.")
    with gr.Tabs():
        with gr.Tab("✨ Enhance"):
            with gr.Row(equal_height=False):
                with gr.Column():
                    inp = gr.Image(label="Your low-light photo", type="numpy", image_mode="RGB",
                                   sources=["upload", "webcam", "clipboard"], height=360)
                    preset = gr.Radio(list(A.PRESETS), value="Auto (balanced)", label="What do you want?",
                                      info="Natural keeps the night feel; Maximum visibility reveals the most detail.")
                    strength = gr.Slider(0.6, 1.4, value=1.0, step=0.05, label="Brightness",
                                         info="1.0 = what the system recommends for this photo")
                    btn = gr.Button("Enhance photo", variant="primary", size="lg", elem_id="enhance-btn")
                with gr.Column():
                    slider = gr.ImageSlider(label="Before | After (drag the handle)", type="numpy", height=360,
                                            buttons=["download", "fullscreen"])
                    summary = gr.Markdown("Upload a dark photo, or tap an example below.")
                    download = gr.File(label="Download full-quality PNG")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🔍 What we noticed")
                    seen = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### 🛠️ What we did (and why)")
                    did = gr.Markdown()
                with gr.Column():
                    gr.Markdown("#### ✅ Quality check")
                    quality = gr.Markdown()
            examples_slot = gr.Column()
        with gr.Tab("🔬 Under the hood"):
            gr.Markdown("Every intermediate step of the last enhancement, with a one-line explanation. "
                        "Heatmaps use the colour-blind-safe *cividis* scale (dark = low, yellow = high).")
            gallery = gr.Gallery(label="Pipeline stages", columns=3, height="auto", object_fit="contain")
            timings = gr.Markdown()
        with gr.Tab("⚖️ Compare methods"):
            gr.Markdown("Runs every method on the photo from the **Enhance** tab: two naive baselines, the "
                        "published Zero-DCE++, our classical-only, deep-learning-only and hybrid pipelines.")
            cmp_btn = gr.Button("Run all methods", variant="primary")
            cmp_gallery = gr.Gallery(label="Results", columns=3, height="auto", object_fit="contain")
            cmp_table = gr.Dataframe(headers=["Method", "Time (ms)", "Brightness %", "Grain sigma",
                                              "Blown-out %", "Colour cast"], label="No-reference checks")
        with gr.Tab("📊 Benchmarks"):
            gr.Markdown(benchmarks_md())
            figs = sorted(f for f in (ROOT / "results" / "figures").glob("*") if f.suffix in (".png", ".jpg"))
            if figs:
                gr.Gallery([(str(f), f.stem.replace("_", " ")) for f in figs], columns=2, height="auto",
                           label="Evaluation figures", object_fit="contain")
        with gr.Tab("ℹ️ About, limits & ethics"):
            gr.Markdown(ABOUT)

    outs = [slider, summary, seen, did, quality, gallery, timings, download]
    if EXAMPLES:
        with examples_slot:
            gr.Examples(EXAMPLES, inputs=inp, outputs=outs, fn=lambda im: enhance(im, "Auto (balanced)", 1.0),
                        run_on_click=True, cache_examples=False,
                        label="Unseen test photos (never used in training) - tap one")
    btn.click(enhance, [inp, preset, strength], outs)
    inp.upload(enhance, [inp, preset, strength], outs)
    cmp_btn.click(compare, [inp, preset, strength], [cmp_gallery, cmp_table])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("HOST", "0.0.0.0"), server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="indigo"), css=CSS,
        ssr_mode=False)  # plain client rendering: no blank first load behind the HF proxy
