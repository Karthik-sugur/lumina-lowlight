---
title: Lumina Low-Light Restoration
emoji: 🌙
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Adaptive hybrid (classical + DL) low-light image restoration
---

# 🌙 Lumina: adaptive low-light image restoration

**Computer Vision Hackathon 2026 · Problem P1: Robust Low-Light Image Restoration and Enhancement**

> *See what the dark hid, without inventing what wasn't there.*

**Live app:** https://lumina-t76.vercel.app  ·  **Engine (+ classic interface):** https://darshuuuu-lumina-lowlight.hf.space  ·  **Code:** https://github.com/darshchandran/lumina-lowlight  ·  **Evaluation:** [`eval_P1_T76.md`](eval_P1_T76.md)  ·  **Design doc:** [`docs/DESIGN.md`](docs/DESIGN.md)  ·  **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

Lumina restores dark, noisy, colour-shifted photos with a **hybrid** of classical computer vision and two small neural networks. Every enhancement setting is chosen per image from measurements of that image, and the app explains in plain language what it noticed and what it did.

![Visual comparison](results/figures/5_visual_comparison.jpg)

## Highlights

All test images are unseen (never used for training, validation or tuning). LOL-v2 real is restricted to the 9 pairs that are **not** pixel-identical to LOL-v1 training images (86 of its 100 inputs are; see the correction in [`eval_P1_T76.md`](eval_P1_T76.md)). CPU only.

| Average over 3 paired test sets (124 pairs, sets weighted equally) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | ΔE2000 ↓ |
|---|---|---|---|---|---|---|
| Zero-DCE++ (official weights) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 18.06 |
| Classical only (ours) | 16.74 | 0.618 | 0.477 | 6.92 | 5.38 | 16.51 |
| Deep learning only (ours) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 15.22 |
| **Hybrid (ours)** | **18.19** | **0.769** | **0.249** | **4.25** | **1.50** | **13.36** |

- **LOL-v1 eval15 / clean LOL-v2 real:** hybrid 19.13 / 18.65 dB PSNR, SSIM 0.777 / 0.770, LPIPS 0.207 / 0.283; best of all methods on both sets except new-overexposure.
- **Darkness sweep (30 unseen references):** best method in all six conditions (×0.5 … ×0.025 by 1.5–2.6 dB; under a strong spotlight only narrowly in PSNR, clearly in SSIM).
- **ExDark real-world night photos (no ground truth):** lowest residual noise (1.89); the official Zero-DCE++ has the best NIQE (4.29 vs 4.62).
- **Image-quality-aware update:** measures noise, sharpness, colourfulness, JPEG blocking and resolution and adapts to them; handles small compressed photos (flower JPEG: blocks 2.03 → 1.31, grain 2.83 → 1.29, colourfulness 0.63 → 0.54) and returns full-size results with real (not invented) detail. Neutral on the LOL-style test sets after one fix (its first version capped saturation before denoising and lost 0.27–0.76 dB; the fix was prompted by that test result and chosen on fresh validation data, disclosed in [`eval_P1_T76.md`](eval_P1_T76.md) §9.5–9.6). A final source audit made its sharpening actually act and stopped pre-denoising from hiding noise from the restorer (both chosen on validation with `scripts/validate.py`, §9.8). ExDark NIQE is worse with the update (4.18 → 4.62).
- **Honest weak spots:** not best on LOL-v2 synthetic (16.80 vs 18.04 dB for DL-only); the most new overexposure of our methods (1.11 % avg); Auto mode cannot beat "always hybrid" on validation, so it keeps the hybrid unless another output is clearly better. Details in [`eval_P1_T76.md`](eval_P1_T76.md) §6–9.
- **Two tiny networks:** 10.6k + 144.6k parameters (~0.6 MB); 0.8–1.0 s per 600×400 image, 2.2–2.7 s at 1280×960, 2.5–3.0 s for a 2048-px photo returned full size, on a laptop CPU (range of our final runs).
- Full evaluation, model selection, ablation and failure analysis: [`eval_P1_T76.md`](eval_P1_T76.md) (raw tables: [`results/tables.md`](results/tables.md)).

## Architecture

```
Browser ──► Vercel: LUMINA web app (web/, the team's LUMORA v4 design, static HTML/CSS/JS)
              │  HTTPS + CORS, image as data URL, JSON response (WebP images)
              ▼
            Hugging Face Space: Python engine (app.py + api.py)
              POST /gradio_api/call/restore → classical + deep learning + hybrid (v3) in one call
              the classic Gradio interface keeps running here as a fallback
```

Frontend integration details (endpoint audit, every control, security, limits, tests): [`docs/FRONTEND.md`](docs/FRONTEND.md).

## How it works

```
photo ─► ① illumination map T (max-RGB + guided filter) + quality diagnosis (brightness, lighting spread, noise,
           sharpness, edges, colourfulness / cast, JPEG blockiness, resolution) ─► adaptive parameters + strategy
        ─► (blocky JPEG / noisy classical path) denoise + boundary deblock BEFORE brightening, lower gain cap
        ─► ② two candidates: Retinex re-lighting (classical) │ Zero-DCE++ fine-tuned (DL, quality-aware curve strength)
        ─► ③ per-pixel quality-weighted Laplacian-pyramid fusion + soft highlight roll-off
        ─► ④ adaptive CLAHE + colour correction (the classical path also caps saturation, after its denoising)
        ─► ⑤ restoration U-Net conditioned on a noise-level map derived from the illumination model
        ─► ⑥ adaptive detail recovery AFTER denoising (above-noise detail only, halo clamp; off when sharp or noisy)
        ─► ⑦ full-size output: enhanced at ≤1280 px, returned at the photo's size (≤2048 px) with its own detail
        ─► your intent (Natural / Auto / Max visibility + brightness slider); Auto also picks the method per photo
```

Full diagram with function names: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| Requirement | Implementation |
|---|---|
| Illumination estimation + decomposition, Retinex | `lowlight/classical.py`: `estimate_illumination`, `retinex_enhance`, `solve_gamma` |
| CLAHE, bilateral / guided-filter denoising, colour correction | `classical.clahe`, `classical.denoise`, `classical.colour_correct`, `protect_highlights` |
| Adaptive parameter selection | `lowlight/analysis.py`: `diagnose` → `choose_params` |
| Deep model (Zero-DCE++), fine-tuned | `lowlight/dce.py`, `scripts/train_dce.py` |
| DL restoration conditioned on classical illumination | `lowlight/restore.py`, `scripts/train_restore.py`, `pipeline.noise_map` |
| Adaptive selection / fusion, quality-aware strategy | `pipeline.fusion_weights`, `pyramid_blend`, `choose_dce_scale`; `analysis.strategy`, `detail_strengths`, `quality_labels`; `classical.pre_denoise`, `recover_detail`, `limit_saturation`; `pipeline.run_full`; `api.recommend` (Auto mode) |
| Classical vs DL vs hybrid, ablation, robustness, failures | `scripts/evaluate.py`, `scripts/report.py` → [`eval_P1_T76.md`](eval_P1_T76.md) |

## Human-centred design

- **Explains itself:** "What we noticed / What we did (and why)" in plain language, plus every intermediate map (illumination, gain, fusion weights, noise map, removed noise) in *Under the hood*.
- **Honest:** a *Recoverability* rating (High/Medium/Low) sets expectations; warnings for blown-out or still-grainy areas; a "magenta = pure white" clipping check.
- **No invented content:** curve and Retinex brightening are pixel-wise and monotone; the restorer predicts residual corrections only. No generative model.
- **User control:** *Natural night mood · Auto · Maximum visibility* presets plus a brightness slider.
- **Accessible and private:** phone camera capture, large touch targets, colour-blind-safe (cividis) heatmaps; images processed in memory and temp files purged within an hour.

## Run it

**With Python 3.10+ (tested):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py
```
**Or with Docker:**
```bash
docker build -t lumina . && docker run -p 7860:7860 lumina
```
_(The Dockerfile was not built in our environment because Docker wasn't available; the Python route above is what we tested.)_
Open http://localhost:7860 for the engine and its classic interface. The trained weights (~0.7 MB) are in `weights/`, so no download is needed.

**Web frontend (LUMINA) against the local engine:**
```bash
cd web && LUMINA_API_BASE=http://127.0.0.1:7860 node build.mjs && python -m http.server 8080 --directory dist
```
Open http://localhost:8080. Deployment to Vercel: [`docs/FRONTEND.md`](docs/FRONTEND.md).

## Reproduce training and every number in the evaluation

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py          # LOL, LOL-v2 test splits, 100 fresh LOL-v2 synthetic train pairs (validation only), ExDark, Zero-DCE++ weights (~770 MB, to $DATA_DIR or ./data)
python scripts/train_dce.py              # Zero-DCE++ fine-tune on LOL train split (435/50 seeded split -> results/split.json)
python scripts/train_restore.py          # restoration U-Net v3 (noise-level map; LOL + synthetic dark pairs)
python scripts/train_restore.py --no-map # ablation model
python scripts/evaluate.py               # all methods x all test sets, ablation, robustness, latency (~20 min, 8 cores)
python scripts/report.py                 # results/tables.md, results/figures/, results/failures/
python scripts/validate.py NAME '{...}'  # validation-only design experiments -> results/validation/ (see its docstring)
python -m pytest -q                      # tests
```
Everything runs on CPU, and seeds are fixed. The training logs we got are in `results/train_log_*.json`.

## Repository layout

```
web/                    LUMINA web frontend for Vercel (index.html, ui.js = teammate's UI, lumina.js = engine integration)
api.py                  JSON API used by the web frontend (all three pipelines in one call)
app.py                  Gradio app: classic interface + registers the API
lowlight/classical.py   classical CV primitives
lowlight/analysis.py    image diagnosis, adaptive parameters, plain-language explanations
lowlight/dce.py         Zero-DCE++ model
lowlight/restore.py     noise-aware restoration U-Net
lowlight/pipeline.py    methods (baselines, classical, DL, hybrid) + ablation switches
lowlight/metrics.py     PSNR, SSIM, LPIPS, NIQE, ΔE2000, EPI, noise, clipping
scripts/                download, train, evaluate, report
weights/                trained weights (+ official Zero-DCE++ weights)
results/                metrics, tables, figures, failure cases, training logs, split
docs/                   design document, architecture, in-app "About"
examples/               unseen test images used as demo examples (sources listed)
tests/                  pytest suite (pipeline + API integration tests)
```

## Limitations and ethics (short version)

Detail below the sensor noise floor cannot be recovered. Coloured lighting vs colour cast is ambiguous. Already-clipped highlights stay clipped. The networks were trained on LOL (indoor), so outdoor scenes rely more on the classical stages. **Outputs are for viewing, not evidence:** don't use them for identification or surveillance. Details: [`docs/DESIGN.md` §8–9](docs/DESIGN.md).

## Credits

Zero-DCE++ (Li, Guo & Loy, TPAMI 2021; official weights), LOL (Wei et al., BMVC 2018), LOL-v2 (Yang et al., TIP 2021), ExDark (Loh & Chan, CVIU 2019), LPIPS/NIQE via [pyiqa](https://github.com/chaofengc/IQA-PyTorch). The guided filter, CLAHE, Laplacian-pyramid blending and exposure-fusion weights are re-implemented from the original papers.
