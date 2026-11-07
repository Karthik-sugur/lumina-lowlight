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

<p align="center">
  <a href="https://lumina-t76.vercel.app">
    <img src="https://img.shields.io/badge/%F0%9F%8C%99%20LIVE%20DEMO-lumina--t76.vercel.app-1a1a2e?style=for-the-badge&labelColor=c9a86c&color=1a1a2e" alt="Live demo at lumina-t76.vercel.app"/>
  </a>
</p>

<h1 align="center">Lumina · Adaptive Low-Light Image Restoration</h1>

<p align="center">
  <strong>Computer Vision Hackathon 2026 · Problem P1</strong><br/>
  Robust Low-Light Image Restoration and Enhancement
</p>

<p align="center">
  Team T76 · Indhu S · Karthikeya D S · Dheeraj Chinta · Darsh Chandran<br/>
  <code>1RUA24CSE0170</code> · <code>1RUA24CSE0195</code> · <code>1RUA24CSE0127</code> · <code>1RUA24CSE0115</code>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PSNR-18.19%20dB-0e7c66?style=flat-square" alt="PSNR 18.19 dB"/>
  <img src="https://img.shields.io/badge/SSIM-0.769-0e7c66?style=flat-square" alt="SSIM 0.769"/>
  <img src="https://img.shields.io/badge/LPIPS-0.249-0e7c66?style=flat-square" alt="LPIPS 0.249"/>
  <img src="https://img.shields.io/badge/NIQE-4.25-0e7c66?style=flat-square" alt="NIQE 4.25"/>
  <img src="https://img.shields.io/badge/CPU%20p50-0.81%20s-555?style=flat-square" alt="CPU p50 0.81 s"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/pipeline-classical%20%2B%20DL%20hybrid-2b6cb0?style=flat-square" alt="hybrid pipeline"/>
  <img src="https://img.shields.io/badge/models-~155k%20params%20·%200.6%20MB-2b6cb0?style=flat-square" alt="model size"/>
  <img src="https://img.shields.io/badge/engine-Python%20Gradio-2b6cb0?style=flat-square" alt="Gradio engine"/>
  <img src="https://img.shields.io/badge/frontend-Vercel%20static-2b6cb0?style=flat-square" alt="Vercel frontend"/>
  <img src="https://img.shields.io/badge/runs%20on-CPU%20%2F%20Docker-555?style=flat-square" alt="CPU Docker"/>
</p>

<p align="center">
  <a href="docs/DESIGN.md"><img src="https://img.shields.io/badge/DESIGN-docs-111?style=for-the-badge&labelColor=444" alt="Design"/></a>
  <a href="docs/ARCHITECTURE.md"><img src="https://img.shields.io/badge/ARCHITECTURE-docs-111?style=for-the-badge&labelColor=444" alt="Architecture"/></a>
  <a href="eval_P1_T76.md"><img src="https://img.shields.io/badge/EVALUATION-eval__P1__T76-111?style=for-the-badge&labelColor=444" alt="Evaluation"/></a>
  <a href="docs/FRONTEND.md"><img src="https://img.shields.io/badge/FRONTEND-docs-111?style=for-the-badge&labelColor=444" alt="Frontend"/></a>
  <a href="docs/DEMO_AND_VIVA.md"><img src="https://img.shields.io/badge/DEMO%20%26%20VIVA-docs-111?style=for-the-badge&labelColor=444" alt="Demo and viva"/></a>
</p>

---

Lumina restores dark, noisy, colour-shifted photos with a **hybrid** of classical computer vision and two small neural networks. Every enhancement setting is chosen per image from measurements of that image, and the app explains in plain language what it noticed and what it did — without inventing detail that was never in the sensor data.

> *See what the dark hid, without inventing what wasn't there.*

![Visual comparison](results/figures/5_visual_comparison.jpg)

## Deliverables

| Deliverable | Where |
|---|---|
| Live web app (LUMINA UI) | [lumina-t76.vercel.app](https://lumina-t76.vercel.app) |
| Engine + classic Gradio interface | [darshuuuu-lumina-lowlight.hf.space](https://darshuuuu-lumina-lowlight.hf.space) |
| Source code | [github.com/darshchandran/lumina-lowlight](https://github.com/darshchandran/lumina-lowlight) |
| Comparative evaluation | [`eval_P1_T76.md`](eval_P1_T76.md) |
| Design document | [`docs/DESIGN.md`](docs/DESIGN.md) |

| | |
|---|---|
| Comparative evaluation results | [`eval_P1_T76.md`](eval_P1_T76.md) · raw tables [`results/tables.md`](results/tables.md) |
| Ablation study | [`eval_P1_T76.md` §6](eval_P1_T76.md) · figure [`results/figures/3_ablation.png`](results/figures/3_ablation.png) |
| Failure-case analysis | [`results/failures/`](results/failures/) · index [`results/failures/index.md`](results/failures/index.md) |
| Robustness / darkness sweep | [`eval_P1_T76.md`](eval_P1_T76.md) · figure [`results/figures/2_robustness_illumination.png`](results/figures/2_robustness_illumination.png) |

## System architecture

```
Browser ──► Vercel: LUMINA web app (web/, static HTML/CSS/JS)
              │  HTTPS + CORS, image as data URL, JSON response (WebP)
              ▼
            Hugging Face Space: Python engine (app.py + api.py)
              POST /gradio_api/call/restore
                → classical + deep learning + hybrid in one call
              classic Gradio UI kept as a fallback at the same host
```

```
photo ─► ① illumination map T + quality diagnosis
        ─► ② Retinex (classical) │ Zero-DCE++ (DL) candidates
        ─► ③ quality-weighted Laplacian-pyramid fusion
        ─► ④ adaptive CLAHE + colour correction
        ─► ⑤ restoration U-Net conditioned on a noise-level map
        ─► ⑥ adaptive detail recovery (above-noise only)
        ─► ⑦ full-size return (enhance ≤1280 px, output ≤2048 px)
        ─► intent: Natural / Auto / Max visibility + brightness slider
```

Function-level diagram with module names: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Requirements, trade-offs and ethics: [`docs/DESIGN.md`](docs/DESIGN.md). Frontend endpoint audit and controls: [`docs/FRONTEND.md`](docs/FRONTEND.md).

## Results (paired test average, n = 124)

All test images are unseen (never used for training, validation or tuning). LOL-v2 real is restricted to the **9 pairs** that are not pixel-identical to LOL-v1 training images (86 of its 100 inputs are; see the correction in [`eval_P1_T76.md`](eval_P1_T76.md)). CPU only.

| Metric | Value |
|---|---|
| PSNR ↑ | **18.19 dB** (best of six methods) |
| SSIM ↑ | **0.769** |
| LPIPS ↓ | **0.249** |
| NIQE ↓ | **4.25** |
| Residual noise σ ↓ | **1.50** |
| Colour shift ΔE2000 ↓ | **13.36** |
| Latency (600×400, laptop CPU) | 0.81 s · 2.2 s at 1280×960 · 2.5 s at 2048 px full-size |
| Model size | 10.6k + 144.6k params (~0.6 MB) |

| Average over 3 paired sets (sets weighted equally) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE2000 ↓ |
|---|---|---|---|---|---|---|
| Zero-DCE++ (official weights) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 18.06 |
| Classical only (ours) | 16.74 | 0.618 | 0.477 | 6.92 | 5.38 | 16.51 |
| Deep learning only (ours) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 15.22 |
| **Hybrid (ours)** | **18.19** | **0.769** | **0.249** | **4.25** | **1.50** | **13.36** |

- **LOL-v1 eval15 / clean LOL-v2 real:** hybrid 19.13 / 18.65 dB PSNR, SSIM 0.777 / 0.770, LPIPS 0.207 / 0.283.
- **Darkness sweep (30 unseen references):** best method in all six conditions (×0.5 … ×0.025).
- **ExDark real-world night photos:** lowest residual noise (1.89); official Zero-DCE++ has the best NIQE (4.29 vs 4.62).
- **Honest weak spots:** not best on LOL-v2 synthetic (16.80 vs 18.04 dB for DL-only); most new overexposure of our methods (1.11 % avg).

Full evaluation, ablation, model selection and failure analysis: [`eval_P1_T76.md`](eval_P1_T76.md). Raw tables: [`results/tables.md`](results/tables.md). Experiment artefacts: [`results/`](results/).

## Beyond metrics: a restoration console people can trust

**Lumina** is built so a phone user, a low-vision viewer, or a judge can understand *why* the output looks the way it does. The UI surfaces recoverability, warnings for blown-out or still-grainy areas, plain-language “what we noticed / what we did”, and every intermediate map under the hood.

| Module | What it does |
|---|---|
| **Illumination & Retinex**<br/>`lowlight/classical.py` | Max-RGB prior + guided-filter illumination map `T`; Retinex re-lighting with per-image γ and a gain cap so near-black regions do not explode into confetti noise. |
| **Quality diagnosis**<br/>`lowlight/analysis.py` | Measures brightness, lighting spread, noise, sharpness, edges, colourfulness / cast, JPEG blockiness and resolution; drives adaptive parameters and the plain-language explanation. |
| **Classical path**<br/>`classical.clahe`, `denoise`, `colour_correct` | Adaptive CLAHE, bilateral + guided denoising, partial gray-world colour correction, highlight soft-knee, saturation ceiling after denoising. |
| **Zero-DCE++ (fine-tuned)**<br/>`lowlight/dce.py`, `scripts/train_dce.py` | Curve-based deep brightening; quality-aware curve strength so clipped scenes are not pushed further. |
| **Hybrid fusion**<br/>`pipeline.fusion_weights`, `pyramid_blend` | Per-pixel exposedness × contrast ÷ amplification weights; five-level Laplacian pyramid blend of Retinex and DCE candidates. |
| **Noise-aware restorer**<br/>`lowlight/restore.py`, `pipeline.noise_map` | 145k-param U-Net conditioned on a noise-level map derived from the illumination model; predicts residual corrections only. |
| **Detail recovery & full-size return**<br/>`classical.recover_detail`, `pipeline.run_full` | Re-injects above-noise detail with halo clamp; enhances at ≤1280 px and returns the photo’s own size (≤2048 px) with real (not invented) fine detail. |
| **Auto method pick**<br/>`api.recommend` | Auto mode keeps hybrid unless another output is clearly better on validation-tuned rules. |
| **Evaluation & ablation**<br/>`scripts/evaluate.py`, `scripts/report.py` | Classical vs DL vs hybrid, 12 ablations, darkness sweep, ExDark, latency, failure cases → [`eval_P1_T76.md`](eval_P1_T76.md). |

Every tunable value is chosen on validation data (or is an operational constant / documented assumption). Design decisions and the disclosure trail live in [`docs/DESIGN.md`](docs/DESIGN.md) and [`eval_P1_T76.md` §9](eval_P1_T76.md).

### Human-centred design

- **Explains itself:** “What we noticed / What we did (and why)” plus intermediate maps (illumination, gain, fusion weights, noise map, removed noise).
- **Honest:** *Recoverability* rating (High / Medium / Low); warnings for blown-out or still-grainy areas; magenta clipping check for pure white.
- **No invented content:** curve and Retinex brightening are pixel-wise and monotone; the restorer predicts residual corrections only. No generative model.
- **User control:** *Natural night mood · Auto · Maximum visibility* plus a brightness slider.
- **Accessible and private:** phone camera capture, large touch targets, colour-blind-safe (cividis) heatmaps; images processed in memory and temp files purged within an hour.

## Documentation

| File | Contents |
|---|---|
| [`eval_P1_T76.md`](eval_P1_T76.md) | Full evaluation: metrics, robustness, ablation, quality-aware update, failure analysis |
| [`results/tables.md`](results/tables.md) | Auto-generated metric tables from `scripts/report.py` |
| [`results/failures/`](results/failures/) | Failure-case gallery and index |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Goals, requirements traceability, method, data, ethics |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Function-level diagrams and module interfaces |
| [`docs/FRONTEND.md`](docs/FRONTEND.md) | Vercel ↔ HF Space integration, controls, security, limits |
| [`docs/DEMO_AND_VIVA.md`](docs/DEMO_AND_VIVA.md) | Demo script and viva talking points |
| [`docs/about_app.md`](docs/about_app.md) | In-app “About” copy |
| [`examples/SOURCES.md`](examples/SOURCES.md) | Provenance of demo images |
| [`results/`](results/) | Metrics JSON/CSV, training logs, split, figures, validation experiments |

## Repository layout

<details>
<summary><strong>Folders and what they hold</strong></summary>

```
web/                    LUMINA web frontend for Vercel
                        (index.html, ui.js, lumina.js, build.mjs)
api.py                  JSON API used by the web frontend
app.py                  Gradio app: classic UI + registers the API
lowlight/
  classical.py          classical CV primitives
  analysis.py           diagnosis, adaptive params, explanations
  dce.py                Zero-DCE++ model
  restore.py            noise-aware restoration U-Net
  pipeline.py           methods + ablation switches
  metrics.py            PSNR, SSIM, LPIPS, NIQE, ΔE2000, EPI, …
scripts/                download, train, evaluate, report, validate
weights/                trained + official Zero-DCE++ weights (~0.7 MB)
results/                metrics, tables, figures, failures, train logs
docs/                   design, architecture, frontend, demo notes
examples/               unseen demo images (sources listed)
tests/                  pytest suite (pipeline + API)
```

</details>

## Quick start

**Python 3.10+ (tested path):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python app.py                          # engine + classic Gradio UI → http://localhost:7860
```

**Docker:**

```bash
docker build -t lumina . && docker run -p 7860:7860 lumina
```

Trained weights (~0.7 MB) ship in `weights/`; no separate download is required to run the app.

**Web frontend against a local engine:**

```bash
cd web && LUMINA_API_BASE=http://127.0.0.1:7860 node build.mjs
python -m http.server 8080 --directory dist   # → http://localhost:8080
```

Vercel deployment notes: [`docs/FRONTEND.md`](docs/FRONTEND.md).

### Reproduce training and every evaluation number

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py          # LOL, LOL-v2, ExDark, Zero-DCE++ weights (~770 MB)
python scripts/train_dce.py              # Zero-DCE++ fine-tune → results/split.json
python scripts/train_restore.py          # restoration U-Net v3 (noise-level map)
python scripts/train_restore.py --no-map # ablation model
python scripts/evaluate.py               # all methods × all sets (~20 min, 8 cores)
python scripts/report.py                 # results/tables.md, figures/, failures/
python scripts/validate.py NAME '{...}'  # validation-only design experiments
python -m pytest -q
```

Everything runs on CPU with fixed seeds. Training logs we produced are in `results/train_log_*.json`.

## Dataset

Training and validation use **LOL-v1** `our485` with a seeded 435 / 50 split (`results/split.json`). Test sets are disjoint from that split:

| Set | Role | Size |
|---|---|---|
| LOL-v1 `eval15` | paired test | 15 |
| LOL-v2 real `Test` (clean subset) | paired test; excludes pairs duplicated in `our485` | 9 of 100 |
| LOL-v2 synthetic `Test` | paired test | 100 |
| ExDark (384 px release) | real-world night photos, no reference | 60 sampled |
| Darkness sweep | 30 unseen synthetic references × 6 exposures | 180 / method |

A pixel-hash audit found that 86/100 LOL-v2 real inputs match LOL-v1 training images; those pairs are excluded. Details and the withdrawn earlier numbers are in [`eval_P1_T76.md`](eval_P1_T76.md). Download tooling: `scripts/download_data.py`.

## Limitations

- Detail below the sensor noise floor cannot be recovered; near-black regions can look smudgy or grainy.
- Coloured lighting vs colour cast is ambiguous; sodium / neon scenes sometimes get the wrong white-balance guess.
- Already-clipped highlights stay clipped; new overexposure is our weakest measured point (~1.11 % avg).
- Networks were trained on LOL (indoor) plus synthetic noise; outdoor and heavily compressed night JPEGs lean on the classical stages.
- Not best on LOL-v2 synthetic, where DL-only wins on PSNR / SSIM / LPIPS.
- **Outputs are for viewing, not evidence** — do not use them for identification or surveillance.

Full ethics and failure discussion: [`docs/DESIGN.md` §8–9](docs/DESIGN.md), [`results/failures/`](results/failures/).

## Credits

Zero-DCE++ (Li, Guo & Loy, TPAMI 2021; official weights), LOL (Wei et al., BMVC 2018), LOL-v2 (Yang et al., TIP 2021), ExDark (Loh & Chan, CVIU 2019), LPIPS / NIQE via [pyiqa](https://github.com/chaofengc/IQA-PyTorch). The guided filter, CLAHE, Laplacian-pyramid blending and exposure-fusion weights are re-implemented from the original papers.

<p align="center">
  <br/>
  Prototype built for the <strong>Computer Vision Hackathon 2026</strong> by Team T76.
</p>
