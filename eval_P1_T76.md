1RUA24CSE0170, 1RUA24CSE0195, 1RUA24CSE0127, 1RUA24CSE0115

# Evaluation: P1 Robust Low-Light Image Restoration and Enhancement (Team T76)

**Team T76:** Indhu S (1RUA24CSE0170), Karthikeya D S (1RUA24CSE0195), Dheeraj Chinta (1RUA24CSE0127), Darsh Chandran (1RUA24CSE0115)
**System:** Lumina, an adaptive hybrid of classical CV and deep learning · **Live app:** https://lumina-t76.vercel.app (engine and classic interface: https://darshuuuu-lumina-lowlight.hf.space) · **Code:** https://github.com/darshchandran/lumina-lowlight

Every number below was produced by `scripts/evaluate.py` and `scripts/report.py`. Per-image values are in `results/per_image.csv`, aggregates in `results/metrics.json`, and the auto-generated tables in `results/tables.md`; the tables in this file are copied from that file programmatically. Metrics that were not measured are marked **n/a** with the reason. Nothing is estimated.

> **Correction (2026-11-07): test leakage found and removed.** LOL-v2 real re-packages LOL-v1. A pixel-hash check showed that **86 of its 100 test inputs and 91 of its 100 references are identical to LOL-v1 `our485` images, which are our training/validation data.** Our earlier submission reported LOL-v2 real on all 100 pairs (19.17 dB) and built the robustness sweep from those references; those numbers were partly measured on training images and are **withdrawn**. This version evaluates only the 9 LOL-v2 real pairs whose input and reference are both absent from `our485` (on those 9, the earlier pipeline scored 18.67 dB, not 19.17), moves the robustness sweep to LOL-v2 *synthetic* references, and runs the ablation on eval15 + LOL-v2 synthetic. eval15, LOL-v2 synthetic and ExDark share no image with `our485`. A demo photo that was a training scene (LOL-v2 00748) was replaced.

## 0. Summary

| | |
|---|---|
| Deployed model | Hybrid pipeline with Zero-DCE++ (fine-tuned), **restoration U-Net v3** (noise-map conditioned) and the **image-quality-aware strategy** (§9) |
| Model and design selection | On **validation data only** (the 50 held-out `our485` pairs, a compressed proxy of them, and 100 fresh LOL-v2 synthetic *training* pairs). One decision, removing the hybrid's saturation ceiling, was **prompted by a test-set observation** and then made on validation data by a rule fixed in advance (§9.5–9.6) |
| Paired test sets, averaged (124 pairs, sets weighted equally) | Hybrid **18.23 dB PSNR · 0.771 SSIM · 0.248 LPIPS · 4.24 NIQE**, best of six methods on 7 of 8 metrics; it has the most new overexposure of our methods (1.08 %) |
| Per set | Hybrid best on LOL-v1 eval15 (19.18 dB / 0.779 SSIM / 0.207 LPIPS) and the clean LOL-v2 real subset (18.67 dB / 0.772 SSIM / 0.283 LPIPS); **not** best on LOL-v2 synthetic, where deep learning alone wins (18.04 vs 16.84 dB) |
| Unseen real-world night photos (ExDark, no reference) | Hybrid has the lowest residual noise (1.85); the official Zero-DCE++ has the best NIQE (4.29 vs hybrid 4.54) |
| Robustness (darkness sweep, 30 unseen references) | Hybrid best in all six conditions, from ×0.5 to ×0.025 exposure and under a strong spotlight (there only narrowly in PSNR: 14.15 vs 14.13 dB, but 0.673 vs 0.494 SSIM) |
| Quality-aware update, honestly | Returns full-size results and handles small compressed photos (flower JPEG: blocks 2.03 → 1.33, grain 2.83 → 1.40, colourfulness 0.63 → 0.55). On the paired test sets it is level with the pipeline before the update (eval15 19.18 = 19.18 dB, clean LOL-v2 real 18.67 = 18.67, synthetic 16.84 vs 16.88), after a fix to its saturation ceiling that was **prompted by a test-set observation**, then chosen on fresh validation data by a rule fixed in advance (§9.5–9.6). On ExDark: less grain (2.10 → 1.85) but worse NIQE (4.18 → 4.54) |
| Latency (laptop CPU) | Hybrid 1001 ms at 600×400, 2.7 s at 1280×960, 3.0 s for a 2048 px photo returned at full size (§2.4) |

## 1. Setup

**Test data:** all disjoint from training and validation (checked by pixel hash for LOL-v2).

| Set | Type | Images | Reference? |
|---|---|---|---|
| LOL-v1 `eval15` | real low/normal-light pairs | 15 | yes: **paired metrics** |
| LOL-v2 real `Test`, **clean subset** | real pairs not duplicated in LOL-v1 `our485` (00705–00707, 00753–00756, 00767–00768; 3 scenes) | 9 of 100 | yes: **paired metrics** |
| LOL-v2 synthetic `Test` | synthetically darkened photos | 100 | yes: **paired metrics** |
| ExDark (384-px release, letterbox trimmed) | real-world night photos, mostly compressed JPEGs | 60 (seed 0) | **no: no-reference metrics only** |
| Robustness sweep | 30 unseen LOL-v2 *synthetic* references degraded physically at 6 conditions | 180 per method | yes: paired |

**Small-sample caveat:** the clean LOL-v2 real subset has only 9 pairs from 3 scenes, so its averages are noisy; it is kept because it is the only clean real-capture set besides eval15.

**Training / validation:** LOL-v1 `our485` split 435 / 50 (seed 0, `results/split.json`). The restoration U-Net also trains on one synthetic degradation of each *training* reference. Validation = the 50 held-out pairs, plus one synthetic degradation of each (model selection) and a 384-px JPEG-q70 version of each (the compressed proxy used for the quality-aware update, §9.3).

**Methods:** two naive baselines (histogram equalisation, CLAHE), the published Zero-DCE++ weights (zero-shot), and our **classical-only**, **deep-learning-only** (fine-tuned Zero-DCE++) and **hybrid** pipelines.

**Metric families**

| Paired-reference (need ground truth) | No-reference (any image) |
|---|---|
| PSNR, SSIM (Gaussian 11×11, σ 1.5, RGB), LPIPS (AlexNet), CIEDE2000 ΔE (colour shift), EPI (edge preservation: Laplacian correlation with the reference), new-overexposure % (clipped in output but not in reference), brightness error, **detail ratio** (sharpness index of the output ÷ that of the reference) | NIQE, residual noise σ (Immerkær, 0–255), clipped-pixel %, colour cast (Lab chroma of the mean mid-tone colour), mean brightness, **colourfulness** (Lab chroma ÷ lightness), **JPEG blockiness** (8×8 grid step energy; 1 = none), **sharpness index** |

**Not available:** PSNR, SSIM, LPIPS, ΔE, EPI and detail ratio on ExDark (**n/a**, because ExDark has no ground truth). FID was not computed (not required for P1; no distribution-level reference set).

**Hardware:** AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only (laptop). Latency is measured in one process after warm-up (§2.4).

## 2. Evaluation metrics (in the order of the problem statement)

Column order in every table: PSNR → SSIM → LPIPS → NIQE → latency → noise suppression → edge preservation, then the advanced-requirement checks (colour shift, overexposure). Best value per column is **bold**.

### 2.1–2.5 Per test set

#### LOL-v1 eval15 (15 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **7** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 14 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 242 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.71 | 0.572 | 0.470 | 7.24 | 245 | 5.98 | 0.415 | 14.85 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 246 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **19.18** | **0.779** | **0.207** | **4.11** | 1001 | **1.19** | **0.622** | **11.22** | 0.08 |

#### LOL-v2 real test (clean subset) (9 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.16 | 0.279 | 0.759 | 10.13 | **7** | 21.42 | 0.389 | 20.78 | 4.36 |
| CLAHE only (baseline) | 10.91 | 0.294 | 0.586 | 9.02 | 14 | 3.24 | 0.484 | 25.59 | **0.00** |
| Zero-DCE++ official (baseline) | 15.30 | 0.396 | 0.486 | 9.85 | 242 | 6.58 | 0.476 | 19.75 | **0.00** |
| Classical only (ours) | 16.95 | 0.547 | 0.639 | 8.57 | 245 | 7.24 | 0.368 | 20.86 | **0.00** |
| Deep learning only (ours, fine-tuned) | 16.29 | 0.375 | 0.541 | 10.15 | 246 | 9.31 | 0.454 | 18.95 | **0.00** |
| Hybrid (ours) | **18.67** | **0.772** | **0.283** | **4.22** | 1001 | **1.19** | **0.528** | **14.74** | 0.00 |

#### LOL-v2 synthetic test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **7** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 14 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 242 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.62 | 0.745 | 0.290 | 4.80 | 245 | 2.87 | 0.807 | 13.76 | **0.26** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 246 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.84 | 0.762 | 0.255 | **4.40** | 1001 | **1.88** | 0.817 | 14.08 | 3.17 |

#### Average over the three paired test sets (124 pairs, sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.04 | 0.479 | 0.551 | 7.79 | 13.05 | 0.530 | 17.72 | 4.39 |
| CLAHE only (baseline) | 10.73 | 0.391 | 0.485 | 7.03 | 3.05 | 0.591 | 27.43 | 0.13 |
| Zero-DCE++ official (baseline) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 0.647 | 18.06 | 0.13 |
| Classical only (ours) | 16.76 | 0.621 | 0.467 | 6.87 | 5.36 | 0.530 | 16.49 | **0.09** |
| Deep learning only (ours, fine-tuned) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 0.628 | 15.22 | 0.40 |
| Hybrid (ours) | **18.23** | **0.771** | **0.248** | **4.24** | **1.42** | **0.655** | **13.34** | 1.08 |

#### ExDark (real-world, no reference) (60 real-world night photos, no ground truth)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.82 | 2.64 | 0.55 | 14.5 | 0.38 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.54 | 1.85 | 3.24 | 14.0 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

PSNR / SSIM / LPIPS / ΔE / EPI for ExDark: **n/a** (no ground truth). Colour cast on night scenes is partly intentional (sodium or neon light), so it is reported, not ranked.

### 2.4 Processing latency and model size

| Method | 600×400 (ms) | 1280×960 (ms) | 2048×1536 upload, full-size output (ms) |
|---|---|---|---|
| Histogram equalisation (baseline) | 7 | 36 | 391 |
| CLAHE only (baseline) | 14 | 49 | 406 |
| Zero-DCE++ official (baseline) | 242 | 549 | 905 |
| Classical only (ours) | 245 | 1396 | 1767 |
| Deep learning only (ours, fine-tuned) | 246 | 522 | 926 |
| Hybrid (ours) | 1001 | 2698 | 3022 |

| Model | Parameters | File size |
|---|---|---|
| Zero-DCE++ (fine-tuned) | 10,561 | 51.0 KB |
| Restoration U-Net | 144,579 | 574.6 KB |

- Laptop CPU above. The hybrid spends most of its time in the restoration U-Net and Zero-DCE++; per-stage timings are shown live in the app. The 2048-px column is what the app now does for a large photo: enhance at ≤ 1280 px, then return the photo's own size with detail transfer (§9). Returning full-size results adds about 324 ms to the hybrid compared with the 1280-px run. Timings on this laptop varied by up to ~25 % between our two final evaluation runs for every method, including the unchanged baselines (e.g. hybrid 837 vs 1001 ms at 600×400), so treat them as ±25 %.
- **Hosted app** (Vercel frontend → free Hugging Face CPU engine), measured in the browser after the update: measured on the production URL on 2026-11-07 (engine warm, final version): a 600×400 sample, all three pipelines, **4.3 s** from click to result on desktop (0.96 s on the engine, the rest is upload and the free Space's queue) and **2.4 s** on a 375 px mobile profile (0.86 s on the engine); a **2048×1365** photo **9.2 s**, 4.19 s on the engine, returned at 2048×1365. The free Space sleeps after 48 h idle; a cold start was not measured.

### 2.5 Noise suppression and edge preservation (reading the tables)
- **Noise:** the hybrid leaves the least residual noise on every paired set (σ 1.19 / 1.19 / 1.88) and on ExDark (1.85). Deep learning only leaves 3.70–9.31, because a pixel-wise curve cannot denoise; classical-only leaves 2.87–7.24.
- **Edges:** the hybrid has the best EPI on eval15 and the clean LOL-v2 real subset, and the best average (0.655). On LOL-v2 synthetic, the published Zero-DCE++ (0.921) and our DL-only (0.907) beat it (0.817): those images are only mildly noisy, so a curve that does no smoothing keeps edges best, while restoration slightly smooths fine texture.
- **Detail ratio** (sharpness relative to the reference, 1 = equal): hybrid 1.01 on eval15, 1.12 on clean LOL-v2 real, 1.12 on synthetic, so it neither over-smooths nor over-sharpens on average; the other methods read 1.1–3.4 because they amplify noise, which the index partly counts as detail.

## 3. Classical-only vs deep-learning-only vs hybrid

| Paired average | PSNR | SSIM | LPIPS | NIQE | Noise σ | EPI | ΔE |
|---|---|---|---|---|---|---|---|
| Classical only | 16.76 | 0.621 | 0.467 | 6.87 | 5.36 | 0.530 | 16.49 |
| Deep learning only | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 0.628 | 15.22 |
| **Hybrid** | **18.23** | **0.771** | **0.248** | **4.24** | **1.42** | **0.655** | **13.34** |

- **Classical only:** good exposure control and the fewest blown pixels, but noisy (σ 5.4 on average) and soft on edges (lowest EPI among ours). Under a strong spotlight it is level with the hybrid in PSNR (14.13 vs 14.15 dB), because it never over-brightens, but far behind in SSIM.
- **Deep learning only:** fine-tuning lifted Zero-DCE++ from 14.32 to 16.81 dB on validation, and it is the best method on LOL-v2 synthetic (18.04 dB). But it learned LOL's "always brighten a lot" behaviour: on mildly dark inputs it blows out (10.22 dB at ×0.5 exposure), and as a pixel-wise curve it cannot remove noise.
- **Hybrid:** best on eval15 and clean LOL-v2 real on every paired metric except overexposure, and best on the paired average; SSIM is 0.15–0.18 higher than either single approach. It **is not best on LOL-v2 synthetic** (16.84 vs 18.04 dB for DL-only): those images are bright and lightly noisy, and the restorer's LOL-learned brightening and smoothing hurt there.

## 4. Generalisation across illumination conditions (robustness sweep)

_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 16.48 / 0.561 | 16.06 / 0.500 | 15.39 / 0.401 | 14.73 / 0.317 | 13.79 / 0.231 | 11.42 / 0.392 |
| Zero-DCE++ official (baseline) | 15.14 / 0.672 | 15.79 / 0.611 | 15.66 / 0.491 | 14.68 / 0.374 | 12.97 / 0.249 | 13.49 / 0.379 |
| Classical only (ours) | 16.84 / 0.707 | 15.58 / 0.637 | 15.15 / 0.543 | 15.08 / 0.485 | 14.99 / 0.425 | 14.13 / 0.494 |
| Deep learning only (ours, fine-tuned) | 10.22 / 0.602 | 11.33 / 0.565 | 13.04 / 0.474 | 14.15 / 0.368 | 14.12 / 0.242 | 12.71 / 0.359 |
| Hybrid (ours) | **19.40** / 0.820 | **18.02** / 0.800 | **17.11** / 0.762 | **16.76** / 0.717 | **16.85** / 0.658 | **14.15** / 0.673 |

_Cells: PSNR (dB) / SSIM._

- The hybrid is the best method at every exposure from ×0.5 down to ×0.025, by 1.5–2.6 dB PSNR and 0.11–0.23 SSIM over the next best method.
- Under the **spotlight** it is only level with classical-only in PSNR (14.15 vs 14.13 dB; it brightens the dark surround more than the reference) but far ahead in SSIM (0.673 vs 0.494).
- **Caveat:** the restorer's synthetic training pairs use the same physical noise model (on *different* images, the LOL-v1 training references). The exposure sweep is therefore not fully out-of-distribution; the spotlight condition and ExDark are.
- **Changed from the earlier version:** the sweep previously used LOL-v2 real references, 91 % of which are LOL-v1 training references (see the correction at the top); those numbers are withdrawn.

## 5. Model selection: v2 vs v3 restorer (validation only)

Full-pipeline PSNR / SSIM on the 50 held-out validation pairs (real) and their synthetic degradations (×0.02–×0.5):

| Restorer | Real | Synthetic | Real − synthetic gap |
|---|---|---|---|
| v2 (LOL pairs only) | **19.65 / 0.839** | 17.54 / 0.766 | 2.11 dB |
| v3 (LOL + synthetic pairs), **selected** | 19.41 / 0.824 | **19.90 / 0.815** | **0.49 dB** |

v3 trades 0.24 dB on real LOL-like images for +2.36 dB across a much wider darkness range, and its real/synthetic gap is about 4× smaller. We weighted generalisation over a quarter-dB on one distribution because judges and users upload arbitrary photos. (These are the pre-update validation numbers; the selection was made then and not revisited.)

## 6. Ablation study (hybrid with one component removed)

Run on LOL-v1 eval15 and LOL-v2 synthetic (the clean LOL-v2 real subset is too small for stable differences). The last two rows switch off the image-quality-aware update.

#### LOL-v1 eval15
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 19.18 | +0.00 | 0.779 | 0.207 | 4.11 | 1.19 | 11.22 | 0.08 |
| w/o adaptive parameters (fixed) | 17.95 | -1.23 | 0.773 | 0.233 | 4.62 | 0.99 | 12.33 | 0.00 |
| w/o Zero-DCE++ branch (Retinex only) | 18.98 | -0.20 | 0.774 | 0.210 | **3.97** | 1.27 | 11.27 | 0.04 |
| w/o Retinex branch (Zero-DCE++ only) | 18.42 | -0.75 | 0.772 | 0.224 | 4.45 | 1.08 | 11.94 | 0.97 |
| w/o quality-aware curve scaling | 19.19 | +0.01 | 0.779 | 0.207 | 4.15 | 1.16 | 11.34 | 0.10 |
| w/o DL restoration (classical denoise) | 17.25 | -1.92 | 0.651 | 0.375 | 6.77 | 4.40 | 14.22 | 0.00 |
| w/o noise-map conditioning | 18.81 | -0.37 | 0.776 | 0.219 | 4.21 | 0.85 | 11.57 | 0.02 |
| w/o CLAHE | **19.50** | **+0.32** | 0.778 | 0.230 | 5.14 | **0.66** | **10.96** | **0.00** |
| w/o colour correction | 19.09 | -0.08 | 0.779 | 0.213 | 4.18 | 1.13 | 11.18 | 0.05 |
| w/o highlight protection | 19.18 | +0.01 | **0.780** | **0.206** | 4.11 | 1.18 | 11.20 | 0.08 |
| w/o detail recovery | 19.18 | +0.00 | 0.779 | 0.206 | 4.05 | 1.18 | 11.22 | 0.08 |
| w/o quality-aware strategy | 19.18 | +0.00 | 0.779 | 0.207 | 4.11 | 1.19 | 11.22 | 0.08 |
| previous pipeline (before quality-aware update) | 19.18 | +0.00 | 0.779 | 0.206 | 4.05 | 1.18 | 11.22 | 0.08 |

#### LOL-v2 synthetic test
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 16.84 | +0.00 | 0.762 | 0.255 | 4.40 | 1.88 | 14.08 | 3.17 |
| w/o adaptive parameters (fixed) | 17.44 | +0.60 | 0.782 | 0.252 | **4.36** | 1.79 | 13.85 | 3.83 |
| w/o Zero-DCE++ branch (Retinex only) | 17.25 | +0.41 | 0.767 | 0.246 | 4.39 | 1.82 | 13.51 | 2.97 |
| w/o Retinex branch (Zero-DCE++ only) | 15.99 | -0.84 | 0.731 | 0.282 | 4.61 | 1.73 | 15.36 | 3.84 |
| w/o quality-aware curve scaling | **17.82** | **+0.99** | **0.792** | **0.243** | 4.39 | 2.14 | **12.56** | 3.69 |
| w/o DL restoration (classical denoise) | 16.64 | -0.20 | 0.743 | 0.269 | 5.07 | 3.12 | 13.96 | **0.22** |
| w/o noise-map conditioning | 16.58 | -0.26 | 0.747 | 0.315 | 4.82 | 1.05 | 14.97 | 2.59 |
| w/o CLAHE | 16.11 | -0.73 | 0.719 | 0.279 | 5.19 | **1.04** | 14.88 | 4.27 |
| w/o colour correction | 16.90 | +0.07 | 0.762 | 0.255 | 4.40 | 1.87 | 13.86 | 3.30 |
| w/o highlight protection | 16.93 | +0.09 | 0.763 | 0.253 | 4.39 | 1.85 | 14.01 | 3.09 |
| w/o detail recovery | 16.83 | -0.00 | 0.761 | 0.256 | 4.41 | 1.86 | 14.08 | 3.17 |
| w/o quality-aware strategy | 16.88 | +0.04 | 0.765 | 0.252 | 4.38 | 1.91 | 14.09 | 3.16 |
| previous pipeline (before quality-aware update) | 16.88 | +0.04 | 0.765 | 0.253 | 4.38 | 1.89 | 14.09 | 3.16 |

![Ablation](results/figures/3_ablation.png)

**Interpretation**
- **DL restoration is the largest single contribution on real captures:** replacing it with classical denoising costs 1.92 dB on eval15, raises LPIPS by 0.17 and NIQE by 2.7, and leaves 3.7× more noise. On LOL-v2 synthetic the cost is smaller (−0.20 dB) but NIQE still rises by 0.67.
- **Fusion:** removing the Retinex branch costs 0.75 / 0.84 dB and raises overexposure (0.08 → 0.97 %, 3.17 → 3.84 %). Removing the Zero-DCE++ branch is mixed (−0.20 / +0.41 dB), so **the DL candidate's contribution to the fused image is small**.
- **Adaptive parameters:** fixed parameters cost 1.23 dB on eval15 but gain 0.60 dB on LOL-v2 synthetic; NIQE favours the adaptive version on eval15 (4.11 vs 4.62) and is level on synthetic (4.40 vs 4.36). Fixed γ brightens less, and PSNR is dominated by agreement on overall exposure.
- **Quality-aware curve scaling:** +0.01 / +0.99 dB when removed, i.e. neutral on eval15 and it **hurts on LOL-v2 synthetic** (it helped on validation and on the now-withdrawn LOL-v2 real numbers). Reported as is.
- **CLAHE:** removing it gains 0.32 dB on eval15 but loses 0.73 dB on synthetic, and worsens LPIPS (0.207 → 0.230) and NIQE (4.11 → 5.14) on eval15: a pixel-fidelity vs perceived-contrast trade-off we keep deliberately.
- **Noise-map conditioning:** −0.37 / −0.26 dB and worse LPIPS without it (0.207 → 0.219, 0.255 → 0.315): on these clean test sets the map helps on both. On validation it was a near tie (19.65 vs 19.66 dB), so we still describe its benefit as small.
- **Colour correction and highlight protection:** small or no effect (−0.08 / +0.07 dB and +0.01 / +0.09 dB when removed). The soft-knee runs *before* the restoration network, which can re-clip highlights (§7).
- **Detail recovery:** no measurable effect on these sets (±0.00 dB), by design: they are noisy LOL-style images where the stage backs off (§9).
- **Quality-aware strategy:** removing it changes nothing on eval15 (+0.00 dB) and gains 0.04 dB on synthetic. On these clean PNG sets the update is neutral by design (no pre-denoising, no deblocking, detail recovery backs off on noise); it targets compressed and large photos (§9). Its first version lost 0.27 dB here through a saturation ceiling that has since been removed from the hybrid (§9.5–9.6).

## 7. Overexposure, colour shift and detail loss (advanced requirement 4)
- **Colour shift:** the hybrid has the lowest ΔE2000 on eval15 (11.22), the clean LOL-v2 real subset (14.74) and on average (13.34). On LOL-v2 synthetic, DL-only is better (12.73 vs 14.08).
- **Oversaturation** (new check): on ExDark the hybrid's colourfulness is 0.38 (0.39 before the update), classical-only 0.41 and the official Zero-DCE++ 0.53. The first version of the update capped the hybrid at 0.28, but at a cost on the paired sets (§9.5).
- **Detail loss:** the hybrid has the best average EPI (0.655) and a detail ratio of 1.01–1.12. Fine periodic textures can still turn into moiré, and near-black flat areas keep low-frequency blotches (failures 3 and 6).
- **Overexposure is still our weakest point:** 1.08 % newly clipped pixels on the paired average (3.17 % on LOL-v2 synthetic) and 3.24 % clipped on ExDark. The highlight-protection ablation shows the soft-knee has little effect, because the restoration network runs after it and can re-clip. **Planned fix:** apply the soft-knee after restoration and add a highlight-preservation term to the restorer's loss.

## 8. Failure-case analysis

Panels are auto-selected by `scripts/report.py`: the 3 lowest-PSNR hybrid results on the real paired sets (eval15 + clean LOL-v2 real; input | hybrid | reference) and the 3 highest-NIQE results on ExDark (input | hybrid). Diagnoses come from the pipeline's own analysis (`results/failures/index.md`).

| # | Panel | Set / image | PSNR | SSIM | NIQE | Pipeline diagnosis of the input |
|---|---|---|---|---|---|---|
| 1 | [failure_1.jpg](results/failures/failure_1.jpg) | LOL-v1 eval15 179.png | 12.69 | 0.66 | 3.45 | brightness 0.078, noise σ 1.4, spread 1.3 stops, cast 7 (red) |
| 2 | [failure_2.jpg](results/failures/failure_2.jpg) | LOL-v1 eval15 780.png | 14.30 | 0.65 | 3.59 | brightness 0.041, noise σ 1.0, spread 1.7 stops, cast 1 (blue) |
| 3 | [failure_3.jpg](results/failures/failure_3.jpg) | LOL-v2 real test (clean subset) 00753.png | 14.93 | 0.72 | 4.07 | brightness 0.042, noise σ 1.3, spread 1.7 stops, cast 3 (cyan) |
| 4 | [failure_4.jpg](results/failures/failure_4.jpg) | ExDark 2015_04779.jpg | n/a | n/a | 7.98 | brightness 0.086, noise σ 0.3, spread 4.9 stops, cast 2 (yellow) |
| 5 | [failure_5.jpg](results/failures/failure_5.jpg) | ExDark 2015_06796.jpg | n/a | n/a | 7.89 | brightness 0.176, noise σ 0.4, spread 4.6 stops, cast 26 (orange), coloured lights |
| 6 | [failure_6.jpg](results/failures/failure_6.jpg) | ExDark 2015_01791.jpg | n/a | n/a | 6.73 | brightness 0.087, noise σ 0.2, spread 3.7 stops, cast 4 (yellow) |

**Failures 1–3: brighter than the reference.** Hanging tassels on wood, an empty hall, a climbing wall.
- *What failed:* PSNR 12.7–14.9 dB, the lowest on the real paired sets.
- *Why (measured):* our outputs are much brighter than these references (brightness error 0.21 / 0.16 / 0.18 on a 0–1 scale); colours and structure are close (SSIM 0.66 / 0.65 / 0.72). PSNR is dominated by agreement on overall exposure, and our adaptive target (0.42) is brighter than these particular references.
- *Secondary, visible:* on the climbing wall (3) the output is paler and the darkest left side keeps **low-frequency blotches**: the restorer removes fine grain, but large-scale noise survives on flat surfaces.
- *What it tells us:* exposure is a convention; the app lets users choose *Natural night mood* or a lower brightness, and we report NIQE and LPIPS beside PSNR.

**Failure 4: a low-key studio photo treated as underexposed.** A white cup lit against pure black.
- *What failed:* the black background is lifted and shows **purple/green blotches and blocks** around the handle and in the reflection (NIQE 7.98).
- *Likely cause:* 4.9 stops of lighting spread lowers the target only slightly, so the near-black background still gets a large gain. Its blockiness (1.18) is below the deblocking threshold (1.3), so it is not deblocked, yet even the reduced gain cap (14×) amplifies JPEG blocks and subsampled chroma in the black areas.
- *What it tells us:* "dark" ≠ "underexposed"; a low-key detector, and deblocking that scales with the planned gain rather than a fixed threshold, are the fixes.

**Failure 5: coloured light neutralised.** A foggy path lit by an orange sodium lamp.
- *What failed:* the fog becomes grey-beige and flat; the warm atmosphere is lost (NIQE 7.89).
- *Likely cause:* coloured lights are detected, so white balance is reduced, but a cast of 26 still drives a 24 %-strength correction; and lifting the dense fog (the photo was also deblocked, blockiness 1.6) removes the contrast that made it look like night. This is the coloured-light vs colour-cast ambiguity listed in our limitations.
- *What it tells us:* when the whole scene is lit by one coloured source, white balance should be off, not reduced.

**Failure 6: camouflage-like colour noise.** A room lit by a laptop screen.
- *What failed:* the dark wall and curtain on the left turn into **large green/brown blotches** (NIQE 6.73).
- *Likely cause:* spatially correlated, compressed noise (phone/JPEG) in near-black areas, amplified by a large gain. The restorer was trained on independent per-pixel sensor noise, and our noise estimator measures high-frequency noise only, so the noise map under-reads it.
- *What it tells us:* a 145k-parameter supervised restorer is only as general as its noise model; real compressed captures are the clear next training data.

**What the failures tell us about the approach**
1. **Exposure is a convention.** The largest PSNR losses come from brightening more than a particular reference. An adaptive target fixes "too dark" but cannot know the photographer's intent; hence the intent presets, and NIQE and LPIPS beside PSNR.
2. **Decide on clean signals.** Every measurement that drives a decision must be taken after noise is removed: the saturation ceiling (§9.5) is the counter-example we found.
3. **A small supervised restorer is only as general as its noise model.** It cleans Poisson/Gaussian sensor noise well, but not spatially correlated, compressed noise (failures 4 and 6).

## 9. Image-quality-aware adaptive enhancement (update of 2026-11-07)

### 9.1 What triggered it: two real outputs, diagnosed by measurement

| Photo | What users saw | Measured cause |
|---|---|---|
| Small flower photo (≈300 px, compressed) | pixelated, oversaturated | (1) Retinex divides by max(R,G,B), so chroma *noise* in near-black pixels became saturated colour: mean chroma 4.7 in the input → 36.7 after brightening; (2) a darkness-based saturation boost (×1.13) and CLAHE added more; (3) an 11–21× local gain on a 300-px JPEG made invisible 8×8 blocks visible |
| Large living-room photo (3872 px original) | soft | (1) the restoration U-Net removes fine texture along with noise (sharpness index 0.050 → 0.021) and no stage put real detail back; (2) the engine returned **1280 px** for a 3872-px photo, which the browser then upscaled |

### 9.2 What changed (design in `docs/DESIGN.md` §5.6)

Quality analysis before any decision (illumination + spread, noise before and after brightening, sharpness after noise removal, edge strength, colourfulness and cast, JPEG blockiness, resolution) → operations chosen from it: chroma denoise and **8×8-boundary deblocking before brightening** for blocky JPEGs (and for noisy inputs on the classical path), a lower **gain cap** (14× instead of 25×) for small or compressed photos, a **saturation ceiling** (colourfulness 0.55; classical path only since §9.6), **adaptive detail recovery after noise removal** (only detail above the measured noise, halo clamp, off when already sharp or still noisy), **full-size output** with structure-gated detail transfer from the photo itself, and **Auto mode** that picks Classical / DL / Hybrid per photo. Every decision and its reason is shown in the app (*Analysis → Enhancement breakdown*).

### 9.3 How the constants were chosen (validation only)

No test image was used to choose anything below. Two validation sets: the 50 held-out `our485` pairs, and a **compressed proxy** of them (384 px, JPEG quality 70; stands in for small real-world JPEGs, all 50 inputs blocky).

| Decision | Validation evidence | Chosen |
|---|---|---|
| Core re-injected detail at the noise of the *pre-restoration* image (not the residual noise) | with residual-noise coring the stage re-injected grain: SSIM 0.8124 → 0.8082, noise σ 1.30 → 1.58 | reference-noise coring (stage becomes neutral on noisy LOL: SSIM 0.8243 → 0.8237) |
| Pre-denoise before the restoration network | on clean-but-noisy PNGs it lowered SSIM 0.824 → 0.812 (the network was trained on raw sensor noise) | only for blocky JPEGs (hybrid); by noise level on the classical path |
| Deblocking scope (compressed proxy) | whole-image: blockiness 1.93 → 1.34 but detail 0.69× reference, LPIPS 0.270 → 0.310, NIQE 4.76 → 6.78 · boundary-only: blockiness → 1.16, detail 0.83×, LPIPS 0.282, NIQE 5.87 | boundary-only |
| CLAHE on deblocked JPEGs | full strength: PSNR 18.70 → 17.86, LPIPS 0.282 → 0.303 | 40 % |
| Fine sharpening on deblocked JPEGs | 0 vs 1.0: LPIPS 0.282 vs 0.281 (no difference) | off (simpler, safer) |
| Full-size detail transfer (process at half size, compare at full size) | upscale only → transfer: hybrid SSIM 0.741 → 0.747, EPI 0.289 → 0.412; classical SSIM 0.642 → 0.653, EPI 0.261 → 0.491 | on, structure-gated (ungated transfer added speckle on a smooth, heavily noisy synthetic photo: LPIPS 0.383 → 0.445; gated: 0.382) |

**Validation result of the whole update, final version** (our485 validation, 50 pairs):

| Method | Version | PSNR | SSIM | ΔE2000 | Colourfulness vs reference | Residual noise σ |
|---|---|---|---|---|---|---|
| Hybrid | before | 19.41 | 0.824 | 10.70 | 1.33× | 1.20 |
| Hybrid | after | 19.41 | 0.823 | 10.70 | 1.33× | 1.21 |
| Classical | before | 16.54 | 0.570 | 16.05 | 3.14× | 6.12 |
| Classical | after | 16.87 | 0.595 | 15.37 | 2.78× | 5.98 |
| Hybrid, compressed proxy | before | 18.55 | 0.739 | – | 0.27 (absolute) | – |
| Hybrid, compressed proxy | after | 18.69 | 0.777 | – | 0.24 (absolute) | – |

On clean, noisy PNG photos the final hybrid is practically unchanged (it neither pre-denoises them nor caps their saturation, and detail recovery backs off on noise). Compressed proxy LPIPS 0.270 → 0.283 and NIQE 4.76 → 5.87 (worse), blockiness 1.93 → 1.16 (better): deblocking trades a little perceptual sharpness for removing blocks. (The first version's hybrid row read 19.37 / 0.824 / 10.52 / 1.19×, with the saturation ceiling; see §9.5–9.6.)

### 9.4 The two reference photos, before vs after

Crops taken from the screenshots (322×237 and 597×355), plus the flower re-saved as a 300×221 JPEG at quality 55 to emulate its original. No ground truth exists, so these are no-reference measurements of the hybrid output (classical in brackets).

| Photo | Version | Colourfulness | Blockiness | Residual noise σ | Detail (sharpness index) |
|---|---|---|---|---|---|
| Flower (screenshot crop) | before | 0.59 (0.78) | 1.19 (1.20) | 3.22 (3.65) | 0.084 (0.088) |
| | after | 0.59 (**0.55**) | 1.17 (1.18) | **2.86 (3.15)** | 0.082 (0.082) |
| Flower, 300-px JPEG q55 | before | 0.63 (0.83) | 2.03 (2.11) | 2.83 (3.64) | 0.083 (0.096) |
| | after | **0.55 (0.55)** | **1.33 (1.40)** | **1.40 (2.09)** | 0.052 (0.065) |
| Living room (crop) | before | 0.072 | 1.14 | 1.24 | 0.021 |
| | after | 0.073 | 1.14 | 1.25 | 0.022 |

- **Flower:** for the compressed version (closest to the original upload) blocking drops from 2.03 to 1.33, grain halves and colourfulness falls from 0.63 to 0.55; the remaining "oversaturation" was mostly amplified colour noise, which is now removed before brightening. On the uncompressed screenshot crop the hybrid's colourfulness is unchanged (0.59): after the saturation ceiling was removed from the hybrid (§9.6), only the classical path caps it (0.78 → 0.55). The lower sharpness index of the compressed version is partly removed block edges and colour noise (which the index counts as "detail") and partly genuine softening; it is the price of not amplifying compression artefacts.
- **Living room:** at the crop's resolution detail recovery adds almost nothing measurable (0.021 → 0.022). The restorer's lost texture sits below the noise floor of the brightened image, and we chose not to sharpen below it. **The real fix for this photo is resolution:** the engine used to return 1280 px for a 3872-px photo; it now returns the uploaded size (up to 2048 px, stated in the app) with the photo's own above-noise fine detail transferred back (§9.3: EPI 0.29 → 0.41 on validation). We could not re-test the original 3872-px file (we only have the screenshot).

### 9.5 Test results of the first version, and what they showed

The first version of the update also capped saturation in the hybrid, *before* its restoration network. Its test results (run of 2026-11-07 ~02:00, `results/metrics.json` at commit 3cc4d0a):

| Test set | Before the update | First version (ceiling before restoration) |
|---|---|---|
| LOL-v1 eval15 | 19.18 dB / 0.779 / 0.206 | 18.91 / 0.775 / 0.226 |
| LOL-v2 real, clean subset | 18.67 / 0.772 / 0.283 | 17.91 / 0.770 / 0.307 |
| LOL-v2 synthetic | 16.88 / 0.765 / 0.253 | 16.75 / 0.760 / 0.258 |
| ExDark (NIQE / noise σ / colourfulness) | 4.18 / 2.10 / 0.39 | 4.60 / 1.68 / 0.28 |

_Cells: PSNR / SSIM / LPIPS. "Before" for clean LOL-v2 real comes from the earlier run's per-image results on the same 9 pairs._

**Cause (failure analysis, measured):** switching off only the saturation ceiling restored eval15 to 19.18 dB and the clean LOL-v2 real pairs to 18.67 dB, i.e. it explained the whole drop. Before restoration, chroma noise inflates colourfulness (0.64 on a clean LOL-v2 input whose reference is 0.17), so the ceiling fired on noise and desaturated true colours. On the original validation it had *improved* colour error (ΔE 10.70 → 10.52), which is why it had been kept.

### 9.6 The fix, re-validated on fresh data (disclosure)

**Disclosure:** this change was prompted by the test-set observation in §9.5, so the test numbers after it are **not fully blind** for this one decision. To limit that, the choice was made only on validation data, including data never used before, with a rule written down before any result was seen.

- **Candidates:** ceiling before restoration (first version) · after restoration and detail recovery · no ceiling in the hybrid. The classical path keeps its ceiling, which already runs after its denoising.
- **Data:** the 50 `our485` validation pairs; their compressed proxy (384 px, JPEG q70); and **100 fresh pairs from the LOL-v2 synthetic *training* split**, never downloaded or used before, checked by pixel hash to share no image with `our485` or any test set.
- **Rule (fixed in advance):** the lowest mean LPIPS over the three validation sets (PSNR breaks ties), among candidates that keep the compressed flower's colourfulness ≤ 0.55, so that the oversaturation fix survives.

| Candidate | our485 val PSNR / LPIPS / ΔE | Compressed proxy PSNR / LPIPS | Fresh LOL-v2 synthetic train PSNR / LPIPS | Mean LPIPS | Compressed flower colourfulness |
|---|---|---|---|---|---|
| before restoration (first version) | 19.37 / 0.2080 / 10.52 | 18.70 / 0.2818 | 16.36 / 0.2682 | 0.25265 | 0.37 |
| after restoration | 19.41 / 0.2072 / 10.69 | 18.69 / 0.2821 | 16.41 / 0.2652 | 0.25148 | 0.55 |
| **none in the hybrid (selected)** | 19.41 / 0.2072 / 10.70 | 18.69 / 0.2822 | **16.45 / 0.2641** | **0.25117** | 0.55 |

Applied after restoration, the ceiling changed only 6 of the 200 validation outputs and was slightly worse on the fresh set, so the rule selected *no ceiling in the hybrid*. The differences between the last two rows are small; the rule decided, and the simpler pipeline won.

**Test results after the fix** (the run that produced every other table in this file):

| Test set | Before the update | First version | **Final** |
|---|---|---|---|
| LOL-v1 eval15 | 19.18 / 0.779 / 0.206 | 18.91 / 0.775 / 0.226 | **19.18 / 0.779 / 0.207** |
| LOL-v2 real (clean subset) | 18.67 / 0.772 / 0.283 | 17.91 / 0.770 / 0.307 | **18.67 / 0.772 / 0.283** |
| LOL-v2 synthetic | 16.88 / 0.765 / 0.253 | 16.75 / 0.760 / 0.258 | **16.84 / 0.762 / 0.255** |
| ExDark (NIQE / noise σ / colourfulness / blockiness) | 4.18 / 2.10 / 0.39 / 1.21 | 4.60 / 1.68 / 0.28 / 1.23 | **4.54 / 1.85 / 0.38 / 1.23** |

_Paired cells: PSNR / SSIM / LPIPS._

- **Paired sets:** the final hybrid matches the pipeline before the update on eval15 (19.18 dB, SSIM 0.779) and the clean LOL-v2 real pairs (18.67 dB, 0.772), and is 0.04 dB below it on LOL-v2 synthetic. The first version's losses (−0.27 / −0.76 dB) are gone, as the validation re-run predicted (19.41 vs 19.41 dB on our485 validation).
- **ExDark:** residual noise 2.10 → 1.85 (better); NIQE 4.18 → 4.54 (worse; the first version had 4.60); blockiness 1.21 → 1.23; colourfulness 0.39 → 0.38. Removing the ceiling gave back most of the first version's oversaturation reduction on ExDark (0.28 → 0.38): there the ceiling was doing real work, but it could not be kept without the paired-set losses. Likely causes of the NIQE cost (seen on the compressed validation proxy, not measured per component on ExDark): deblocking on the 17 blocky photos, and the 14× gain cap, which applies to all 60 (they are 384 px).
- **Robustness:** the hybrid gained 0.15–0.88 dB in every condition compared with the first version (e.g. ×0.025: 15.97 → 16.85 dB) and now also wins the spotlight condition.
- **What the update is for:** compressed, small and large real-world photos (blocks, colour noise, full-size output). On the LOL-style test sets it is now neutral, which is what validation predicted.

### 9.7 Auto mode (per-photo method choice)

| Test set | Picks (classical / DL / hybrid) | Auto PSNR / SSIM | Always hybrid PSNR / SSIM | Oracle (best PSNR per image) |
|---|---|---|---|---|
| LOL-v1 eval15 | 0 / 0 / 15 | 19.18 / 0.779 | 19.18 / 0.779 | 20.51 / 0.677 |
| LOL-v2 real test (clean subset) | 0 / 0 / 9 | 18.67 / 0.772 | 18.67 / 0.772 | 19.52 / 0.611 |
| LOL-v2 synthetic test | 49 / 2 / 49 | 16.65 / 0.762 | 16.84 / 0.762 | 20.46 / 0.838 |
| ExDark (real-world, no reference) | 25 / 0 / 35 | NIQE 4.54 | NIQE 4.54 | - |

_The oracle needs the ground truth, so no app can reach it; it shows the headroom of choosing per image._

On validation, choosing per photo from no-reference checks never beat always using the hybrid (zero margin: −0.32 dB on the 50 validation pairs and −1.49 dB on the compressed proxy), although an oracle shows real headroom. Auto therefore keeps the hybrid unless another output's penalty is lower by more than 0.3 (validation: 0.00 / −0.17 dB vs always-hybrid). On the test sets it keeps the hybrid on every real paired photo, switches on 51 of 100 synthetic photos (49 classical, 2 DL; 16.65 vs 16.84 dB for always-hybrid) and to classical on 25 of 60 ExDark photos (NIQE 4.54 vs 4.54). The app shows the full formula and says when it keeps the hybrid by default.

## 10. Reproduce

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/train_dce.py && python scripts/train_restore.py && python scripts/train_restore.py --no-map
python scripts/evaluate.py && python scripts/report.py
python -m pytest -q
```
