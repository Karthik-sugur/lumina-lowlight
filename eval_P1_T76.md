1RUA24CSE0170, 1RUA24CSE0195, 1RUA24CSE0127, 1RUA24CSE0115

# Evaluation: P1 Robust Low-Light Image Restoration and Enhancement (Team T76)

**Team T76:** Indhu S (1RUA24CSE0170), Karthikeya D S (1RUA24CSE0195), Dheeraj Chinta (1RUA24CSE0127), Darsh Chandran (1RUA24CSE0115)
**System:** Lumina, an adaptive hybrid of classical CV and deep learning · **Live app:** https://darshuuuu-lumina-lowlight.hf.space · **Code:** https://github.com/darshchandran/lumina-lowlight

Every number below was produced by `scripts/evaluate.py` and `scripts/report.py`. Per-image values are in `results/per_image.csv`, aggregates in `results/metrics.json`, and the auto-generated tables in `results/tables.md`; the tables in this file are copied from that file programmatically. Metrics that were not measured are marked **n/a** with the reason. Nothing is estimated.

## 0. Summary

| | |
|---|---|
| Deployed model | Hybrid pipeline with Zero-DCE++ (fine-tuned) and **restoration U-Net v3** (noise-map conditioned, trained on real + synthetic pairs) |
| Model selection | On the **validation split only** (§5); test sets were never used to choose anything |
| Paired test sets, averaged (215 pairs) | Hybrid **18.41 dB PSNR · 0.781 SSIM · 0.227 LPIPS · 4.16 NIQE**, best of all six methods on 7 of 8 metrics; it is *worst but one* on new overexposure (1.08 %), see §7 |
| Unseen real-world night photos (ExDark, no reference) | Hybrid has the best NIQE (4.18) and lowest residual noise (2.10), but the most clipped pixels after histogram equalisation (3.09 %) |
| Robustness (darkness sweep) | Hybrid best from ×0.25 to ×0.025 exposure (23.09 dB at ×0.025); classical-only is better at mild ×0.5 darkening and under a spotlight |
| Latency (laptop CPU) | 816 ms at 600×400, 1.87 s at 1280×960 (§2.4) |

## 1. Setup

**Test data:** all disjoint from training and validation.

| Set | Type | Images | Reference? |
|---|---|---|---|
| LOL-v1 `eval15` | real low/normal-light pairs | 15 | yes: **paired metrics** |
| LOL-v2 real `Test` | real pairs, different captures | 100 | yes: **paired metrics** |
| LOL-v2 synthetic `Test` | synthetically darkened photos | 100 | yes: **paired metrics** |
| ExDark (384-px release, letterbox trimmed) | real-world night photos | 60 (seed 0) | **no: no-reference metrics only** |
| Robustness sweep | 30 unseen LOL-v2-real references degraded physically at 6 conditions | 180 per method | yes: paired |

**Training / validation:** LOL-v1 `our485` split 435 / 50 (seed 0, `results/split.json`). The restoration U-Net also trains on one synthetic degradation of each *training* reference. Validation = the 50 held-out pairs plus one synthetic degradation of each.

**Methods:** two naive baselines (histogram equalisation, CLAHE), the published Zero-DCE++ weights (zero-shot), and our **classical-only**, **deep-learning-only** (fine-tuned Zero-DCE++) and **hybrid** pipelines.

**Metric families**

| Paired-reference (need ground truth) | No-reference (any image) |
|---|---|
| PSNR, SSIM (Gaussian 11×11, σ 1.5, RGB), LPIPS (AlexNet), CIEDE2000 ΔE (colour shift), EPI (edge preservation: Laplacian correlation with the reference), new-overexposure % (clipped in output but not in reference), brightness error | NIQE, residual noise σ (Immerkær, 0–255), clipped-pixel %, colour cast (Lab chroma of the mean mid-tone colour), mean brightness |

**Not available:** PSNR, SSIM, LPIPS, ΔE and EPI on ExDark (**n/a**, because ExDark has no ground truth). FID was not computed (not required for P1; no distribution-level reference set).

**Hardware:** AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only (laptop). Latency is measured in one process after warm-up (§2.4).

## 2. Evaluation metrics (in the order of the problem statement)

Column order in every table: PSNR → SSIM → LPIPS → NIQE → latency → noise suppression → edge preservation, then the advanced-requirement checks (colour shift, overexposure). Best value per column is **bold**.

### 2.1–2.5 Per test set

#### LOL-v1 eval15 (15 pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **7** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 10 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 216 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.56 | 0.559 | 0.471 | 7.31 | 138 | 6.22 | 0.395 | 15.25 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 224 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **19.18** | **0.779** | **0.206** | **4.05** | 816 | **1.18** | **0.623** | **11.22** | 0.08 |

#### LOL-v2 real test (100 pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.98 | 0.418 | 0.551 | 8.55 | **7** | 16.49 | 0.512 | 17.93 | 3.49 |
| CLAHE only (baseline) | 11.96 | 0.435 | 0.421 | 7.71 | 10 | 4.03 | 0.625 | 22.91 | **0.00** |
| Zero-DCE++ official (baseline) | 18.21 | 0.575 | 0.316 | 8.14 | 216 | 6.59 | 0.627 | 14.55 | 0.00 |
| Classical only (ours) | 17.58 | 0.588 | 0.485 | 7.16 | 138 | 6.27 | 0.490 | 15.95 | 0.00 |
| Deep learning only (ours, fine-tuned) | 17.44 | 0.532 | 0.355 | 8.28 | 224 | 8.74 | 0.594 | 15.25 | 0.00 |
| Hybrid (ours) | **19.17** | **0.800** | **0.223** | **4.05** | 816 | **1.58** | **0.660** | **11.61** | 0.01 |

#### LOL-v2 synthetic test (100 pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **7** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 10 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 216 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.58 | 0.745 | 0.284 | 4.70 | 138 | 3.18 | 0.815 | 13.80 | **0.27** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 224 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.88 | 0.765 | 0.253 | **4.38** | 816 | **1.89** | 0.817 | 14.09 | 3.16 |

#### Average over the three paired sets (sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.32 | 0.526 | 0.481 | 7.27 | 11.41 | 0.571 | 16.77 | 4.10 |
| CLAHE only (baseline) | 11.08 | 0.438 | 0.430 | 6.60 | 3.31 | 0.638 | 26.53 | 0.13 |
| Zero-DCE++ official (baseline) | 17.13 | 0.654 | 0.281 | 6.84 | 5.07 | 0.698 | 16.33 | 0.13 |
| Classical only (ours) | 16.91 | 0.631 | 0.413 | 6.39 | 5.22 | 0.567 | 15.00 | **0.09** |
| Deep learning only (ours, fine-tuned) | 18.04 | 0.642 | 0.287 | 6.95 | 6.43 | 0.675 | 13.98 | 0.40 |
| Hybrid (ours) | **18.41** | **0.781** | **0.227** | **4.16** | **1.55** | **0.700** | **12.31** | 1.08 |

#### ExDark: no-reference only (60 real-world night photos)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.35 | 3.45 | 1.00 | 21.0 | 0.39 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.18 | 2.10 | 3.09 | 14.0 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

PSNR / SSIM / LPIPS / ΔE / EPI for ExDark: **n/a** (no ground truth). Colour cast on night scenes is partly intentional (sodium or neon light), so it is reported, not ranked.

### 2.4 Processing latency and model size
| Method | 600×400 (ms) | 1280×960 (ms) |
|---|---|---|
| Histogram equalisation (baseline) | 7 | 34 |
| CLAHE only (baseline) | 10 | 38 |
| Zero-DCE++ official (baseline) | 216 | 534 |
| Classical only (ours) | 138 | 773 |
| Deep learning only (ours, fine-tuned) | 224 | 495 |
| Hybrid (ours) | 816 | 1867 |

| Model | Parameters | File size |
|---|---|---|
| Zero-DCE++ (fine-tuned) | 10,561 | 51.0 KB |
| Restoration U-Net | 144,579 | 574.6 KB |

- Laptop CPU above. The hybrid spends most of its time in the restoration U-Net and the Zero-DCE++ curve estimation (per-stage timings are shown live in the app's *Under the hood* tab).
- **Hosted app (Hugging Face, free ZeroGPU Space, CPU inference): measured on the live Space (2026-11-06, Space warm):
  - **Browser, click → finished result:** 1.59 s (desktop, 600×400 example), 1.32 s (mobile profile, 380×348 example).
  - **Server processing (reported by the app):** 0.4 s at 600×400 · 1.3 s at 1280×960 · 1.4 s for a 4000×3000 upload (downsized to 1280×960).
  - **Compare all six methods:** about 1.6 s after the click (hybrid 374 ms, classical 112 ms, DL-only 130 ms at 600×400). **Under the hood**, 13 intermediate images loaded: 1.0 s.
  - **Caveats:** a scripted API client that downloads *every* full-resolution output (14 intermediate images) took 35–38 s per request; browsers fetch intermediates only when that tab is opened. **Cold start after the Space sleeps (48 h idle) was not measured.****
- The app resizes uploads to ≤1280 px on the long side, which keeps every request well under the 10-second target.

### 2.5 Noise suppression and edge preservation (reading the tables)
- **Noise:** the hybrid leaves the least residual noise on every paired set (σ 1.18 / 1.58 / 1.89) and on ExDark (2.10). The deep-learning-only method leaves 3.70–8.74, because a pixel-wise curve cannot denoise. Classical-only leaves 3.18–6.27; its bilateral + guided filtering is limited.
- **Edges:** the hybrid has the best EPI on LOL-v1 (0.623) and LOL-v2 real (0.660), and the best average (0.700). On LOL-v2 synthetic the published Zero-DCE++ (0.921) and our DL-only (0.907) beat the hybrid (0.817). Those images are only mildly noisy, so a curve that does no smoothing keeps edges best, while our restoration slightly smooths fine texture.

## 3. Classical-only vs deep-learning-only vs hybrid

| Paired average | PSNR | SSIM | LPIPS | NIQE | Noise σ | EPI | ΔE |
|---|---|---|---|---|---|---|---|
| Classical only | 16.91 | 0.631 | 0.413 | 6.39 | 5.22 | 0.567 | 15.00 |
| Deep learning only | 18.04 | 0.642 | 0.287 | 6.95 | 6.43 | 0.675 | 13.98 |
| **Hybrid** | **18.41** | **0.781** | **0.227** | **4.16** | **1.55** | **0.700** | **12.31** |

- **Classical only:** good exposure control and the fewest blown pixels (0.09 %), but noisy and halo-free-but-soft (lowest EPI, 0.567). It is the strongest method under *mild* darkening (×0.5: 19.95 dB) and spotlight lighting (17.06 dB), because it never over-brightens.
- **Deep learning only:** fine-tuning lifted Zero-DCE++ from 14.32 to 16.81 dB on validation, and it is the best method on LOL-v2 synthetic (18.04 dB). But it learned LOL's "always brighten a lot" behaviour: on mildly dark inputs it blows out (8.05 dB at ×0.5 exposure), and as a pixel-wise curve it cannot remove noise.
- **Hybrid:** best on LOL-v1 and LOL-v2 real on every paired metric except overexposure, and best on the paired average; SSIM is +0.14 over both single approaches. It **is not best on LOL-v2 synthetic** (16.88 dB vs 18.04 for DL-only): those images are bright and lightly noisy, and the restorer's LOL-learned brightening and smoothing hurt there (§8).

## 4. Generalisation across illumination conditions (robustness sweep)
_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 13.85 / 0.476 | 13.67 / 0.399 | 13.30 / 0.289 | 12.82 / 0.208 | 12.06 / 0.138 | 11.38 / 0.321 |
| Zero-DCE++ official (baseline) | 12.93 / 0.608 | 14.65 / 0.521 | 16.57 / 0.365 | 16.40 / 0.245 | 14.66 / 0.150 | 15.07 / 0.332 |
| Classical only (ours) | **19.95** / 0.771 | 19.27 / 0.703 | 18.93 / 0.598 | 19.07 / 0.551 | 19.75 / 0.580 | **17.06** / 0.544 |
| Deep learning only (ours, fine-tuned) | 8.05 / 0.546 | 9.15 / 0.470 | 11.07 / 0.325 | 12.60 / 0.212 | 13.10 / 0.125 | 11.00 / 0.295 |
| Hybrid (ours) | 18.15 / 0.829 | **19.32** / 0.821 | **19.96** / 0.788 | **21.11** / 0.752 | **23.09** / 0.695 | 16.10 / 0.724 |

_Cells: PSNR (dB) / SSIM._

- The hybrid is the best method from ×0.25 down to ×0.025 exposure. Its quality *rises* as images get darker (19.32 → 23.09 dB) because the adaptive classical stage normalises exposure and the restorer was trained on this range.
- It is **not** best at mild darkening (×0.5: 18.15 vs classical 19.95) or with a strong spotlight (16.10 vs 17.06). In both cases it brightens more than the reference, and the spotlight condition was deliberately kept out of training.
- **Caveat:** the restorer's synthetic training pairs use the same physical noise model (on *different* images, the LOL-v1 training references). The exposure sweep is therefore not fully out-of-distribution; the spotlight condition and ExDark are.

## 5. Model selection: v2 vs v3 restorer (validation only)

Full-pipeline PSNR / SSIM on the 50 held-out validation pairs (real) and their synthetic degradations (×0.02–×0.5):

| Restorer | Real | Synthetic | Real − synthetic gap |
|---|---|---|---|
| v2 (LOL pairs only) | **19.65 / 0.839** | 17.54 / 0.766 | 2.11 dB |
| v3 (LOL + synthetic pairs), **selected** | 19.41 / 0.824 | **19.90 / 0.815** | **0.49 dB** |

v3 trades 0.24 dB on real LOL-like images for +2.36 dB across a much wider darkness range, and its real/synthetic gap is about 4× smaller. We weighted generalisation over a quarter-dB on one distribution because judges and users upload arbitrary photos.

**Test-set consequence (observed only after selection, not used for it):** v3 improved LOL-v1 eval15 (18.56 → 19.18 dB) and the darkness sweep at ×0.025 (12.87 → 23.09 dB). It *lost* 0.94 dB on LOL-v2 real (20.11 → 19.17 dB), more than validation predicted. We report it rather than re-select on test data.

## 6. Ablation study (hybrid with one component removed)

### LOL-v1 eval15
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 19.18 | +0.00 | 0.779 | 0.206 | 4.05 | 1.18 | 11.22 | 0.08 |
| w/o adaptive parameters (fixed) | 17.95 | -1.23 | 0.773 | 0.234 | 4.51 | 0.97 | 12.33 | 0.00 |
| w/o Zero-DCE++ branch (Retinex only) | 18.98 | -0.20 | 0.774 | 0.210 | **3.93** | 1.26 | 11.27 | 0.04 |
| w/o Retinex branch (Zero-DCE++ only) | 18.42 | -0.75 | 0.772 | 0.224 | 4.37 | 1.07 | 11.94 | 0.97 |
| w/o quality-aware curve scaling | 19.19 | +0.02 | 0.779 | 0.207 | 4.09 | 1.15 | 11.34 | 0.10 |
| w/o DL restoration (classical denoise) | 17.32 | -1.86 | 0.643 | 0.377 | 6.83 | 4.23 | 14.18 | 0.00 |
| w/o noise-map conditioning | 18.81 | -0.37 | 0.776 | 0.219 | 4.20 | 0.84 | 11.57 | 0.02 |
| w/o CLAHE | **19.49** | **+0.32** | 0.777 | 0.234 | 5.10 | **0.64** | **10.97** | **0.00** |
| w/o colour correction | 19.10 | -0.08 | 0.779 | 0.213 | 4.13 | 1.12 | 11.18 | 0.05 |
| w/o highlight protection | 19.18 | +0.01 | **0.780** | **0.205** | 4.06 | 1.17 | 11.20 | 0.08 |

### LOL-v2 real test
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 19.17 | +0.00 | 0.800 | 0.223 | 4.05 | 1.58 | 11.61 | 0.01 |
| w/o adaptive parameters (fixed) | **20.66** | **+1.49** | **0.818** | 0.222 | 4.29 | 1.41 | **9.46** | 0.00 |
| w/o Zero-DCE++ branch (Retinex only) | 19.28 | +0.11 | 0.790 | 0.232 | **3.96** | 1.63 | 11.66 | 0.01 |
| w/o Retinex branch (Zero-DCE++ only) | 17.92 | -1.25 | 0.798 | 0.233 | 4.26 | 1.49 | 12.44 | 0.24 |
| w/o quality-aware curve scaling | 18.74 | -0.43 | 0.796 | 0.223 | 4.13 | 1.52 | 12.09 | 0.01 |
| w/o DL restoration (classical denoise) | 18.42 | -0.75 | 0.670 | 0.406 | 6.78 | 4.32 | 14.71 | **0.00** |
| w/o noise-map conditioning | 19.20 | +0.02 | 0.804 | 0.230 | 4.29 | 1.30 | 11.61 | 0.01 |
| w/o CLAHE | 19.48 | +0.31 | 0.790 | 0.272 | 5.04 | **0.82** | 11.47 | 0.00 |
| w/o colour correction | 19.22 | +0.05 | 0.801 | 0.226 | 4.08 | 1.51 | 11.34 | 0.01 |
| w/o highlight protection | 19.19 | +0.02 | 0.801 | **0.222** | 4.05 | 1.57 | 11.60 | 0.01 |

![Ablation](results/figures/3_ablation.png)

**Interpretation**
- **DL restoration is the largest single contribution:** replacing it with classical denoising costs 1.86 / 0.75 dB, raises LPIPS by about 0.17–0.18, NIQE by about 2.7–2.8, and leaves 2.7–3.6× more noise.
- **Fusion helps:** removing the Retinex branch costs 0.75 / 1.25 dB and raises overexposure (0.97 % / 0.24 %). Removing the Zero-DCE++ branch is mixed (−0.20 / +0.11 dB; NIQE slightly *better* without it), so **the DL candidate's contribution to the fused image is small**.
- **Adaptive parameters:** fixed parameters are −1.23 dB on LOL-v1 but **+1.49 dB on LOL-v2 real**. The fixed γ = 0.5 brightens less, which happens to match LOL-v2's darker reference images (mean luma 0.371, vs 0.453 for LOL-v1 eval15; measured over all reference images). PSNR is dominated by agreement on overall exposure. NIQE favours the adaptive version on both sets (4.05 vs 4.51 / 4.29).
- **CLAHE lowers PSNR** (+0.32 / +0.31 dB when removed) **but improves perceptual quality:** LPIPS 0.206 vs 0.234 and 0.223 vs 0.272; NIQE 4.05 vs 5.10 and 5.04. It was kept deliberately as a pixel-fidelity vs perceived-contrast trade-off.
- **Quality-aware curve scaling:** +0.02 / −0.43 dB when removed. It helps mainly on LOL-v2 real.
- **Colour correction:** small effect (−0.08 / +0.05 dB).
- **Highlight protection: no measurable effect** (+0.01 / +0.02 dB, overexposure unchanged). The soft-knee runs *before* the restoration network, which can push highlights back toward white. This is a design flaw we found through the ablation (§7).

**Noise-map conditioning: honest conclusion.** The test ablation is mixed: −0.37 dB without the map on LOL-v1, +0.02 dB on LOL-v2 real. The no-map model leaves *less* noise (σ 0.84 vs 1.18) but has worse LPIPS (0.219 vs 0.206) and NIQE (4.20 vs 4.05). On validation, the map helps on real captures in both model generations (+0.25 dB v2, +0.20 dB v3) but **hurts on synthetic noise** (−1.48 dB v2, −0.21 dB v3), and v3's combined validation score is a tie (19.65 vs 19.66). **The noise map gives a small, real-data-specific benefit, not a general one.** We kept it because real photos are the deployment target, but the case for it is weak.

## 7. Overexposure, colour shift and detail loss (advanced requirement 4)
- **Colour shift:** the hybrid has the lowest ΔE2000 on LOL-v1 (11.22), LOL-v2 real (11.61) and on average (12.31). On LOL-v2 synthetic, DL-only is better (12.73 vs 14.09).
- **Detail loss:** the hybrid has the best average EPI (0.700). Fine periodic textures (nets, meshes) can turn into moiré (failure cases 1–2).
- **Overexposure is our weakest point:** 1.08 % newly clipped pixels on the paired average (3.16 % on LOL-v2 synthetic) and 3.09 % clipped on ExDark. The highlight-protection ablation shows the soft-knee has no effect, because the restoration network runs after it and can re-clip. **Planned fix:** apply the soft-knee after restoration and add a highlight-preservation term to the restorer's loss.

## 8. Failure-case analysis

Panels are auto-selected by `scripts/report.py`: the 3 lowest-PSNR hybrid results on LOL-v2 real (input | hybrid | reference) and the 3 highest-NIQE results on ExDark (input | hybrid). Diagnoses come from the pipeline's own analysis (`results/failures/index.md`).

| # | Panel | Set / image | PSNR | SSIM | NIQE | Pipeline diagnosis of the input |
|---|---|---|---|---|---|---|
| 1 | [failure_1.jpg](results/failures/failure_1.jpg) | LOL-v2 real 00759 | 13.74 | 0.69 | 3.49 | brightness 0.064, noise σ 1.1, lighting spread 2.7 stops |
| 2 | [failure_2.jpg](results/failures/failure_2.jpg) | LOL-v2 real 00760 | 13.99 | 0.73 | 3.46 | brightness 0.063, noise σ 1.1, spread 2.9 stops |
| 3 | [failure_3.jpg](results/failures/failure_3.jpg) | LOL-v2 real 00779 | 14.11 | 0.73 | 3.59 | brightness 0.042, noise σ 1.0, spread 1.6 stops |
| 4 | [failure_4.jpg](results/failures/failure_4.jpg) | ExDark 2015_04779 (cup) | n/a | n/a | 8.40 | brightness 0.086, noise σ 0.3, spread 4.9 stops |
| 5 | [failure_5.jpg](results/failures/failure_5.jpg) | ExDark 2015_03556 (black cat) | n/a | n/a | 6.84 | brightness 0.128, noise σ 1.0, spread 1.9 stops |
| 6 | [failure_6.jpg](results/failures/failure_6.jpg) | ExDark 2015_06912 (glass on table) | n/a | n/a | 6.47 | brightness 0.109, noise σ 0.3, spread 2.4 stops |

**Failures 1–3: brighter than the reference (exposure convention).** Two near-identical shots of a table-tennis hall and one of an atrium.
- *What failed:* PSNR of about 14 dB, the lowest in LOL-v2 real.
- *Why (measured):* our outputs are far brighter than these references (mean luma 0.474 / 0.467 / 0.406 vs 0.295 / 0.295 / 0.246). Applying **one global brightness gain** to our output raises PSNR to **23.01 / 23.52 / 22.67 dB** and SSIM to 0.806 / 0.843 / 0.871. Almost all of the error is overall exposure, not structure. Their NIQE (3.46–3.59) is among the best in the set, so the outputs look natural; they are just not as dim as these particular references.
- *Secondary, visible:* the fine barrier net in 1–2 turns into **moiré**. At 600×400 its mesh period is close to the noise scale, so restoration cannot separate texture from noise. In 3 the very dark floor (4 % brightness) keeps **low-frequency blotches**: the restorer removes fine grain, but large-scale noise survives on flat surfaces.
- *What it tells us:* the adaptive target (0.42, near LOL-v1's 0.45) is one convention among several (LOL-v2 real ≈ 0.37). PSNR rewards matching the dataset's exposure, so we let users set intent and also report NIQE and LPIPS.

**Failure 4: a low-key studio photo treated as underexposed.** A white cup lit against pure black.
- *What failed:* the black background is lifted and shows **coloured speckle and blocks** around the handle and in the reflection; the cup's shading flattens toward white. NIQE is 8.40, the worst in the sample.
- *Likely cause (from the diagnosis, not a controlled test):* 4.9 stops of lighting spread only lowers the target slightly (×0.93), so the near-black background still gets a large gain. That amplifies **JPEG blocking and chroma subsampling**, which the restorer never saw in training (sensor noise only). The darkness here is artistic intent, not underexposure.
- *What it tells us:* "dark" ≠ "underexposed". The pipeline lacks a low-key/intent detector, and the recoverability rating doesn't flag this case.

**Failure 5: noise outside the training distribution.** A black cat in what looks like a low-quality video frame.
- *What failed:* after brightening, the whole frame is covered in **multicoloured grain**, and the fur is indistinguishable from noise.
- *Likely cause:* the noise is **spatially correlated and compressed** (camcorder/JPEG), not the independent per-pixel sensor noise the restorer was trained on (LOL + Poisson/Gaussian synthesis). Our noise estimator also measures high-frequency noise only, so it under-reads correlated noise. The noise map therefore asks for too little cleaning. A black subject also means low signal everywhere.
- *What it tells us:* a 145k-parameter supervised restorer is only as general as its noise model. Training on real compressed captures (e.g. phone and video frames) is the clear next step.

**Failure 6: banding and fringes in smooth dark areas.** A glass on a table; the overall result is plausible.
- *What failed:* the smooth background wall shows **vertical banding**, and dark chair edges get **red/purple fringes**. NIQE is 6.47.
- *Likely cause:* in an 8-bit dark gradient, adjacent tones are 1–2 code values apart; multiplying by the brightening gain turns those steps into visible bands. JPEG's half-resolution chroma creates colour fringes at dark edges, which are then amplified.
- *What it tells us:* the gain cap (25×) prevents the worst cases, but any 8-bit input has a bit-depth floor. RAW or 10-bit input, or a de-banding step, would be needed.

**What the failures tell us about the approach**
1. **Exposure is a convention, not a fact.** Our largest PSNR losses come from brightening more than a particular dataset's reference does (failures 1–3). An adaptive target fixes "too dark" but cannot know a photographer's intended brightness. That's why the app exposes *Natural night mood / Auto / Maximum visibility* and a brightness slider, and why we report NIQE and LPIPS beside PSNR.
2. **A small supervised restorer is only as general as its noise model.** It cleans Poisson/Gaussian sensor noise (LOL + synthetic) well, but not spatially correlated camcorder noise or JPEG blocking (failures 4–6). The noise-level estimator, which measures high-frequency noise, also under-reads correlated noise.
3. **"Dark" is not always "underexposed".** Low-key photos (a lit object on black) are treated as underexposed and lifted, which amplifies compression artefacts in the black areas (failure 4). The recoverability rating does not flag this case; a low-key detector is future work.

## 9. Reproduce

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/train_dce.py && python scripts/train_restore.py && python scripts/train_restore.py --no-map
python scripts/evaluate.py && python scripts/report.py
```
