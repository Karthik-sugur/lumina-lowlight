# Benchmark results
_Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only. All test images are unseen (never used for training, validation or tuning); LOL-v2 real is restricted to the pairs whose input and reference are not pixel-identical to any LOL-v1 our485 image. Best value per column in **bold**._

### LOL-v1 eval15 (15 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **7** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 14 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 242 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.71 | 0.572 | 0.470 | 7.24 | 245 | 5.98 | 0.415 | 14.85 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 246 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **19.18** | **0.779** | **0.207** | **4.11** | 1001 | **1.19** | **0.622** | **11.22** | 0.08 |

### LOL-v2 real test (clean subset) (9 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.16 | 0.279 | 0.759 | 10.13 | **7** | 21.42 | 0.389 | 20.78 | 4.36 |
| CLAHE only (baseline) | 10.91 | 0.294 | 0.586 | 9.02 | 14 | 3.24 | 0.484 | 25.59 | **0.00** |
| Zero-DCE++ official (baseline) | 15.30 | 0.396 | 0.486 | 9.85 | 242 | 6.58 | 0.476 | 19.75 | **0.00** |
| Classical only (ours) | 16.95 | 0.547 | 0.639 | 8.57 | 245 | 7.24 | 0.368 | 20.86 | **0.00** |
| Deep learning only (ours, fine-tuned) | 16.29 | 0.375 | 0.541 | 10.15 | 246 | 9.31 | 0.454 | 18.95 | **0.00** |
| Hybrid (ours) | **18.67** | **0.772** | **0.283** | **4.22** | 1001 | **1.19** | **0.528** | **14.74** | 0.00 |

### LOL-v2 synthetic test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **7** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 14 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 242 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.62 | 0.745 | 0.290 | 4.80 | 245 | 2.87 | 0.807 | 13.76 | **0.26** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 246 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.84 | 0.762 | 0.255 | **4.40** | 1001 | **1.88** | 0.817 | 14.08 | 3.17 |

### Average over the three paired test sets (124 pairs, sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.04 | 0.479 | 0.551 | 7.79 | 13.05 | 0.530 | 17.72 | 4.39 |
| CLAHE only (baseline) | 10.73 | 0.391 | 0.485 | 7.03 | 3.05 | 0.591 | 27.43 | 0.13 |
| Zero-DCE++ official (baseline) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 0.647 | 18.06 | 0.13 |
| Classical only (ours) | 16.76 | 0.621 | 0.467 | 6.87 | 5.36 | 0.530 | 16.49 | **0.09** |
| Deep learning only (ours, fine-tuned) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 0.628 | 15.22 | 0.40 |
| Hybrid (ours) | **18.23** | **0.771** | **0.248** | **4.24** | **1.42** | **0.655** | **13.34** | 1.08 |

### ExDark (real-world, no reference) (60 real-world night photos, no ground truth)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.82 | 2.64 | 0.55 | 14.5 | 0.38 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.54 | 1.85 | 3.24 | 14.0 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

## Quality-aware update: before vs after (hybrid)

| Test set | Version | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | Colourfulness | Blockiness ↓ | Detail |
|---|---|---|---|---|---|---|---|---|---|
| LOL-v1 eval15 | before | 19.18 | 0.779 | 0.206 | 4.05 | 1.18 | - | - | 1.01 |
| LOL-v1 eval15 | after | 19.18 | 0.779 | 0.207 | 4.11 | 1.19 | - | - | 1.01 |
| LOL-v2 synthetic test | before | 16.88 | 0.765 | 0.253 | 4.38 | 1.89 | - | - | 1.12 |
| LOL-v2 synthetic test | after | 16.84 | 0.762 | 0.255 | 4.40 | 1.88 | - | - | 1.13 |
| ExDark (real-world, no reference) | before | - | - | - | 4.18 | 2.10 | 0.39 | 1.21 | 0.047 |
| ExDark (real-world, no reference) | after | - | - | - | 4.54 | 1.85 | 0.38 | 1.23 | 0.043 |

_Detail: paired sets = sharpness relative to the reference (1 = equal); ExDark = absolute sharpness index._

## Ablation study (hybrid, one component removed at a time)

### LOL-v1 eval15
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

### LOL-v2 synthetic test
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

## Robustness across illumination conditions (synthetic, 30 unseen LOL-v2 synthetic references)
_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 16.48 / 0.561 | 16.06 / 0.500 | 15.39 / 0.401 | 14.73 / 0.317 | 13.79 / 0.231 | 11.42 / 0.392 |
| Zero-DCE++ official (baseline) | 15.14 / 0.672 | 15.79 / 0.611 | 15.66 / 0.491 | 14.68 / 0.374 | 12.97 / 0.249 | 13.49 / 0.379 |
| Classical only (ours) | 16.84 / 0.707 | 15.58 / 0.637 | 15.15 / 0.543 | 15.08 / 0.485 | 14.99 / 0.425 | 14.13 / 0.494 |
| Deep learning only (ours, fine-tuned) | 10.22 / 0.602 | 11.33 / 0.565 | 13.04 / 0.474 | 14.15 / 0.368 | 14.12 / 0.242 | 12.71 / 0.359 |
| Hybrid (ours) | **19.40** / 0.820 | **18.02** / 0.800 | **17.11** / 0.762 | **16.76** / 0.717 | **16.85** / 0.658 | **14.15** / 0.673 |

_Cells: PSNR (dB) / SSIM._

## Auto mode: per-image method choice from no-reference checks

| Test set | Picks (classical / DL / hybrid) | Auto PSNR / SSIM | Always hybrid PSNR / SSIM | Oracle (best PSNR per image) |
|---|---|---|---|---|
| LOL-v1 eval15 | 0 / 0 / 15 | 19.18 / 0.779 | 19.18 / 0.779 | 20.51 / 0.677 |
| LOL-v2 real test (clean subset) | 0 / 0 / 9 | 18.67 / 0.772 | 18.67 / 0.772 | 19.52 / 0.611 |
| LOL-v2 synthetic test | 49 / 2 / 49 | 16.65 / 0.762 | 16.84 / 0.762 | 20.46 / 0.838 |
| ExDark (real-world, no reference) | 25 / 0 / 35 | NIQE 4.54 | NIQE 4.54 | - |

_The oracle needs the ground truth, so no app can reach it; it shows the headroom of choosing per image._

## Efficiency (CPU only)

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
