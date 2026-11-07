# Benchmark results
_Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only. All test images are unseen (never used for training, validation or tuning); LOL-v2 real is restricted to the pairs whose input and reference are not pixel-identical to any LOL-v1 our485 image. Best value per column in **bold**._

### LOL-v1 eval15 (15 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **5** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 11 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 217 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.70 | 0.569 | 0.481 | 7.32 | 193 | 6.00 | 0.402 | 14.87 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 219 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **19.13** | **0.777** | **0.207** | **4.13** | 808 | **1.27** | **0.623** | **11.23** | 0.08 |

### LOL-v2 real test (clean subset) (9 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.16 | 0.279 | 0.759 | 10.13 | **5** | 21.42 | 0.389 | 20.78 | 4.36 |
| CLAHE only (baseline) | 10.91 | 0.294 | 0.586 | 9.02 | 11 | 3.24 | 0.484 | 25.59 | **0.00** |
| Zero-DCE++ official (baseline) | 15.30 | 0.396 | 0.486 | 9.85 | 217 | 6.58 | 0.476 | 19.75 | **0.00** |
| Classical only (ours) | 16.94 | 0.543 | 0.656 | 8.61 | 193 | 7.17 | 0.346 | 20.89 | **0.00** |
| Deep learning only (ours, fine-tuned) | 16.29 | 0.375 | 0.541 | 10.15 | 219 | 9.31 | 0.454 | 18.95 | **0.00** |
| Hybrid (ours) | **18.65** | **0.770** | **0.283** | **4.20** | 808 | **1.24** | **0.528** | **14.75** | 0.00 |

### LOL-v2 synthetic test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **5** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 11 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 217 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.58 | 0.742 | 0.292 | 4.82 | 193 | 2.97 | 0.805 | 13.78 | **0.31** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 219 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.80 | 0.760 | 0.256 | **4.43** | 808 | **1.98** | 0.814 | 14.09 | 3.25 |

### Average over the three paired test sets (124 pairs, sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.04 | 0.479 | 0.551 | 7.79 | 13.05 | 0.530 | 17.72 | 4.39 |
| CLAHE only (baseline) | 10.73 | 0.391 | 0.485 | 7.03 | 3.05 | 0.591 | 27.43 | 0.13 |
| Zero-DCE++ official (baseline) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 0.647 | 18.06 | 0.13 |
| Classical only (ours) | 16.74 | 0.618 | 0.477 | 6.92 | 5.38 | 0.518 | 16.51 | **0.10** |
| Deep learning only (ours, fine-tuned) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 0.628 | 15.22 | 0.40 |
| Hybrid (ours) | **18.19** | **0.769** | **0.249** | **4.25** | **1.50** | **0.655** | **13.36** | 1.11 |

### ExDark (real-world, no reference) (60 real-world night photos, no ground truth)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.91 | 2.66 | 0.65 | 14.5 | 0.38 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.62 | 1.89 | 3.35 | 13.9 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

## Quality-aware update: before vs after (hybrid)

| Test set | Version | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | Colourfulness | Blockiness ↓ | Detail |
|---|---|---|---|---|---|---|---|---|---|
| LOL-v1 eval15 | before | 19.18 | 0.779 | 0.206 | 4.05 | 1.18 | - | - | 1.01 |
| LOL-v1 eval15 | after | 19.13 | 0.777 | 0.207 | 4.13 | 1.27 | - | - | 1.08 |
| LOL-v2 synthetic test | before | 16.88 | 0.765 | 0.253 | 4.38 | 1.89 | - | - | 1.12 |
| LOL-v2 synthetic test | after | 16.80 | 0.760 | 0.256 | 4.43 | 1.98 | - | - | 1.19 |
| ExDark (real-world, no reference) | before | - | - | - | 4.18 | 2.10 | 0.39 | 1.21 | 0.047 |
| ExDark (real-world, no reference) | after | - | - | - | 4.62 | 1.89 | 0.38 | 1.24 | 0.045 |

_Detail: paired sets = sharpness relative to the reference (1 = equal); ExDark = absolute sharpness index._

## Ablation study (hybrid, one component removed at a time)

### LOL-v1 eval15
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 19.13 | +0.00 | 0.777 | 0.207 | 4.13 | 1.27 | 11.23 | 0.08 |
| w/o adaptive parameters (fixed) | 17.92 | -1.20 | 0.772 | 0.234 | 4.68 | 1.06 | 12.34 | 0.00 |
| w/o Zero-DCE++ branch (Retinex only) | 18.94 | -0.19 | 0.772 | 0.211 | **4.00** | 1.34 | 11.29 | 0.04 |
| w/o Retinex branch (Zero-DCE++ only) | 18.37 | -0.76 | 0.770 | 0.225 | 4.51 | 1.17 | 11.96 | 0.98 |
| w/o quality-aware curve scaling | 19.13 | +0.01 | 0.776 | 0.209 | 4.18 | 1.24 | 11.36 | 0.11 |
| w/o DL restoration (classical denoise) | 17.25 | -1.88 | 0.649 | 0.393 | 6.88 | 4.37 | 14.23 | 0.00 |
| w/o noise-map conditioning | 18.76 | -0.37 | 0.774 | 0.220 | 4.25 | 0.90 | 11.59 | 0.02 |
| w/o CLAHE | **19.52** | **+0.39** | **0.781** | 0.226 | 5.23 | **0.71** | **10.95** | **0.00** |
| w/o colour correction | 19.04 | -0.08 | 0.777 | 0.214 | 4.19 | 1.20 | 11.20 | 0.05 |
| w/o highlight protection | 19.13 | +0.01 | 0.778 | **0.206** | 4.12 | 1.26 | 11.22 | 0.08 |
| w/o detail recovery | 19.18 | +0.05 | 0.779 | 0.206 | 4.05 | 1.18 | 11.22 | 0.08 |
| w/o quality-aware strategy | 19.13 | +0.00 | 0.777 | 0.207 | 4.13 | 1.27 | 11.23 | 0.08 |
| previous pipeline (before quality-aware update) | 19.18 | +0.05 | 0.779 | 0.206 | 4.05 | 1.18 | 11.22 | 0.08 |

### LOL-v2 synthetic test
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 16.80 | +0.00 | 0.760 | 0.256 | 4.43 | 1.98 | 14.09 | 3.25 |
| w/o adaptive parameters (fixed) | 17.40 | +0.60 | 0.781 | 0.253 | 4.42 | 1.90 | 13.87 | 3.91 |
| w/o Zero-DCE++ branch (Retinex only) | 17.23 | +0.42 | 0.765 | 0.246 | 4.44 | 1.93 | 13.52 | 3.03 |
| w/o Retinex branch (Zero-DCE++ only) | 15.96 | -0.85 | 0.729 | 0.283 | 4.64 | 1.83 | 15.38 | 3.94 |
| w/o quality-aware curve scaling | **17.76** | **+0.96** | **0.789** | **0.246** | 4.42 | 2.26 | **12.59** | 3.79 |
| w/o DL restoration (classical denoise) | 16.58 | -0.22 | 0.740 | 0.271 | 5.04 | 3.23 | 13.98 | **0.26** |
| w/o noise-map conditioning | 16.55 | -0.25 | 0.746 | 0.314 | 4.87 | 1.12 | 14.98 | 2.65 |
| w/o CLAHE | 16.11 | -0.70 | 0.721 | 0.279 | 5.27 | **1.11** | 14.87 | 4.34 |
| w/o colour correction | 16.87 | +0.07 | 0.760 | 0.256 | 4.45 | 1.98 | 13.87 | 3.38 |
| w/o highlight protection | 16.89 | +0.09 | 0.761 | 0.253 | 4.43 | 1.96 | 14.02 | 3.17 |
| w/o detail recovery | 16.83 | +0.03 | 0.761 | 0.256 | 4.41 | 1.86 | 14.08 | 3.17 |
| w/o quality-aware strategy | 16.85 | +0.04 | 0.764 | 0.252 | 4.41 | 2.02 | 14.10 | 3.24 |
| previous pipeline (before quality-aware update) | 16.88 | +0.07 | 0.765 | 0.253 | **4.38** | 1.89 | 14.09 | 3.16 |

## Robustness across illumination conditions (synthetic, 30 unseen LOL-v2 synthetic references)
_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 16.48 / 0.561 | 16.06 / 0.500 | 15.39 / 0.401 | 14.73 / 0.317 | 13.79 / 0.231 | 11.42 / 0.392 |
| Zero-DCE++ official (baseline) | 15.14 / 0.672 | 15.79 / 0.611 | 15.66 / 0.491 | 14.68 / 0.374 | 12.97 / 0.249 | 13.49 / 0.379 |
| Classical only (ours) | 16.84 / 0.710 | 15.58 / 0.642 | 15.16 / 0.548 | 15.10 / 0.495 | 15.03 / 0.459 | 14.14 / 0.511 |
| Deep learning only (ours, fine-tuned) | 10.22 / 0.602 | 11.33 / 0.565 | 13.04 / 0.474 | 14.15 / 0.368 | 14.12 / 0.242 | 12.71 / 0.359 |
| Hybrid (ours) | **19.39** / 0.820 | **18.02** / 0.801 | **17.12** / 0.764 | **16.77** / 0.719 | **16.85** / 0.660 | **14.15** / 0.675 |

_Cells: PSNR (dB) / SSIM._

## Auto mode: per-image method choice from no-reference checks

| Test set | Picks (classical / DL / hybrid) | Auto PSNR / SSIM | Always hybrid PSNR / SSIM | Oracle (best PSNR per image) |
|---|---|---|---|---|
| LOL-v1 eval15 | 0 / 0 / 15 | 19.13 / 0.777 | 19.13 / 0.777 | 20.49 / 0.677 |
| LOL-v2 real test (clean subset) | 0 / 0 / 9 | 18.65 / 0.770 | 18.65 / 0.770 | 19.51 / 0.610 |
| LOL-v2 synthetic test | 49 / 2 / 49 | 16.61 / 0.759 | 16.80 / 0.760 | 20.42 / 0.837 |
| ExDark (real-world, no reference) | 25 / 1 / 34 | NIQE 4.67 | NIQE 4.62 | - |

_The oracle needs the ground truth, so no app can reach it; it shows the headroom of choosing per image._

## Efficiency (CPU only)

| Method | 600×400 (ms) | 1280×960 (ms) | 2048×1536 upload, full-size output (ms) |
|---|---|---|---|
| Histogram equalisation (baseline) | 5 | 29 | 319 |
| CLAHE only (baseline) | 11 | 39 | 321 |
| Zero-DCE++ official (baseline) | 217 | 490 | 760 |
| Classical only (ours) | 193 | 1082 | 1357 |
| Deep learning only (ours, fine-tuned) | 219 | 484 | 766 |
| Hybrid (ours) | 808 | 2194 | 2476 |

| Model | Parameters | File size |
|---|---|---|
| Zero-DCE++ (fine-tuned) | 10,561 | 51.0 KB |
| Restoration U-Net | 144,579 | 574.6 KB |
