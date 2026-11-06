# Benchmark results
_Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only. All test images are unseen (never used for training, validation or tuning). Best value per column in **bold**._

### LOL-v1 eval15 (15 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **7** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 10 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 216 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.56 | 0.559 | 0.471 | 7.31 | 138 | 6.22 | 0.395 | 15.25 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 224 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **19.18** | **0.779** | **0.206** | **4.05** | 816 | **1.18** | **0.623** | **11.22** | 0.08 |

### LOL-v2 real test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.98 | 0.418 | 0.551 | 8.55 | **7** | 16.49 | 0.512 | 17.93 | 3.49 |
| CLAHE only (baseline) | 11.96 | 0.435 | 0.421 | 7.71 | 10 | 4.03 | 0.625 | 22.91 | **0.00** |
| Zero-DCE++ official (baseline) | 18.21 | 0.575 | 0.316 | 8.14 | 216 | 6.59 | 0.627 | 14.55 | 0.00 |
| Classical only (ours) | 17.58 | 0.588 | 0.485 | 7.16 | 138 | 6.27 | 0.490 | 15.95 | 0.00 |
| Deep learning only (ours, fine-tuned) | 17.44 | 0.532 | 0.355 | 8.28 | 224 | 8.74 | 0.594 | 15.25 | 0.00 |
| Hybrid (ours) | **19.17** | **0.800** | **0.223** | **4.05** | 816 | **1.58** | **0.660** | **11.61** | 0.01 |

### LOL-v2 synthetic test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **7** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 10 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 216 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.58 | 0.745 | 0.284 | 4.70 | 138 | 3.18 | 0.815 | 13.80 | **0.27** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 224 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.88 | 0.765 | 0.253 | **4.38** | 816 | **1.89** | 0.817 | 14.09 | 3.16 |

### Average over the three paired test sets (215 pairs, sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.32 | 0.526 | 0.481 | 7.27 | 11.41 | 0.571 | 16.77 | 4.10 |
| CLAHE only (baseline) | 11.08 | 0.438 | 0.430 | 6.60 | 3.31 | 0.638 | 26.53 | 0.13 |
| Zero-DCE++ official (baseline) | 17.13 | 0.654 | 0.281 | 6.84 | 5.07 | 0.698 | 16.33 | 0.13 |
| Classical only (ours) | 16.91 | 0.631 | 0.413 | 6.39 | 5.22 | 0.567 | 15.00 | **0.09** |
| Deep learning only (ours, fine-tuned) | 18.04 | 0.642 | 0.287 | 6.95 | 6.43 | 0.675 | 13.98 | 0.40 |
| Hybrid (ours) | **18.41** | **0.781** | **0.227** | **4.16** | **1.55** | **0.700** | **12.31** | 1.08 |

### ExDark (real-world, no reference) (60 real-world night photos, no ground truth)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.35 | 3.45 | 1.00 | 21.0 | 0.39 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.18 | 2.10 | 3.09 | 14.0 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

## Ablation study (hybrid, one component removed at a time)

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

## Robustness across illumination conditions (synthetic, 30 unseen LOL-v2 references)
_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 13.85 / 0.476 | 13.67 / 0.399 | 13.30 / 0.289 | 12.82 / 0.208 | 12.06 / 0.138 | 11.38 / 0.321 |
| Zero-DCE++ official (baseline) | 12.93 / 0.608 | 14.65 / 0.521 | 16.57 / 0.365 | 16.40 / 0.245 | 14.66 / 0.150 | 15.07 / 0.332 |
| Classical only (ours) | **19.95** / 0.771 | 19.27 / 0.703 | 18.93 / 0.598 | 19.07 / 0.551 | 19.75 / 0.580 | **17.06** / 0.544 |
| Deep learning only (ours, fine-tuned) | 8.05 / 0.546 | 9.15 / 0.470 | 11.07 / 0.325 | 12.60 / 0.212 | 13.10 / 0.125 | 11.00 / 0.295 |
| Hybrid (ours) | 18.15 / 0.829 | **19.32** / 0.821 | **19.96** / 0.788 | **21.11** / 0.752 | **23.09** / 0.695 | 16.10 / 0.724 |

_Cells: PSNR (dB) / SSIM._

## Efficiency (CPU only)

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
