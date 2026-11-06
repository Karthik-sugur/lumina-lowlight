# Benchmark results
_Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (16 threads), CPU only. All test images are unseen (never used for training, validation or tuning); LOL-v2 real is restricted to the pairs whose input and reference are not pixel-identical to any LOL-v1 our485 image. Best value per column in **bold**._

### LOL-v1 eval15 (15 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.24 | 0.419 | 0.592 | 8.36 | **5** | 12.90 | 0.406 | 17.27 | 4.28 |
| CLAHE only (baseline) | 9.37 | 0.367 | 0.501 | 7.52 | 12 | 3.19 | 0.514 | 32.68 | 0.00 |
| Zero-DCE++ official (baseline) | 15.14 | 0.562 | 0.342 | 7.94 | 225 | 5.19 | 0.545 | 19.61 | **0.00** |
| Classical only (ours) | 16.71 | 0.572 | 0.470 | 7.24 | 193 | 5.98 | 0.415 | 14.85 | 0.00 |
| Deep learning only (ours, fine-tuned) | 18.63 | 0.560 | 0.343 | 7.97 | 234 | 6.84 | 0.524 | 13.97 | 0.00 |
| Hybrid (ours) | **18.91** | **0.775** | **0.226** | **4.17** | 837 | **1.05** | **0.625** | **11.46** | 0.08 |

### LOL-v2 real test (clean subset) (9 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 12.16 | 0.279 | 0.759 | 10.13 | **5** | 21.42 | 0.389 | 20.78 | 4.36 |
| CLAHE only (baseline) | 10.91 | 0.294 | 0.586 | 9.02 | 12 | 3.24 | 0.484 | 25.59 | **0.00** |
| Zero-DCE++ official (baseline) | 15.30 | 0.396 | 0.486 | 9.85 | 225 | 6.58 | 0.476 | 19.75 | **0.00** |
| Classical only (ours) | 16.95 | 0.547 | 0.639 | 8.57 | 193 | 7.24 | 0.368 | 20.86 | **0.00** |
| Deep learning only (ours, fine-tuned) | 16.29 | 0.375 | 0.541 | 10.15 | 234 | 9.31 | 0.454 | 18.95 | **0.00** |
| Hybrid (ours) | **17.91** | **0.770** | **0.307** | **4.29** | 837 | **0.90** | **0.540** | **14.78** | **0.00** |

### LOL-v2 synthetic test (100 image pairs)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Latency ms (600×400) ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 15.73 | 0.740 | 0.302 | 4.88 | **5** | 4.83 | 0.797 | 15.10 | 4.54 |
| CLAHE only (baseline) | 11.92 | 0.510 | 0.367 | 4.56 | 12 | 2.72 | 0.774 | 24.01 | 0.38 |
| Zero-DCE++ official (baseline) | 18.03 | 0.826 | 0.184 | 4.45 | 225 | 3.42 | **0.921** | 14.83 | 0.40 |
| Classical only (ours) | 16.62 | 0.745 | 0.290 | 4.80 | 193 | 2.87 | 0.807 | 13.76 | **0.26** |
| Deep learning only (ours, fine-tuned) | **18.04** | **0.833** | **0.163** | 4.60 | 234 | 3.70 | 0.907 | **12.73** | 1.19 |
| Hybrid (ours) | 16.75 | 0.760 | 0.258 | **4.40** | 837 | **1.87** | 0.817 | 14.20 | 3.17 |

### Average over the three paired test sets (124 pairs, sets weighted equally)
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Residual noise σ ↓ | Edge preservation (EPI) ↑ | Colour shift ΔE2000 ↓ | New overexposure % ↓ |
|---|---|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 14.04 | 0.479 | 0.551 | 7.79 | 13.05 | 0.530 | 17.72 | 4.39 |
| CLAHE only (baseline) | 10.73 | 0.391 | 0.485 | 7.03 | 3.05 | 0.591 | 27.43 | 0.13 |
| Zero-DCE++ official (baseline) | 16.16 | 0.595 | 0.337 | 7.41 | 5.06 | 0.647 | 18.06 | 0.13 |
| Classical only (ours) | 16.76 | 0.621 | 0.467 | 6.87 | 5.36 | 0.530 | 16.49 | **0.09** |
| Deep learning only (ours, fine-tuned) | 17.65 | 0.589 | 0.349 | 7.58 | 6.62 | 0.628 | 15.22 | 0.40 |
| Hybrid (ours) | **17.86** | **0.768** | **0.264** | **4.29** | **1.27** | **0.661** | **13.48** | 1.08 |

### ExDark (real-world, no reference) (60 real-world night photos, no ground truth)
| Method | NIQE ↓ | Residual noise σ ↓ | Clipped pixels % ↓ | Colour cast ↓ | Mean brightness |
|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 4.68 | 4.18 | 9.23 | 7.2 | 0.48 |
| CLAHE only (baseline) | 4.65 | 2.22 | 1.51 | 10.2 | 0.20 |
| Zero-DCE++ official (baseline) | 4.29 | 2.88 | 1.48 | 16.6 | 0.35 |
| Classical only (ours) | 4.82 | 2.64 | 0.55 | 14.5 | 0.38 |
| Deep learning only (ours, fine-tuned) | 4.35 | 3.38 | 2.20 | 18.0 | 0.48 |
| Hybrid (ours) | 4.60 | 1.68 | 3.03 | 10.2 | 0.43 |

_Lower NIQE = more natural statistics. Brightness is reported, not ranked._

## Quality-aware update: before vs after (hybrid)

| Test set | Version | PSNR ↑ | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | Colourfulness | Blockiness ↓ | Detail |
|---|---|---|---|---|---|---|---|---|---|
| LOL-v1 eval15 | before | 19.18 | 0.779 | 0.206 | 4.05 | 1.18 | - | - | 1.01 |
| LOL-v1 eval15 | after | 18.91 | 0.775 | 0.226 | 4.17 | 1.05 | - | - | 1.00 |
| LOL-v2 synthetic test | before | 16.88 | 0.765 | 0.253 | 4.38 | 1.89 | - | - | 1.12 |
| LOL-v2 synthetic test | after | 16.75 | 0.760 | 0.258 | 4.40 | 1.87 | - | - | 1.12 |
| ExDark (real-world, no reference) | before | - | - | - | 4.18 | 2.10 | 0.39 | 1.21 | 0.047 |
| ExDark (real-world, no reference) | after | - | - | - | 4.60 | 1.68 | 0.28 | 1.23 | 0.042 |

_Detail: paired sets = sharpness relative to the reference (1 = equal); ExDark = absolute sharpness index._

## Ablation study (hybrid, one component removed at a time)

### LOL-v1 eval15
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 18.91 | +0.00 | 0.775 | 0.226 | 4.17 | 1.05 | 11.46 | 0.08 |
| w/o adaptive parameters (fixed) | 17.88 | -1.03 | 0.772 | 0.239 | 4.64 | 0.94 | 12.44 | 0.00 |
| w/o Zero-DCE++ branch (Retinex only) | 18.68 | -0.23 | 0.768 | 0.237 | 4.06 | 1.07 | 11.63 | 0.03 |
| w/o Retinex branch (Zero-DCE++ only) | 18.33 | -0.58 | 0.771 | 0.232 | 4.47 | 1.00 | 11.98 | 0.88 |
| w/o quality-aware curve scaling | 19.02 | +0.11 | 0.775 | 0.226 | 4.25 | 1.03 | 11.57 | 0.10 |
| w/o DL restoration (classical denoise) | 17.35 | -1.56 | 0.651 | 0.378 | 6.77 | 4.36 | 14.03 | **0.00** |
| w/o noise-map conditioning | 18.57 | -0.34 | 0.773 | 0.234 | 4.28 | 0.80 | 11.77 | 0.02 |
| w/o CLAHE | 19.12 | +0.21 | 0.772 | 0.254 | 5.25 | **0.53** | 11.36 | 0.00 |
| w/o colour correction | 19.09 | +0.18 | 0.779 | 0.213 | 4.18 | 1.13 | **11.18** | 0.05 |
| w/o highlight protection | 18.91 | +0.00 | 0.776 | 0.225 | 4.18 | 1.04 | 11.45 | 0.08 |
| w/o detail recovery | 18.91 | +0.00 | 0.776 | 0.226 | 4.12 | 1.04 | 11.46 | 0.08 |
| w/o quality-aware strategy | 19.18 | +0.27 | 0.779 | 0.207 | 4.11 | 1.19 | 11.22 | 0.08 |
| previous pipeline (before quality-aware update) | **19.18** | **+0.27** | **0.779** | **0.206** | **4.05** | 1.18 | 11.22 | 0.08 |

### LOL-v2 synthetic test
| Method | PSNR ↑ | ΔPSNR vs full | SSIM ↑ | LPIPS ↓ | NIQE ↓ | Noise σ ↓ | ΔE ↓ | Overexp. % ↓ |
|---|---|---|---|---|---|---|---|---|
| Hybrid (full) | 16.75 | +0.00 | 0.760 | 0.258 | 4.40 | 1.87 | 14.20 | 3.17 |
| w/o adaptive parameters (fixed) | 17.37 | +0.62 | 0.781 | 0.254 | **4.37** | 1.79 | 13.92 | 3.83 |
| w/o Zero-DCE++ branch (Retinex only) | 17.11 | +0.36 | 0.764 | 0.252 | 4.40 | 1.82 | 13.72 | 2.96 |
| w/o Retinex branch (Zero-DCE++ only) | 15.94 | -0.81 | 0.730 | 0.284 | 4.62 | 1.72 | 15.43 | 3.84 |
| w/o quality-aware curve scaling | **17.76** | **+1.01** | **0.791** | **0.246** | 4.39 | 2.13 | **12.65** | 3.69 |
| w/o DL restoration (classical denoise) | 16.55 | -0.20 | 0.742 | 0.272 | 5.07 | 3.12 | 14.06 | **0.21** |
| w/o noise-map conditioning | 16.52 | -0.23 | 0.746 | 0.318 | 4.82 | 1.05 | 15.08 | 2.59 |
| w/o CLAHE | 16.03 | -0.73 | 0.717 | 0.282 | 5.19 | **1.03** | 15.00 | 4.27 |
| w/o colour correction | 16.90 | +0.15 | 0.762 | 0.255 | 4.40 | 1.87 | 13.86 | 3.30 |
| w/o highlight protection | 16.84 | +0.08 | 0.762 | 0.256 | 4.40 | 1.85 | 14.14 | 3.09 |
| w/o detail recovery | 16.75 | -0.00 | 0.760 | 0.259 | 4.41 | 1.85 | 14.20 | 3.17 |
| w/o quality-aware strategy | 16.88 | +0.13 | 0.765 | 0.252 | 4.38 | 1.91 | 14.09 | 3.16 |
| previous pipeline (before quality-aware update) | 16.88 | +0.12 | 0.765 | 0.253 | 4.38 | 1.89 | 14.09 | 3.16 |

## Robustness across illumination conditions (synthetic, 30 unseen LOL-v2 synthetic references)
_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._

| Method | exposure x0.5 | x0.25 | x0.1 | x0.05 | x0.025 | spotlight (non-uniform) |
|---|---|---|---|---|---|---|
| Histogram equalisation (baseline) | 16.48 / 0.561 | 16.06 / 0.500 | 15.39 / 0.401 | 14.73 / 0.317 | 13.79 / 0.231 | 11.42 / 0.392 |
| Zero-DCE++ official (baseline) | 15.14 / 0.672 | 15.79 / 0.611 | 15.66 / 0.491 | 14.68 / 0.374 | 12.97 / 0.249 | 13.49 / 0.379 |
| Classical only (ours) | 16.84 / 0.707 | 15.58 / 0.637 | 15.15 / 0.543 | 15.08 / 0.485 | 14.99 / 0.425 | **14.13** / 0.494 |
| Deep learning only (ours, fine-tuned) | 10.22 / 0.602 | 11.33 / 0.565 | 13.04 / 0.474 | 14.15 / 0.368 | 14.12 / 0.242 | 12.71 / 0.359 |
| Hybrid (ours) | **19.25** / 0.817 | **17.85** / 0.797 | **16.83** / 0.757 | **16.13** / 0.706 | **15.97** / 0.646 | 13.96 / 0.668 |

_Cells: PSNR (dB) / SSIM._

## Auto mode: per-image method choice from no-reference checks

| Test set | Picks (classical / DL / hybrid) | Auto PSNR / SSIM | Always hybrid PSNR / SSIM | Oracle (best PSNR per image) |
|---|---|---|---|---|
| LOL-v1 eval15 | 0 / 0 / 15 | 18.91 / 0.775 | 18.91 / 0.775 | 20.43 / 0.669 |
| LOL-v2 real test (clean subset) | 0 / 0 / 9 | 17.91 / 0.770 | 17.91 / 0.770 | 19.53 / 0.580 |
| LOL-v2 synthetic test | 48 / 2 / 50 | 16.60 / 0.762 | 16.75 / 0.760 | 20.44 / 0.838 |
| ExDark (real-world, no reference) | 22 / 0 / 38 | NIQE 4.58 | NIQE 4.60 | - |

_The oracle needs the ground truth, so no app can reach it; it shows the headroom of choosing per image._

## Efficiency (CPU only)

| Method | 600×400 (ms) | 1280×960 (ms) | 2048×1536 upload, full-size output (ms) |
|---|---|---|---|
| Histogram equalisation (baseline) | 5 | 31 | 318 |
| CLAHE only (baseline) | 12 | 40 | 325 |
| Zero-DCE++ official (baseline) | 225 | 528 | 764 |
| Classical only (ours) | 193 | 1174 | 1430 |
| Deep learning only (ours, fine-tuned) | 234 | 538 | 760 |
| Hybrid (ours) | 837 | 2438 | 2730 |

| Model | Parameters | File size |
|---|---|---|
| Zero-DCE++ (fine-tuned) | 10,561 | 51.0 KB |
| Restoration U-Net | 144,579 | 574.6 KB |
