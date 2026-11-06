# Demo script (3 minutes) and viva prep

## Before judging (5 minutes earlier)
1. Open the live URL on **both** a laptop and a phone. Free Spaces sleep when idle, and the first load wakes them.
2. Have a few **unseen** dark photos ready on the phone (take one in a dim corridor). Judges may bring their own.
3. Keep `eval_P1_T76.md` and `results/figures/` open in another tab.

## 3-minute demo

| Time | Who | What to show / say |
|---|---|---|
| 0:00–0:20 | Darsh | "Night photos are dark, grainy and colour-shifted. Brightening alone amplifies noise and blows out lamps. Lumina restores them, and explains every decision." |
| 0:20–1:00 | Darsh | **Enhance tab:** tap the **first example** (a near-black meeting room), then drag the before/after slider. The 5th (neon sign) shows that coloured lights are kept. Read the **Recoverability** badge and one line of *What we noticed* aloud. Switch to *Natural night mood* to show user control. |
| 1:00–1:40 | Indhu | **Under the hood:** illumination map → Retinex gain ("dark corners get a lot of gain, lamps none, so no blow-outs") → noise-level map → removed noise ("it's grain, not edges"). |
| 1:40–2:10 | Karthikeya | **Compare methods:** histogram equalisation (noisy, blown out) vs Zero-DCE++ vs ours. "Classical alone leaves noise, DL alone over-brightens, the hybrid gets the best of both." |
| 2:10–2:40 | Dheeraj | **Benchmarks tab:** headline numbers (LOL-v2 real: best PSNR/SSIM/LPIPS/NIQE), the ablation chart ("removing the DL restorer costs the most"), one failure case. |
| 2:40–3:00 | Any | **Judge's photo:** "Want to try one of yours?" Upload it from the phone camera. |

## Viva: likely questions and crisp answers

**Why Zero-DCE++?** 10k parameters, CPU-fast, needs no paired data (but we had LOL, so we fine-tuned with supervision). It predicts *tone curves*: pixel-wise and monotone, so it can't hallucinate objects. That's a trust property we wanted.

**Why add your own restoration network?** A curve can't remove noise: it maps each pixel independently. Noise was the bottleneck. Removing the restorer costs about 1.5 dB PSNR and doubles LPIPS (ablation).

**How exactly is classical integrated with DL?** (1) The classical illumination map drives Retinex, which is fused per pixel with the DL candidate. (2) The gain implied by the illumination model becomes a **noise-level map** that conditions the restoration U-Net. Be honest about the ablation: the map gives about +0.2 dB on real photos but is a tie overall and slightly hurts on synthetic noise. (3) Classical diagnostics set the DL curve strength (quality-aware search).

**What makes it adaptive?** `analysis.choose_params`: brightness target, Retinex γ, CLAHE clip/weight, white-balance strength, saturation, denoise strength and curve strength are all functions of measured brightness, noise, lighting spread (in stops), colour cast and contrast. Ablation "fixed parameters" quantifies it.

**Guided filter vs Gaussian for illumination?** Gaussian-smoothed illumination bleeds across edges, so Retinex creates halos around lamps. The guided filter is edge-aware: smooth inside regions, sharp at boundaries.

**Why bilateral + guided for denoising in the classical path?** Bilateral on luma keeps edges whose contrast exceeds the noise (range σ tied to measured noise). Chroma noise is the ugliest low-light artefact and chroma has little detail, so we smooth it with a guided filter steered by the clean luma. On the training split two bilateral passes beat one pass, guided-only and NLM.

**How do you avoid overexposure?** The Retinex gain is capped at 25×; the quality-aware search penalises *newly* clipped pixels; fusion prefers the less-amplified candidate; a soft-knee highlight roll-off replaces hard clipping. The "new overexposure %" column measures it.

**Colour shift?** Retinex keeps colour ratios (reflectance untouched). White balance is partial and backs off when coloured light sources are detected. Measured with CIEDE2000.

**Train/validation/test?** LOL our485 split 435/50 with a fixed seed (`results/split.json`). Test = LOL eval15, LOL-v2 real and synthetic test splits, ExDark: all unseen. Design choices were decided on validation only.

**Metrics: what does each mean?** PSNR: pixel error (log scale). SSIM: structural similarity (local means, variances, covariance). LPIPS: deep-feature perceptual distance (lower = closer to human judgement). NIQE: no-reference naturalness (distance of image statistics from natural images; lower is better), used on ExDark where there's no ground truth. EPI: correlation of high-frequency edges with the reference. ΔE2000: perceptual colour difference.

**Why is PSNR only ~20 dB when papers report 25+?** Those are much larger GPU-trained models (millions of parameters) evaluated on LOL alone. Ours is 155k parameters, CPU-only, ~0.8 s per 600×400 image, and evaluated on four test sets including unseen cameras. Also, PSNR punishes exposure differences that are a matter of taste, which is why we let users choose.

**A failure case?** Live: tap the **last example** (dogs, near-black, heavily compressed). It is rated *Medium* and comes out with red/magenta blotchy noise, because compressed, spatially correlated noise is outside the restorer's training data (eval file §8, failure 5). Also see `results/failures/` and the eval file: near-black regions (no signal, so smudged texture), coloured lighting (neon/sodium read as a cast), and synthetic data that differs from LOL's noise. Be ready to explain one with the image on screen.

**Latency?** 0.8 s at 600×400 and 1.9 s at 1280×960 on a laptop CPU (see eval file §2.4 for the hosted app). The app downsizes uploads to ≤1280 px to stay well under 10 s on the free 2-vCPU host.

**Ethics?** No storage; temp files purged. Not for evidence or surveillance. The restorer predicts corrections, not content. The limitations are listed in the app itself.

## Who explains what
Roles are in `docs/DESIGN.md` §10 (swap freely). Each member should be able to open their files and walk through one function:
- **Indhu S:** `classical.py` (`estimate_illumination`, `retinex_enhance`, `denoise`) + `analysis.py`
- **Karthikeya D S:** `dce.py`, `restore.py`, `train_dce.py`, `train_restore.py`
- **Dheeraj Chinta:** `pipeline.py` (`fusion_weights`, `pyramid_blend`, `noise_map`), `evaluate.py`, ablation and failures
- **Darsh Chandran:** `app.py`, Dockerfile, deployment, docs
