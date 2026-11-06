# Demo script (3 minutes) and viva prep

## Before judging (5 minutes earlier)
1. Open **https://lumina-t76.vercel.app** on **both** a laptop and a phone. The status chip should read **Engine online · v3**. The free engine sleeps after 48 h idle, so a first request can take up to a minute; opening the page early wakes it.
2. Keep the fallback ready: **https://darshuuuu-lumina-lowlight.hf.space** (classic interface, same engine).
3. Have a few **unseen** dark photos on the phone (take one in a dim corridor). Judges may bring their own.
4. Keep `eval_P1_T76.md` and `results/figures/` open in another tab.

## 3-minute demo (LUMINA web app)

| Time | Who | What to show / say |
|---|---|---|
| 0:00–0:20 | Darsh | Home page: "Night photos are dark, grainy and colour-shifted. Brightening alone amplifies noise and blows out lamps. LUMINA restores them, and explains every decision." |
| 0:20–1:00 | Darsh | **Studio:** tap the **Meeting room** sample, press **Auto Enhance**, drag the before/after divider. Switch the mode between Classical, Deep Learning and Hybrid (instant). Open the **Analysis** tab: read the *Image quality diagnostics* (Illumination / Noise / Sharpness / Colour / Recoverability, each with its measured value) and the *Enhancement breakdown* ("this is why it denoised first / did or didn't sharpen"). In **Adjust → Engine settings**, pick *Natural night* and press Restore to show user control. |
| 1:00–1:40 | Indhu | **Technology:** step through the stages: illumination map → reflectance and Retinex gain ("dark corners get a lot of gain, lamps none") → fusion weights → noise-level map → removed noise ("it's grain, not edges"). All are real outputs for this photo. |
| 1:40–2:10 | Karthikeya | **Compare:** Original vs the three pipelines with synced zoom and per-method metrics. "Deep learning alone leaves grain; classical leaves noise and colour cast; the recommendation is computed per photo, and it is not always Hybrid." |
| 2:10–2:40 | Dheeraj | **Benchmarks:** 18.23 dB / 0.771 SSIM on 124 unseen pairs (say: "we found that LOL-v2 real duplicates LOL-v1 training images and removed them"), the Classical vs Deep vs Hybrid chart, the before/after card of the quality-aware update (an honest trade-off), the ablation table ("removing the DL restorer costs the most on real photos"), one failure case. |
| 2:40–3:00 | Any | **Judge's photo:** "Want to try one of yours?" In the Studio, tap the upload area on the phone and take a photo or pick one. |

**Honest limitation to show if asked:** the *Near-black (hard)* sample comes out rated *Medium* with coloured grain, because compressed, spatially correlated noise is outside the restorer's training data (eval file §8, failure 5).

## Viva: likely questions and crisp answers

**Why Zero-DCE++?** 10k parameters, CPU-fast, needs no paired data (but we had LOL, so we fine-tuned with supervision). It predicts *tone curves*: pixel-wise and monotone, so it can't hallucinate objects. That's a trust property we wanted.

**Why add your own restoration network?** A curve can't remove noise: it maps each pixel independently. Noise was the bottleneck. Removing the restorer costs 1.92 dB PSNR and raises LPIPS from 0.207 to 0.375 on eval15 (ablation).

**How exactly is classical integrated with DL?** (1) The classical illumination map drives Retinex, which is fused per pixel with the DL candidate. (2) The gain implied by the illumination model becomes a **noise-level map** that conditions the restoration U-Net. Be honest about the ablation: on the clean test sets removing the map costs 0.37 / 0.26 dB, but on validation it was a near tie, so we call it a small benefit. (3) Classical diagnostics set the DL curve strength (quality-aware search).

**What makes it adaptive?** `analysis.choose_params`: brightness target, Retinex γ, CLAHE clip/weight, white-balance strength, saturation, denoise strength and curve strength are all functions of measured brightness, noise, lighting spread (in stops), colour cast and contrast. Ablation "fixed parameters" quantifies it.

**Guided filter vs Gaussian for illumination?** Gaussian-smoothed illumination bleeds across edges, so Retinex creates halos around lamps. The guided filter is edge-aware: smooth inside regions, sharp at boundaries.

**Why bilateral + guided for denoising in the classical path?** Bilateral on luma keeps edges whose contrast exceeds the noise (range σ tied to measured noise). Chroma noise is the ugliest low-light artefact and chroma has little detail, so we smooth it with a guided filter steered by the clean luma. On the training split two bilateral passes beat one pass, guided-only and NLM.

**How do you avoid overexposure?** The Retinex gain is capped at 25×; the quality-aware search penalises *newly* clipped pixels; fusion prefers the less-amplified candidate; a soft-knee highlight roll-off replaces hard clipping. The "new overexposure %" column measures it.

**Colour shift?** Retinex keeps colour ratios (reflectance untouched). White balance is partial and backs off when coloured light sources are detected. Measured with CIEDE2000.

**Train/validation/test?** LOL our485 split 435/50 with a fixed seed (`results/split.json`). Test = LOL eval15, LOL-v2 real (clean subset), LOL-v2 synthetic, ExDark. Design choices were decided on validation only.

**Any data leakage?** We found one and fixed it. LOL-v2 real re-packages LOL-v1: a pixel-hash check showed 86 of its 100 test inputs and 91 of 100 references are identical to LOL-v1 our485 images (our training data). We now evaluate only the 9 LOL-v2 real pairs with no duplicate, moved the robustness sweep to LOL-v2 synthetic references, and replaced a demo photo that was a training scene. Earlier LOL-v2-real numbers are withdrawn (eval file §1).

**What does "image-quality-aware" mean here?** Before choosing anything, the engine measures illumination and its spread, noise (before and after brightening), sharpness after denoising, colourfulness and cast, JPEG blockiness and resolution. Those decide: denoise and deblock *before* brightening (compressed photos), a lower gain cap (small/compressed photos), a saturation ceiling, and how much detail to recover *after* noise removal (none if already sharp or still noisy; only detail above the measured noise; halo clamp). The Analysis tab shows every decision and why.

**Why was the flower pixelated and oversaturated before?** Three measured causes: Retinex divides by max(R,G,B), so chroma noise in near-black pixels became saturated colour; a darkness-based saturation boost plus CLAHE added more; and an 11–21× gain on a 300-px JPEG made invisible 8×8 blocks visible. Fixes: chroma denoise + block-boundary deblock before brightening, gain cap 14× for small/compressed photos (and a colourfulness ceiling on the classical path). We first also capped saturation in the hybrid, *before* restoration; the test set showed that it fired on colour noise, so we re-validated the placement on fresh validation data with a rule fixed in advance, and removed it from the hybrid (eval file §9.6). Be open that this decision was prompted by a test-set observation.

**And the soft living room?** The restoration network removes fine texture with the noise, and the engine returned 1280 px for a 3872-px photo, which the browser upscaled. Now: detail recovery after restoration, and results are returned at the uploaded size (≤ 2048 px) with the photo's own above-noise detail transferred back, never invented.

**Metrics: what does each mean?** PSNR: pixel error (log scale). SSIM: structural similarity (local means, variances, covariance). LPIPS: deep-feature perceptual distance (lower = closer to human judgement). NIQE: no-reference naturalness (distance of image statistics from natural images; lower is better), used on ExDark where there's no ground truth. EPI: correlation of high-frequency edges with the reference. ΔE2000: perceptual colour difference.

**Why is PSNR only ~18–19 dB when papers report 25+?** Those are much larger GPU-trained models (millions of parameters) evaluated on LOL alone, and LOL-v2 real results can be inflated by its overlap with LOL-v1 training images. Ours is 155k parameters, CPU-only, ~0.8 s per 600×400 image, and evaluated on four de-duplicated test sets including unseen cameras. Also, PSNR punishes exposure differences that are a matter of taste, which is why we let users choose.

**A failure case?** Live: tap the **last example** (dogs, near-black, heavily compressed). It is rated *Medium* and comes out with red/magenta blotchy noise, because compressed, spatially correlated noise is outside the restorer's training data (eval file §8, failure 5). Also see `results/failures/` and the eval file: near-black regions (no signal, so smudged texture), coloured lighting (neon/sodium read as a cast), and synthetic data that differs from LOL's noise. Be ready to explain one with the image on screen.

**Latency?** See eval file §2.4 (laptop CPU and the hosted app). The engine enhances at ≤ 1280 px to stay fast on the free 2-vCPU host, and returns results at the uploaded size (≤ 2048 px) with detail transfer; any resizing is stated in the app.

**Ethics?** No storage; temp files purged. Not for evidence or surveillance. The restorer predicts corrections, not content. The limitations are listed in the app itself.

## Who explains what
Roles are in `docs/DESIGN.md` §10 (swap freely). Each member should be able to open their files and walk through one function:
- **Indhu S:** `classical.py` (`estimate_illumination`, `retinex_enhance`, `denoise`) + `analysis.py`
- **Karthikeya D S:** `dce.py`, `restore.py`, `train_dce.py`, `train_restore.py`
- **Dheeraj Chinta:** `pipeline.py` (`fusion_weights`, `pyramid_blend`, `noise_map`), `evaluate.py`, ablation and failures
- **Darsh Chandran:** `app.py`, Dockerfile, deployment, docs
