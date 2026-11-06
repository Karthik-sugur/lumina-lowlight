# Design document: Lumina, robust low-light image restoration (P1)

| | |
|---|---|
| Problem | P1: Robust Low-Light Image Restoration and Enhancement |
| Team | T76: Indhu S, Karthikeya D S, Dheeraj Chinta, Darsh Chandran |
| Live app | see README |
| Status | Complete: all mandatory and advanced requirements implemented and evaluated |

---

## 1. Problem and goals

Photos taken in the dark are underexposed, noisy and colour-shifted. Naive brightening (gamma, histogram equalisation) amplifies noise, blows out light sources, creates halos and shifts colours. We want a system that **estimates the illumination, adapts to each image, suppresses noise, restores colour and keeps edges and texture**, and that a person can **trust and understand**.

**Goals**
1. Output that is visibly and measurably better than classical-only and DL-only approaches on unseen images (PSNR, SSIM, LPIPS, NIQE).
2. Adaptive: no fixed enhancement strength; everything derived from measured image properties.
3. Runs on a plain CPU in seconds (deployed on a free 2-vCPU host; must answer within about 10 s).
4. Human-centred: explains itself, sets honest expectations, keeps the user in control, never invents content.

**Non-goals:** RAW-domain processing, video, generative "hallucinated" detail.

## 2. Users and human-centred design

| Persona | Need | How the design answers it |
|---|---|---|
| Phone user with a dim indoor or night photo | "Make it look good, quickly, from my phone" | One-tap enhance, camera capture, results in ~1–3 s, before/after slider, PNG download |
| Person with low vision | "Show me what's actually there" | *Maximum visibility* preset; text explanations; high-contrast UI; colour-blind-safe heatmaps |
| Field / inspection worker documenting dim spaces | "Don't make things up" | Curve-based brightening (pixel-wise, monotone) plus a residual-only restorer; a *recoverability* rating says when detail cannot be trusted |
| Curious user / judge | "Why does it look like that?" | *What we noticed / What we did* in plain language; every intermediate map in *Under the hood*; *Compare methods* tab |

**Design principles:** explain, don't just output · honest uncertainty (recoverability rating, blown-out / residual-grain warnings) · user agency (intent presets, brightness slider; *Auto* recommended, never forced) · privacy by default (in-memory processing, temp files purged within an hour, nothing logged) · accessibility (mobile layout, large targets, text alternatives for every visual, cividis colour maps).

**Interface states:** empty (prompt plus examples) · loading (Gradio progress on every output) · success (slider, summary, diagnosis, actions, quality table) · warning (inline ⚠️ lines) · error (friendly `gr.Error` messages for: no image, non-image, < 32 px, processing failure; never a stack trace). Images over 1280 px are downscaled with a visible note.

## 3. Requirements traceability

| Requirement (problem statement) | Where in the code | Evidence |
|---|---|---|
| **A. Illumination estimation and decomposition** | `classical.estimate_illumination` (max-RGB prior + guided-filter refinement), `retinex_enhance` (I = R·T) | *Under the hood*: Illumination map T, Reflectance R |
| **A. Retinex-based enhancement** | `classical.retinex_enhance`, `solve_gamma` | Classical-only method; Retinex candidate in hybrid |
| **A. CLAHE** | `classical.clahe` (Lab L channel, 16-bit, adaptive clip and weight) | ablation "w/o CLAHE" |
| **A. Edge-preserving denoising (bilateral / guided)** | `classical.denoise` (2× bilateral on luma + luma-guided filter on chroma), `pipeline.classical_denoise` (gain-aware) | Classical method; ablation "w/o DL restoration (classical denoise)" |
| **A. Contrast and colour correction** | `classical.clahe`, `colour_correct` (partial gray-world + saturation), `protect_highlights` | ablations "w/o colour correction", "w/o highlight protection" |
| **A. Adaptive parameter selection** | `analysis.diagnose` → `analysis.choose_params` (table §5.4) | ablation "w/o adaptive parameters (fixed)" |
| **B. Implement / adapt a low-light architecture** | `dce.py`: Zero-DCE++ re-implemented, official weights loaded | baseline "Zero-DCE++ official" |
| **B. Fine-tune where data is available** | `scripts/train_dce.py` (LOL train split, supervised L1+SSIM + zero-reference regularisers); `scripts/train_restore.py` | `results/train_log_*.json`, figure 4 |
| **B. Generalisation across illumination conditions** | LOL-v2 real + synthetic + ExDark (unseen), synthetic exposure sweep ×0.5 … ×0.025 + spotlight | eval file §Robustness, figure 2 |
| **B. Image-quality-aware enhancement** | `pipeline.choose_dce_scale` (quality search over curve strength); quality-weighted fusion; noise-map-conditioned restoration | ablation "w/o quality-aware curve scaling" |
| **C. Integrate classical illumination with DL restoration** | `pipeline.noise_map` (from illumination/gain) → `restore.RestoreNet` conditioning input | ablation "w/o noise-map conditioning" (separately trained net) |
| **C. Adaptive select / combine outputs** | `pipeline.fusion_weights` + `pyramid_blend` | ablations "w/o Retinex branch", "w/o Zero-DCE++ branch"; fusion weight map |
| **C. Preserve edges and textures, minimise noise amplification** | gain cap, amplification-aware fusion, guided-filter T (no halos), restoration with noise map | EPI and residual-noise columns |
| **C. Extremely dark and non-uniform inputs** | exposure sweep down to ×0.025; spotlight condition; adaptive target for high lighting spread | figure 2 |
| **Adv 1. Adaptive strategy** | §5.4 | ablation (fixed parameters) |
| **Adv 2. Compare classical / DL / hybrid** | `pipeline.METHODS` (+ 3 baselines) | eval file §1, *Compare methods* tab |
| **Adv 3. Ablation** | `pipeline.ABLATIONS` (9 removals) | eval file §2, figure 3 |
| **Adv 4. Overexposure, colour shift, detail loss** | `protect_highlights`, gain cap, coloured-light-aware WB; measured by new-overexposure %, ΔE2000, EPI | eval file columns |
| **Adv 5. Unseen images** | All test sets disjoint from training/tuning; app examples from test sets | eval file |
| Metrics: PSNR, SSIM, LPIPS, NIQE, latency, noise and edges | `lowlight/metrics.py`, `scripts/evaluate.py` | eval file |
| Deliverable: deployed app, comparative eval, intermediate visualisations, failure analysis | `app.py`, `Dockerfile`, `eval_P1_T76.md`, `results/failures/` | live URL |

## 4. Data

| Set | Use | Size |
|---|---|---|
| LOL-v1 `our485` | **train 435 / validation 50**, seeded split (`results/split.json`) | 485 pairs, 600×400 |
| LOL-v1 `eval15` | test (paired) | 15 |
| LOL-v2 real `Test` | test, unseen capture conditions (paired) | 100 |
| LOL-v2 synthetic `Test` | test, synthetic darkening of RAISE photos (paired) | 100 |
| ExDark (384 px release) | test, real-world night photos, **no reference** (letterbox trimmed) | 60 sampled (seed 0) |

There is no leakage: the test sets are never read by `train_dce.py`, `train_restore.py`, or when choosing any constant. Checkpoint selection uses only the 50-image validation split.

## 5. Method

### 5.1 Classical stages
- **Illumination T:** max(R,G,B) (Retinex/LIME prior: reflectance ≤ 1), refined with a **guided filter** in √-space (radius 1.5% of image size) so T is smooth within regions but follows object edges. This is why Retinex brightening produces no halos at lamp/wall boundaries.
- **Retinex re-lighting:** output = I · min(T^(γ−1), 25). γ is solved per image by bisection on a 256-px thumbnail so mean brightness hits the adaptive target. The 25× cap exists because beyond it an 8-bit input holds only quantisation/JPEG noise; we found this cause of "confetti" artefacts in near-black regions during development.
- **CLAHE** on Lab lightness in 16-bit (no banding), blended at an adaptive weight.
- **Denoising (classical path):** two bilateral passes on luma (range σ tied to the measured noise), chroma smoothed by a guided filter steered by the clean luma. The bilateral pair beat a single pass, a guided-only variant and OpenCV NLM on the LOL **training** split (PSNR 16.24 vs 16.07 / 16.11 / 16.10).
- **Colour:** partial gray-world white balance (strength from measured cast, reduced 70% when coloured light sources are detected) + saturation compensation; **soft-knee highlight roll-off** instead of clipping.

### 5.2 Deep-learning stages
- **Zero-DCE++** (re-implemented in `dce.py`, official weights loaded and verified): 10,561 parameters; predicts a per-pixel curve map applied 8×. **Fine-tuned** on our LOL train split with L1 + 0.3·(1−SSIM) + Zero-DCE's illumination-smoothness (TV) and colour-constancy terms as regularisers. Adam 3e-4, cosine schedule, 256-px crops, flips. Validation PSNR 14.32 → 16.81 dB. Stopped at epoch 8 after the plateau (≤0.05 dB change over 5 epochs).
- **Restoration U-Net** (`restore.py`, ours): 3 scales (16/32/64 channels), residual output, 144,579 parameters. Input = hybrid front-end output + 1-channel **noise-level map**. Trained 4000 iterations, batch 16, 128-px crops, OneCycle LR 1e-3, L1 + 0.2·(1−SSIM). Training inputs are produced by the exact inference code (`pipeline.prepare`), so there is no train/test mismatch.
  - **Training data = 435 real LOL pairs + 435 synthetic pairs.** Each synthetic pair degrades one LOL *training* reference physically: exposure ×0.02–×0.5 in linear light (log-uniform), Poisson shot noise and Gaussian read noise. Real LOL pairs only cover LOL's own noise range. Without the synthetic pairs, the network stopped denoising once the noise map went beyond it: on validation references darkened ×0.025, the restorer output was 15.8 dB while plain classical denoising reached 19.6. Checkpoints are selected on the mean of real and synthetic **validation** PSNR (50 + 50 held-out images).
- **Quality-aware strategy:** (a) the DL curve strength is chosen from 20 candidates (×0.1–×2.0) by a no-reference quality score (|brightness − target| + 2 × newly clipped fraction); (b) fusion weights are per-pixel quality scores; (c) the restorer is conditioned on the measured noise.

### 5.3 Hybrid integration
1. Classical T + diagnosis → adaptive parameters.
2. Retinex candidate and fine-tuned Zero-DCE++ candidate.
3. **Per-pixel fusion** by Mertens-style quality (well-exposedness × blurred-Laplacian contrast) **÷ amplification** (1 + (gain/12)²). This prefers the candidate that needed less gain, which suppresses noise amplification. Blended with a 5-level **Laplacian pyramid** (seamless, no halos).
4. Highlight guard → adaptive CLAHE → colour correction.
5. **Noise map** = noise σ measured on the brightened image (all channels) × relative local gain, derived from the illumination model. It is fed to the restoration U-Net, which removes noise where it was amplified and keeps texture elsewhere.
6. User intent (preset × slider) applied last as an exposure adjustment, so the restorer always sees its training distribution.

### 5.4 Adaptive parameter rules (`analysis.choose_params`)

| Parameter | Rule (monotone interpolation of a measured property) | Why |
|---|---|---|
| Target brightness | 0.42 (LOL train GT mean = 0.456) × intent × interp(lighting spread 4→7 stops, 1→0.75) | night scenes with lamps should stay night-like |
| Retinex γ | bisection to reach target | per-image exposure |
| DCE curve strength | argmin of quality score over 0.5…2.0 | quality-aware |
| CLAHE clip | interp(contrast 0.3→1.2, 2.5→1.2) | low-contrast images get more |
| CLAHE weight | interp(expected noise 4→30, 0.6→0.15) | don't boost grain |
| White balance | interp(cast 3→15, 0→0.8) × (0.3 if coloured lights) | keep neon / sodium colours |
| Saturation | interp(brightness 0.02→0.25, 1.15→1.0) | darker images lose more chroma |
| Detail β (classical) | interp(expected noise 3→25, 1.15→0.85) | sharpen clean images only |
| Denoise strength | 90th percentile of the noise map | measured, not guessed |

### 5.5 Alternatives tried and rejected
- **Illumination-guided Zero-DCE++** (T as a 4th input channel, zero-initialised): after 4 epochs it was indistinguishable from plain Zero-DCE++ (same validation PSNR to 0.01 dB, 4th-channel weights ≈ 0.003). A curve-only network cannot denoise, and noise was the bottleneck, so the integration effort moved to the noise-map-conditioned restoration stage.
- **Uncapped Retinex gain:** produced coloured "confetti" in near-black JPEG regions (160× gain). Replaced by the 25× cap plus the amplification-aware fusion.
- **Input-domain noise estimate for the map:** under-estimates noise in 8-bit/JPEG dark images (e.g. σ = 0.6 on a visibly grainy ExDark photo). Replaced by measuring on the brightened image.
- **NLM denoising (classical path):** slower and no better than two bilateral passes on the training split.
- **Curve-strength grid ×0.5–×2.0:** moderately dark inputs always picked the lower bound (×0.5) and were still over-brightened. A search pinned to its boundary means the range was too narrow, so it was widened to ×0.1–×2.0.
- **"Exposure lock"** (forcing the restorer to keep the input's mean brightness): it helped on LOL-v2-derived data, whose references are darker (mean luma 0.371) than LOL's (0.453 test, 0.456 train). On our **validation** split it *cost* 0.6 dB, so we rejected it. Its apparent benefit was a dataset brightness convention, not robustness. (How we found this: a test-time robustness sweep exposed a weakness; every candidate fix was then decided on the validation split only, so test data never selected a design.)

## 6. Evaluation methodology
- **Paired sets:** PSNR, SSIM (Gaussian 11×11, σ=1.5, RGB), LPIPS (AlexNet, pyiqa), NIQE (pyiqa), residual noise σ (Immerkær), edge preservation index (Laplacian correlation with ground truth), CIEDE2000 colour shift, newly over-exposed pixel %, brightness error.
- **Unpaired (ExDark):** NIQE, residual noise, clipping, colour cast, brightness.
- **Robustness:** 30 unseen LOL-v2-real references degraded physically (linear light × k, Poisson shot noise at 800 e⁻ full-well, Gaussian read noise 0.002, sRGB re-encode, 8-bit) at k ∈ {0.5, 0.25, 0.1, 0.05, 0.025} plus a strongly non-uniform spotlight.
- **Latency:** sequential single process, after warm-up, mean of 6 LOL images (600×400) and 3 runs at 1280×960, laptop CPU (no GPU).
- **Ablation:** hybrid with one component removed, on LOL-v1 eval15 + LOL-v2 real.

Results, ablation and failure analysis: **[`eval_P1_T76.md`](../eval_P1_T76.md)**. Headline (test sets, hybrid): 18.41 dB / 0.781 SSIM / 0.227 LPIPS / 4.16 NIQE averaged over 215 unseen pairs (best of six methods on 7 of 8 metrics); best NIQE on ExDark; best from ×0.25 to ×0.025 exposure.

**Model selection (validation only).** Restorer v2 (LOL pairs only) vs v3 (LOL + synthetic pairs), full pipeline, 50 held-out real pairs + their synthetic degradations: v2 19.65 / 17.54 dB (real / synthetic), v3 19.41 / 19.90 dB. **v3 selected**: −0.24 dB on real, +2.36 dB on synthetic, real/synthetic gap 0.49 vs 2.11 dB. On the test sets (seen only afterwards) v3 gained on LOL-v1 (+0.62 dB) and extreme darkness, but lost 0.94 dB on LOL-v2 real; reported, not re-selected.

**Noise-map conditioning, honestly:** it helps on real captures (+0.25 dB v2, +0.20 dB v3 on validation), hurts on synthetic noise (−1.48 / −0.21 dB), and is a tie overall for v3 (19.65 vs 19.66). It is a small, real-data-specific benefit.

## 7. Deployment and reproducibility
- `docker build -t lumina . && docker run -p 7860:7860 lumina`, or `pip install -r requirements.txt torch && python app.py`.
- Primary interface: the LUMINA web app (team's LUMORA v4 design) on Vercel, calling the engine's JSON API (`api.py`) cross-origin; details in [`FRONTEND.md`](FRONTEND.md).
- Engine hosted on a Hugging Face Space (Gradio SDK, free ZeroGPU hardware, CPU inference), which also serves the classic Gradio interface as a fallback. The Dockerfile is for self-hosting and was **not built in our environment** (no Docker available); the pip + `python app.py` route is the tested one.
- Reproduce everything: `python scripts/download_data.py && python scripts/train_dce.py && python scripts/train_restore.py && python scripts/train_restore.py --no-map && python scripts/evaluate.py && python scripts/report.py`. Seeds are fixed, and the split is saved in `results/split.json`.
- Tests: `python -m pytest -q` (filter properties, gamma solver, every method and ablation, tiny/flat/extreme inputs).
- No secrets anywhere: the app needs no API keys.

## 8. Limitations
See also *About* in the app. In short: (1) no method recovers detail below the noise floor, and near-black regions stay soft; (2) coloured lighting vs colour cast is ambiguous; (3) already-clipped highlights can't be recovered; (4) the restorer was trained on LOL (indoor, one sensor family) plus synthetic sensor noise; compressed/correlated noise (video frames, heavy JPEG) is outside its training distribution; (5) heavy JPEG blocking is amplified; (6) PSNR and SSIM reward matching a dataset's exposure convention (our three worst LOL-v2 cases gain ~9 dB from one global brightness gain), which isn't necessarily what a user wants (hence presets); (7) highlight protection runs before the restorer, which can re-clip highlights; the ablation shows it has no measurable effect, so overexposure (1.08 % new clipped pixels on average) is our weakest metric.

## 9. Ethics and responsible use
- **No hallucination by design:** brightening is pixel-wise and monotone; restoration is a residual correction conditioned on measured noise, with no generative prior.
- **Not for evidence or identification:** stated in the app. Enhancement changes pixel values; faces and plates may *look* clearer without being more reliable.
- **Privacy:** no storage or logging; temp files are purged automatically.
- **Dataset licences:** LOL, LOL-v2 and ExDark are research datasets, used for research/education only; example images are attributed in `examples/SOURCES.md`.
- **Accessibility and inclusion:** mobile-first, colour-blind-safe visuals, plain-language explanations.

## 10. Team roles
| Member | USN | Area |
|---|---|---|
| Indhu S | 1RUA24CSE0170 | Classical pipeline (illumination, Retinex, CLAHE, denoising, colour), adaptive parameters |
| Karthikeya D S | 1RUA24CSE0195 | Deep learning: Zero-DCE++ fine-tuning, restoration U-Net, training |
| Dheeraj Chinta | 1RUA24CSE0127 | Hybrid fusion, evaluation, ablation, failure analysis |
| Darsh Chandran | 1RUA24CSE0115 | App/UX, deployment, documentation |

## 11. Future work
Fine-tune the restorer on more diverse real data (e.g. SID sRGB, phone captures); a learned recoverability/uncertainty map; tiled processing for full-resolution phone images; on-device (ONNX/TFLite) export. Together the two networks are about 155k parameters, which makes on-device use realistic.
