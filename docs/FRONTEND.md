# Web frontend (LUMINA) and engine API

The primary interface is the team's **LUMORA v4** design (`lumora-v4.html`, by a teammate), rebranded **LUMINA** and connected to the real restoration engine. It is a static site on **Vercel**; the Python engine stays on **Hugging Face Spaces**.

```
Browser ──► Vercel (static: web/index.html, ui.js, lumina.js, config.js)
   │
   └──► Hugging Face Space (darshuuuu/lumina-lowlight)
          POST /gradio_api/call/restore   (api.py, registered with gr.api in app.py)
          GET  /gradio_api/call/restore/{event_id}   → server-sent event "complete" with JSON
          GET  /gradio_api/info           → health check
          (the classic Gradio UI keeps running at the same address as a fallback)
```

## What came from the teammate, what changed

| Part | Status |
|---|---|
| Markup, CSS, typography, gradients, lens hero, animations, 6 pages, routing, burger menu | **Unchanged** (`web/index.html` is generated from their file; inline scripts moved to files for the Content-Security-Policy) |
| UI behaviour: sliders + tooltips, tabs, zoom, drag/keyboard before–after divider, synced zoom on Compare | **Unchanged**, now in `web/ui.js` |
| Name **LUMORA** → **LUMINA** (title, nav, hero, about, footer, downloads) | Changed |
| Text that described an unconnected Retinexformer backend or "processed in your browser" | Rewritten to describe what actually runs |
| In-browser "classical" approximation, synthetic sample scenes, localhost FastAPI stubs | **Replaced** by real engine results (`web/lumina.js`) |

## Endpoint audit (what the HTML expected vs our backend)

| Expected by `lumora-v4.html` | In our backend? | Now handled by |
|---|---|---|
| `GET /api/health` | no | `GET /gradio_api/info` (status chip: connecting / online / offline) |
| `POST /api/analyze` | no | diagnosis in the `restore` response (Analysis tab) |
| `POST /api/enhance/deep`, `/api/enhance/hybrid` | no | `restore` → `results.dl`, `results.hybrid` |
| `POST /api/compare` | no | `restore` returns all three methods at once |
| `/api/upload`, `/api/restore/*`, `/api/metrics`, `/api/result/{id}` (stubs that only threw errors) | no | removed; one `restore` call covers them |
| classical in the browser (gain/gamma/blur/sharpen) | n/a | the real classical pipeline on the engine (`results.classical`) |

**Why one endpoint instead of the five FastAPI routes:** the free Hugging Face tier only allows a Gradio (ZeroGPU) Space; a separate FastAPI server would need other hosting. `gr.api` adds a typed JSON endpoint to the existing app without touching the Gradio UI, the models or the weights, and one request returns everything, so switching modes, the Comparison Lab and the Technology previews are instant after a single run.

## Every control and what it does

| Control | Behaviour |
|---|---|
| Drag & drop / browse / Replace | JPG, PNG, WebP up to 15 MB; EXIF-orientation aware; photos over 2048 px are reduced to 2048 px before upload **and the status line says so with both sizes**. The engine enhances at ≤1280 px and returns results at the uploaded size (≤2048 px) with full-size detail transfer; the status line states the processing and output sizes |
| Sample photos | 6 real unseen test photos (LOL-v2 real clean subset, ExDark) |
| **Auto Enhance** | resets fine-tuning, Intent = Auto, Brightness = 1.0, runs the engine and shows the **recommended** method |
| **Restore image** | runs all three pipelines on the engine (one request), shows the selected mode |
| Mode: Classical / Deep Learning / Hybrid | shows that pipeline's real output; instant after a run |
| Engine settings: Intent, Brightness | change how the engine restores (re-run with Restore) |
| Fine-tune sliders (exposure, contrast, saturation, denoising, sharpness) | the teammate's adjustments, applied **in the browser on top of the engine result**; all at 0 = the engine's exact output |
| Analysis tab | **Image quality diagnostics** (Illumination Low/Medium/High (+ uneven), Noise Low/Medium/High, Sharpness Soft/Balanced/Sharp, Colour quality, Compression, Resolution, Recoverability), each with the measured value behind the label; **Enhancement breakdown** (detected cases such as *Extremely dark*, *Uneven illumination*, *Compressed (JPEG blocks)*, *Soft / blurry*, and every stage's decision for this photo with the reason); then the raw measurements, what we noticed / did and the quick in-browser estimate |
| Download | PNG of the displayed result (`lumina-<method>.png`, `-finetuned` when adjusted) |
| Zoom − / + / Fit, before–after divider | unchanged teammate behaviour |
| Comparison Lab | Original + 3 methods with synced zoom, no-reference metrics (brightness, grain, blown-out, colour cast, detail, colourfulness, blockiness), time per method, recommendation with its full formula, a download per pane |
| Technology | 7 stages, each with this photo's real intermediate output (illumination map, reflectance, Retinex gain, candidates, curve strength, fusion weights, noise map, removed noise, recovered detail, quality checks) |
| Benchmarks | the evaluation's numbers (`results/metrics.json` → `web/benchmarks.json` via `scripts/export_web_data.py`) |

Metrics on a user's photo are **no-reference** checks; PSNR, SSIM and LPIPS need a ground-truth image, so they appear only on Benchmarks (measured on reference datasets).

## Security and limits

- **No secrets in the frontend.** The engine API is public, stateless and stores nothing. `config.js` holds only the public engine URL, injected at build time from `LUMINA_API_BASE` (`web/build.mjs`); there is no localhost default.
- **CORS:** the Space answers cross-origin requests with the caller's origin. `api.py` also supports an optional `ALLOWED_ORIGINS` allow-list (with wildcards, e.g. `https://*.vercel.app`); it is unset, because there are no cookies or credentials to protect.
- **Headers** (`web/vercel.json`): Content-Security-Policy (scripts only from the site; network only to the site and `*.hf.space`; no framing), `nosniff`, `Referrer-Policy`, `Permissions-Policy`. Vercel adds HSTS.
- **Sizes and time:** browser limit 15 MB, upload ≤2048 px JPEG, engine limit 15 MB decoded and ≥32 px per side (it also caps at 2048 px and says so); results are WebP at the uploaded size (about 0.35–0.5 MB for a 600×400 photo, more for 2048 px). The client times out after 120 s (covers a cold start of the free Space) with a retry message; stale responses are discarded if a new photo was loaded.

## Run locally

```bash
python app.py                                    # engine + classic UI on :7860
cd web && LUMINA_API_BASE=http://127.0.0.1:7860 node build.mjs && python -m http.server 8080 --directory dist
```

## Deploy

```bash
cd web
npx vercel@latest link --project lumina-t76
npx vercel@latest env add LUMINA_API_BASE production   # https://darshuuuu-lumina-lowlight.hf.space
npx vercel@latest deploy --prod
```

## Tested (2026-11-07)

`tests/test_api.py` (11 tests: three real outputs, resizing, PNG/greyscale, intent, bad inputs, recommendation, origin allow-list). In the browser, locally and on the production URL:
- upload → restore (after the image-quality-aware update, production): 4.3 s on desktop for a 600×400 sample (0.96 s on the engine for all three pipelines), 2.4 s on a 375 px mobile profile, 9.2 s for a 2048×1365 photo returned at full size (4.19 s on the engine); earlier, before the update: 2.15 s / 1.86 s;
- Analysis tab: image-quality diagnostics and enhancement breakdown render on desktop and mobile; status line states processing and output sizes; Benchmarks show the de-duplicated evaluation, the before/after card and Auto mode;
- mode switching, fine-tune, Analysis, all five downloads (valid PNGs), Comparison Lab, all 7 Technology stages (every image loads), Benchmarks numbers equal to the evaluation;
- errors: GIF, >15 MB, corrupt file, 12×12 and 16×16 images (rejected by the engine's own validation), engine unreachable → message + Retry;
- 6000×4000 photo (resized, note shown), near-black (recoverability Medium), very noisy (Low), spotlight (Classical recommended);
- mobile: no horizontal scroll, burger menu, stacked layout.

## Limitations

- The fine-tune sliders are browser-side adjustments on top of the engine output, not engine parameters (the engine's own controls are Intent and Brightness).
- The free Space sleeps after 48 h without visitors; the first request then takes up to a minute (the UI says so).
- Per-photo recommendation is a transparent heuristic on no-reference checks, not a ground-truth quality score.
