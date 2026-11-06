# System architecture

Every box below is a function in the code; file and function names are given so the diagram can be checked against the source.

```mermaid
flowchart TD
    U([User: phone / laptop browser]) -->|upload / camera / example| APP["app.py — Gradio UI<br/>input validation, resize ≤1280 px,<br/>explanations, quality checks"]
    APP --> RUN["pipeline.run(img, 'hybrid', preset, strength)"]

    subgraph S1["① Classical analysis (lowlight/classical.py, lowlight/analysis.py)"]
        T["estimate_illumination<br/>max-RGB prior + guided filter → T"]
        D["diagnose<br/>brightness, noise σ (Immerkær), lighting spread,<br/>colour cast, coloured lights, contrast"]
        PR["choose_params + solve_gamma<br/>adaptive γ, CLAHE clip/weight, WB, saturation"]
        T --> D --> PR
    end

    subgraph S2["② Two brightening candidates"]
        RX["retinex_enhance<br/>I/T · T^γ (gain-capped)"]
        DC["Zero-DCE++ (lowlight/dce.py)<br/>fine-tuned on LOL train split"]
        QA["choose_dce_scale<br/>quality-aware curve strength<br/>(exposure error + new clipping)"]
        DC --> QA
    end

    subgraph S3["③ Adaptive fusion (pipeline.py)"]
        FW["fusion_weights<br/>exposedness × contrast ÷ amplification"]
        PB["pyramid_blend<br/>Laplacian pyramid, 5 levels"]
        HG["protect_highlights<br/>soft-knee roll-off"]
        FW --> PB --> HG
    end

    subgraph S4["④ Contrast & colour (classical.py)"]
        CL["clahe (Lab L, 16-bit)"] --> CC["colour_correct<br/>partial gray-world + saturation"]
    end

    subgraph S5["⑤ DL restoration (lowlight/restore.py)"]
        NM["noise_map<br/>σ measured after brightening × relative gain"]
        RN["RestoreNet U-Net (145k params)<br/>input: RGB + noise map → residual"]
        NM --> RN
    end

    RUN --> T
    PR --> RX & DC
    RX --> FW
    QA --> FW
    HG --> CL
    CC --> RN
    HG --> NM
    RN --> UI["adjust_exposure (user intent)"]
    UI --> OUT([Result + before/after slider + diagnosis +<br/>what-we-did + quality check + intermediate maps])

    style S5 fill:#eef4fc,stroke:#2a78d6
    style S1 fill:#f4fbf7,stroke:#1baf7a
```

## Methods compared (same code path, different switches)

| Method | Path in `pipeline.run` |
|---|---|
| Histogram equalisation (baseline) | `classical.histogram_equalize` |
| CLAHE only (baseline) | `classical.clahe` |
| Zero-DCE++ official (baseline) | published weights, no fine-tuning |
| **Classical only** | ① → Retinex → highlight guard → gain-aware bilateral + guided denoise → ④ → detail enhancement |
| **Deep learning only** | fine-tuned Zero-DCE++ alone |
| **Hybrid** | ① → ② → ③ → ④ → ⑤ (diagram above) |

Ablations (`pipeline.ABLATIONS`) switch off one stage at a time using the same function.

## Training and evaluation flow

```mermaid
flowchart LR
    DL[scripts/download_data.py] --> LOL[(LOL our485)]
    LOL -->|seeded split 435/50| TD[scripts/train_dce.py<br/>Zero-DCE++ fine-tune]
    TD --> W1[weights/dce_finetuned.pth]
    W1 --> TR[scripts/train_restore.py<br/>inputs = pipeline.prepare<br/>LOL pairs + synthetic dark pairs]
    LOL --> TR
    TR --> W2[weights/restore_net.pth<br/>+ restore_net_nomap.pth for ablation]
    W1 & W2 --> EV[scripts/evaluate.py]
    T1[(LOL eval15)] & T2[(LOL-v2 real test)] & T3[(LOL-v2 synthetic test)] & T4[(ExDark sample)] --> EV
    EV --> RP[scripts/report.py] --> R[results/tables.md + figures + failures]
```

## Deployment

Live app: Hugging Face Space using the **Gradio SDK on free ZeroGPU hardware** (Docker and CPU Spaces now require a paid plan). Inference runs on the Space's CPU; the GPU is never requested, which avoids GPU queueing and the daily quota. A `Dockerfile` (python:3.12-slim + CPU PyTorch) is included for self-hosting. The app has no state and no database, and stores nothing. Gradio's temporary files are purged hourly (`delete_cache`), and download files after an hour.
