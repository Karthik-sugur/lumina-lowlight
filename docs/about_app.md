## How Lumina works (30-second version)

1. **Understand the photo.** We estimate how light fell on the scene (an *illumination map*) and measure darkness, sensor noise, uneven lighting, colour cast and coloured light sources.
2. **Choose settings for *this* photo.** Every strength (brightening, contrast, white balance, noise removal) is derived from those measurements; nothing is a fixed number.
3. **Brighten two ways, keep the best of each.** Classical *Retinex* re-lighting and a fine-tuned deep curve model (*Zero-DCE++*) each produce a candidate. A per-pixel quality score (good exposure plus real detail) blends them.
4. **Restore.** A small restoration network removes the noise that brightening uncovered. It is told *where* noise was amplified most, using a map computed from the illumination estimate.
5. **Respect your intent.** *Natural night mood* keeps a photo feeling like night; *Maximum visibility* reveals the most. The brightness slider fine-tunes either.

## Designed for people, not just metrics

| Principle | What it means here |
|---|---|
| **Explain, don't just output** | "What we noticed / what we did" in plain language, plus every intermediate step in *Under the hood*. |
| **Honest expectations** | A *Recoverability* rating tells you up front when there was too little light to recover fine detail. Warnings flag blown-out or still-grainy areas. |
| **No invented content** | Brightening uses per-pixel tone curves and Retinex gain. The restoration network only predicts small residual corrections, and nothing is generated from scratch. What you see was in the photo. |
| **You stay in control** | Intent presets and a brightness slider. *Auto* is the recommended default, never forced. |
| **Fast and light** | Runs on a plain CPU in a few seconds; the two networks together are about 155k parameters (around 0.6 MB). |
| **Accessible** | Works on phones (camera capture supported), large touch targets, colour-blind-safe heatmaps (cividis), text explanations for every visual. |

**Who it helps:** anyone with a phone photo taken in a dim room or at night; people with low vision who need to *see what's there*; field and inspection workers documenting dim environments; and anyone who wants to understand *why* an enhancement looks the way it does.

## Privacy
Photos are processed in memory on the server. Temporary copies the web framework needs (uploads and your download file) are deleted automatically within about an hour. Nothing is logged, kept, or used for training.

## Limitations (known failure modes)
- **Almost no light = almost no information.** In near-black regions the result can be smudgy or grainy. No method can recover what the sensor never captured.
- **Coloured lighting is ambiguous.** A sodium-orange street or neon sign can be "colour cast" or "the real colour". We detect coloured light sources and back off white balance, but we sometimes guess wrong.
- **Blown-out lamps stay blown out.** Pixels that were already pure white carry no detail to recover. We try to limit *new* clipping, but it is our weakest measured point (about 1 % of pixels on average, more on bright scenes).
- **Training data bias.** The networks were trained on LOL (indoor scenes, one camera type) plus synthetic sensor noise. Heavily compressed or video-frame noise is outside what they learned and can remain as coloured grain.
- **Heavy JPEG compression** in the dark shows up as blocky artefacts once brightened.

## Responsible use
Enhanced images are **for viewing, not evidence**. Restoration changes pixel values. Don't use outputs for forensic identification, surveillance of people, or to make claims about what an original photo "proves". Faces and number plates may look clearer, but they are not more reliable.

## Credits
Zero-DCE++ (Li, Guo & Loy, TPAMI 2021), LOL (Wei et al., BMVC 2018), LOL-v2 (Yang et al., TIP 2021), ExDark (Loh & Chan, CVIU 2019). The guided filter (He et al., ECCV 2010), CLAHE (Zuiderveld 1994), Laplacian-pyramid blending (Burt & Adelson 1983) and exposure-fusion weights (Mertens et al. 2007) are re-implemented from their papers.
