"""Image diagnosis -> adaptive parameters -> plain-language explanation.

This is the "adaptive parameter selection based on image characteristics" requirement: nothing
in the pipeline uses a fixed strength. Each rule is a short monotone interpolation so it can be
explained to a user (and in a viva) in one sentence.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .classical import luminance

# Global knobs, tuned on the LOL training split only (scripts/tune.py). Presets scale the target.
TARGET_BRIGHTNESS = 0.42
KNEE = 0.82
PRESETS = {  # user intent -> brightness multiplier
    "Auto (balanced)": 1.0,
    "Natural night mood": 0.72,
    "Maximum visibility": 1.25,
}


@dataclass
class Diagnosis:
    brightness: float       # mean luma, 0..1
    dark_fraction: float    # share of pixels with luma < 0.1
    clipped_fraction: float  # share already blown out (>= 0.98)
    noise_sigma: float      # input luma noise std, 0..255 units (Immerkaer)
    lighting_spread: float  # stops between bright (p90) and dark (p10) illumination
    colour_cast: float      # chroma of the mean mid-tone colour (Lab), 0 = neutral
    cast_hue: str
    contrast: float         # RMS contrast of luma
    coloured_lights: bool   # saturated light sources (neon, sodium lamps) present


@dataclass
class Params:
    target: float
    gamma: float            # filled in by the pipeline (solved per image)
    denoise_sigma: float    # expected output noise to remove, 0..255 units
    clahe_clip: float
    clahe_weight: float
    wb_strength: float
    saturation: float
    detail_beta: float
    knee: float
    dce_scale: float = 1.0  # filled in by the quality-aware DCE exposure search


def estimate_noise(y):
    """Immerkaer (1996) fast noise std estimate on luma, skipping the 10% strongest edges."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(y * 255.0, -1, k))[1:-1, 1:-1]
    grad = cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 90)  # <= keeps it non-empty on perfectly flat images
    return float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6.0)


_HUES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]


def diagnose(img, t):
    """Measure the things that decide how an image should be enhanced. `t` = illumination map."""
    y = luminance(img)
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
    mid = (y > 0.02) & (y < 0.95)
    a, b = (lab[..., 1][mid].mean(), lab[..., 2][mid].mean()) if mid.sum() > 100 else (0.0, 0.0)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    bright = y > max(0.3, np.percentile(y, 99) * 0.6)
    angle = (np.degrees(np.arctan2(b, a)) + 360) % 360
    lt = np.log2(t + 1 / 255)
    return Diagnosis(
        brightness=float(y.mean()),
        dark_fraction=float((y < 0.1).mean()),
        clipped_fraction=float((img.max(axis=2) >= 0.98).mean()),
        noise_sigma=estimate_noise(y),
        lighting_spread=float(np.percentile(lt, 90) - np.percentile(lt, 10)),
        colour_cast=float(np.hypot(a, b)),
        cast_hue=_HUES[int(((angle + 22.5) % 360) // 45)],
        contrast=float(y.std() / (y.mean() + 1e-3)),
        coloured_lights=bool((bright & (chroma > 35)).mean() > 0.002),
    )


def choose_params(d, target_mult=1.0, adaptive=True):
    """Map a Diagnosis to pipeline parameters. adaptive=False gives the fixed-parameter baseline
    used in the ablation study."""
    target, knee = TARGET_BRIGHTNESS * target_mult, KNEE
    if adaptive:  # scenes with strong light sources (street lamps, neon) are meant to look like night
        target *= float(np.interp(d.lighting_spread, [4.0, 7.0], [1.0, 0.75]))
    else:
        return Params(target=target, gamma=0.5, denoise_sigma=12.0, clahe_clip=2.0, clahe_weight=0.5,
                      wb_strength=0.5, saturation=1.1, detail_beta=1.0, knee=knee)
    gain = max(target / max(d.brightness, 1e-3), 1.0)  # expected average amplification
    eff_noise = d.noise_sigma * gain                   # noise the viewer would see after brightening
    return Params(
        target=target,
        gamma=1.0,
        denoise_sigma=float(np.clip(eff_noise, 0, 40)),
        clahe_clip=float(np.interp(d.contrast, [0.3, 1.2], [2.5, 1.2])),
        clahe_weight=float(np.interp(eff_noise, [4, 30], [0.6, 0.15])),
        wb_strength=float(np.interp(d.colour_cast, [3, 15], [0.0, 0.8]) * (0.3 if d.coloured_lights else 1.0)),
        saturation=float(np.interp(d.brightness, [0.02, 0.25], [1.15, 1.0])),
        detail_beta=float(np.interp(eff_noise, [3, 25], [1.15, 0.85])),
        knee=knee,
    )


def explain(d, p):
    """Plain-language 'what we noticed' / 'what we did' bullet lists for the UI."""
    seen = []
    level = ("extremely dark" if d.brightness < 0.04 else "very dark" if d.brightness < 0.1
             else "dim" if d.brightness < 0.25 else "reasonably exposed")
    seen.append(f"Image is **{level}** (average brightness {d.brightness * 100:.0f}%, "
                f"{d.dark_fraction * 100:.0f}% of pixels near-black).")
    if d.lighting_spread > 3:
        seen.append(f"Lighting is **uneven**: bright and dark areas differ by ~{d.lighting_spread:.1f} stops.")
    if d.noise_sigma * max(p.target / max(d.brightness, 1e-3), 1) > 8:
        seen.append(f"Sensor **noise** will become visible once brightened (input noise sigma {d.noise_sigma:.1f}).")
    if d.colour_cast > 5:
        seen.append(f"A **{d.cast_hue} colour cast** is present (strength {d.colour_cast:.0f}).")
    if d.coloured_lights:
        seen.append("**Coloured light sources** detected (e.g. neon/street lamps) - their colour is kept.")
    if d.clipped_fraction > 0.01:
        seen.append(f"{d.clipped_fraction * 100:.1f}% of the image is already blown out (lamps/sky).")

    did = [
        f"Lifted shadows with Retinex re-lighting (illumination exponent gamma={p.gamma:.2f}, aiming at "
        f"{p.target * 100:.0f}% brightness) - already-lit areas get little or no gain.",
        f"Ran the deep curve model (Zero-DCE++) and picked curve strength x{p.dce_scale:.2f} after checking "
        "candidates for exposure and blown-out pixels; blended both results pixel-by-pixel by quality.",
        f"Local contrast (CLAHE) clip {p.clahe_clip:.1f} applied at {p.clahe_weight * 100:.0f}% "
        "(kept low on noisy images so grain isn't boosted).",
        f"White balance correction {p.wb_strength * 100:.0f}%"
        + (" (reduced to keep coloured lights)" if d.coloured_lights else "") + f", saturation x{p.saturation:.2f}.",
        "Removed noise with a restoration network that is told where noise was amplified most "
        f"(measured grain after brightening up to ~{p.denoise_sigma:.0f}/255).",
        f"Highlights rolled off softly above {p.knee * 100:.0f}% instead of clipping.",
    ]
    return seen, did


def recoverability(d, p):
    """Honest expectation-setting: how much real information is there to recover?
    Uses the grain measured *after* brightening (p.denoise_sigma): in the dark input itself, 8-bit
    quantisation hides how noisy the signal really is."""
    if d.brightness < 0.015 or p.denoise_sigma > 25:
        return "Low", "Very little light reached the sensor. Expect smudged detail and residual noise; nothing can recover what wasn't captured."
    if d.brightness < 0.06 or p.denoise_sigma > 10:
        return "Medium", "Enough signal for a clear result, but fine textures in the darkest areas may be softened by noise removal."
    return "High", "Plenty of signal - the result should be clean and faithful."


def as_dict(x):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(x).items()}
