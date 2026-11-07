"""Image diagnosis -> adaptive parameters -> plain-language explanation.

This is the "adaptive parameter selection based on image characteristics" requirement: nothing
in the pipeline uses a fixed strength. Each rule is a short monotone interpolation so it can be
explained to a user (and in a viva) in one sentence.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .classical import block_grid, luminance

# Global knobs, set from the LOL training split only (target vs the training references' mean
# brightness 0.456). Presets scale the target.
TARGET_BRIGHTNESS = 0.42
KNEE = 0.82
# Image-quality-aware strategy knobs, chosen on validation data only: scripts/validate.py writes the
# evidence to results/validation/ (eval_P1_T76.md section 9)
LOW_INFO_SIDE = 400     # shorter side below this = little real information (lower gain cap)
BLOCKY = 1.3            # JPEG blockiness above this = visible 8x8 blocks (PNG 1.01-1.09, JPEG q60 2-4.6)
PRE_DENOISE_AT = 10.0   # expected grain (/255) after brightening above which chroma is denoised first
REINJECT = 0.6          # share of restoration-removed detail (above the noise floor) put back
DETAIL_FINE = 0.3       # fine-texture sharpening amount (before noise / softness scaling)
DETAIL_MID = 0.15       # local-contrast amount (validation: results/validation/sharpening_round1.json)
DEBLOCK_FINE = 0.0      # fine-sharpening share kept on deblocked JPEGs (validation: compressed proxy)
SOFT, SHARP = 0.010, 0.030  # restored-image sharpness index: clean references span ~0.012-0.040
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
    sharpness: float = 0.0  # detail index of the exposure-normalised input (higher = crisper)
    edge_strength: float = 0.0  # p90 gradient of the exposure-normalised input
    blockiness: float = 1.0  # JPEG 8x8 grid energy ratio (1 = none, >1.3 = visible blocking)
    chroma: float = 0.0     # mean Lab chroma relative to lightness (colourfulness)
    width: int = 0
    height: int = 0


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
    pre_denoise: bool = False   # edge-preserving chroma denoise of the dark input before brightening
    deblock: bool = False       # also flatten JPEG 8x8 block edges before brightening
    gain_cap: float = 25.0      # max Retinex gain (lower when the photo holds little information)
    max_colour: float = 0.55    # colourfulness ceiling (chroma/lightness); LOL references p90 = 0.39
    detail_reinject: float = 0.0  # filled in after restoration (adaptive detail recovery)
    detail_fine: float = 0.0
    detail_mid: float = 0.0
    residual_noise: float = 0.0   # grain left after noise removal (/255), measured
    sharp_restored: float = 0.0   # sharpness index after noise removal, measured


def estimate_noise(y):
    """Immerkaer (1996) fast noise std estimate on luma, skipping the 10% strongest edges."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(y * 255.0, -1, k))[1:-1, 1:-1]
    grad = cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    flat = grad <= np.percentile(grad, 90)  # <= keeps it non-empty on perfectly flat images
    return float(np.sqrt(np.pi / 2) * resp[flat].mean() / 6.0)


def sharpness_index(img):
    """Exposure-normalised detail: mean |Laplacian| of luma/mean after a sigma-1 blur, so sensor
    grain contributes little and dark vs bright photos are comparable. Higher = crisper."""
    y = luminance(np.clip(img, 0, 1)).astype(np.float32)
    y = y / (y.mean() + 1e-3)
    return float(np.abs(cv2.Laplacian(cv2.GaussianBlur(y, (0, 0), 1.0), cv2.CV_32F)).mean())


def edge_strength(img):
    y = luminance(np.clip(img, 0, 1)).astype(np.float32)
    y = cv2.GaussianBlur(y / (y.mean() + 1e-3), (0, 0), 1.0)
    return float(np.percentile(cv2.magnitude(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)), 90))


def blockiness(img):
    """JPEG blocking: luma steps on an 8-px grid vs elsewhere, best over the 8 grid offsets (works
    for cropped / shifted images). ~1.0 = no blocking; > 1.3 = visible 8x8 blocks."""
    y = luminance(np.clip(img, 0, 1)).astype(np.float32)
    return float(np.mean([r for r, _ in block_grid(y / (y.mean() + 1e-3))]))


def colourfulness(img):
    """Mean Lab chroma of mid-tone pixels divided by their mean lightness (scale-free saturation)."""
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    m = (lab[..., 0] > 3) & (lab[..., 0] < 97)
    if m.sum() < 100:
        return 0.0
    return float(np.hypot(lab[..., 1], lab[..., 2])[m].mean() / (lab[..., 0][m].mean() + 1e-3))


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
        sharpness=sharpness_index(img),
        edge_strength=edge_strength(img),
        blockiness=blockiness(img),
        chroma=colourfulness(img),
        width=int(img.shape[1]),
        height=int(img.shape[0]),
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
    blocky = d.blockiness > BLOCKY
    low_info = blocky or min(d.width, d.height) < LOW_INFO_SIDE
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
        pre_denoise=bool(blocky or eff_noise > PRE_DENOISE_AT),
        deblock=bool(blocky),
        gain_cap=14.0 if low_info else 25.0,
    )


def detail_strengths(p, d, residual_noise, sharp_restored, recoverability_level):
    """Adaptive detail-recovery amounts, decided AFTER noise removal (sharpen only what is clean).
    Less when residual grain is visible, when the restored image is already sharp, when the
    photo is blocky (fine sharpening would sharpen the blocks) or when little signal exists."""
    noise_factor = float(np.interp(residual_noise, [1.5, 6.0], [1.0, 0.3]))
    soft_factor = float(np.interp(sharp_restored, [SOFT, SHARP], [1.0, 0.25]))
    trust = {"High": 1.0, "Medium": 0.8, "Low": 0.5}.get(recoverability_level, 0.8)
    p.detail_reinject = REINJECT * noise_factor * trust
    p.detail_fine = DETAIL_FINE * noise_factor * soft_factor * trust * (DEBLOCK_FINE if p.deblock else 1.0)
    p.detail_mid = DETAIL_MID * noise_factor * soft_factor * trust
    return p


def quality_labels(d, p, sharp_restored, residual_noise, level, sizes=None):
    """Plain-language image-quality diagnostics (each value is a real measurement, labelled)."""
    illum = "Low" if d.brightness < 0.10 else "Medium" if d.brightness < 0.30 else "High"
    grain = p.denoise_sigma
    noise = "Low" if grain < 6 else "Medium" if grain < 14 else "High"
    sharp = "Soft" if sharp_restored < SOFT else "Sharp" if sharp_restored > SHARP else "Balanced"
    if d.coloured_lights:
        colour = "Coloured lighting (kept)"
    elif d.colour_cast > 8:
        colour = f"{d.cast_hue.capitalize()} cast"
    else:
        colour = "Natural"
    return {
        "illumination": {"label": illum + (", uneven" if d.lighting_spread > 3 else ""), "value": f"{d.brightness * 100:.0f}% brightness, {d.lighting_spread:.1f} stops spread"},
        "noise": {"label": noise, "value": f"grain σ {grain:.1f}/255 after brightening; {residual_noise:.1f} after restoration"},
        "sharpness": {"label": sharp, "value": f"detail index {sharp_restored:.3f} after noise removal (clean photos ≈ 0.012–0.040)"},
        "colour": {"label": colour, "value": f"cast {d.colour_cast:.0f}, colourfulness {d.chroma:.2f}"},
        "compression": {"label": "JPEG blocking" if p.deblock else "Clean", "value": f"blockiness {d.blockiness:.2f} (1.0 = none)"},
        "resolution": {"label": "Low" if min(d.width, d.height) < 400 else "OK",
                       "value": sizes or f"{d.width}×{d.height} px analysed"},
        "recoverability": {"label": level, "value": ""},
    }


def strategy(d, p, f):
    """Which operations were selected for this photo and why (for the UI's enhancement breakdown)."""
    gain = max(p.target / max(d.brightness, 1e-3), 1.0)
    cases = []
    if d.brightness < 0.04:
        cases.append("Extremely dark")
    elif d.brightness < 0.15 and p.denoise_sigma >= 10:
        cases.append("Dark and noisy")
    elif d.brightness < 0.15:
        cases.append("Dark")
    if d.lighting_spread > 3:
        cases.append("Uneven illumination")
    if d.brightness >= p.target * 0.9:
        cases.append("Already well lit")
    if p.deblock:
        cases.append("Compressed (JPEG blocks)")
    if 0 < p.sharp_restored < SOFT:
        cases.append("Soft / blurry")
    ops = []
    q = f.get("quality", True)
    ops.append(("Brightening", f"Retinex re-lighting + deep curves; reaching the target needs about {gain:.1f}× on average" + (f" (Retinex gain capped at {p.gain_cap:.0f}×)" if gain > p.gain_cap else "") if gain > 1.05
                else "skipped - the photo is already bright enough", "brightness relative to target"))
    if q and p.pre_denoise:
        ops.append(("Noise control before brightening", "chroma denoised" + (" and JPEG blocks flattened" if p.deblock else ""),
                    "brightening multiplies noise; cleaning first stops random colours and blocks being amplified"))
    if q and p.gain_cap < 25:
        ops.append(("Gain limit", f"capped at {p.gain_cap:.0f}×", "low-resolution or compressed photo: little real information in the shadows"))
    ops.append(("Local contrast (CLAHE)", f"{p.clahe_weight * 100:.0f}% strength", "kept low when grain or blocks would be boosted"))
    ops.append(("Noise removal", f"restoration network, told where grain is (up to σ {p.denoise_sigma:.0f}/255)", "noise grows where shadows were lifted; removed BEFORE any sharpening"))
    if f.get("details", True):
        if p.detail_reinject + p.detail_fine + p.detail_mid > 0.02:
            ops.append(("Detail recovery", f"restore {p.detail_reinject * 100:.0f}% of lost detail, fine sharpening {p.detail_fine:.2f}, local contrast {p.detail_mid:.2f}",
                        "only detail above the noise floor is kept, and a halo guard limits overshoot"))
        else:
            ops.append(("Detail recovery", "skipped", "already sharp, or too noisy to sharpen safely"))
    ops.append(("Colour", f"white balance {p.wb_strength * 100:.0f}%, saturation ×{p.saturation:.2f}",
                "fix casts without over-saturating; colour noise is removed before brightening (JPEGs) and by restoration"))
    return {"cases": cases or ["Moderately dark"], "operations": [{"stage": a, "decision": b, "why": c} for a, b, c in ops]}



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
    if p.pre_denoise:
        did.insert(0, "Cleaned colour noise" + (" and JPEG block edges" if p.deblock else "")
                   + " in the dark photo before brightening, so they were not amplified.")
    if p.detail_reinject + p.detail_fine + p.detail_mid > 0.02:
        did.append(f"Recovered detail after noise removal (re-injected up to {p.detail_reinject * 100:.0f}% of the lost "
                   f"detail, sharpening {p.detail_fine:.2f} fine / {p.detail_mid:.2f} mid), only where it is stronger "
                   "than the measured noise, with a halo guard.")
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
