"""Classical CV building blocks.

Every function takes / returns float32 RGB images in [0, 1] (HxWx3) unless noted.
Covers the mandatory classical requirements: illumination estimation + decomposition,
Retinex enhancement, CLAHE, edge-preserving (guided + bilateral) denoising, contrast and
colour correction.
"""
import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)  # Rec.601


def to_float(img_u8):
    return img_u8.astype(np.float32) / 255.0


def to_u8(img):
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def luminance(img):
    return img @ LUMA


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide, src, r, eps):
    """He et al. (2010) guided filter with a single-channel guide, O(N) via box filters.

    Output is locally a*guide + b: it smooths where the guide is flat (variance << eps)
    and keeps the guide's edges where variance >> eps -> edge-preserving, no halos.
    """
    if src.ndim == 3:
        return np.dstack([guided_filter(guide, src[..., c], r, eps) for c in range(src.shape[2])])
    m_g, m_s = _box(guide, r), _box(src, r)
    a = (_box(guide * src, r) - m_g * m_s) / (_box(guide * guide, r) - m_g * m_g + eps)
    b = m_s - a * m_g
    return _box(a, r) * guide + _box(b, r)


def estimate_illumination(img, floor=0.004):
    """Illumination map T (HxW): max-RGB prior refined with an edge-aware guided filter.

    T >= max(R,G,B) is the Retinex/LIME prior (I = R * T with reflectance R <= 1). Refinement
    happens in a gamma-encoded space so dark regions get as much edge detail as bright ones,
    and the radius scales with image size.
    """
    t0 = img.max(axis=2)
    g = np.sqrt(t0)
    r = max(3, int(0.015 * max(t0.shape)))
    t = guided_filter(g, g, r, 2e-3) ** 2
    return np.clip(t, floor, 1.0)


GAIN_CAP = 25.0  # past ~25x an 8-bit input has no signal left, only quantisation/JPEG noise


def retinex_gain(t, gamma):
    return np.minimum(t ** (gamma - 1.0), GAIN_CAP)


def solve_gamma(y, t, target, lo=0.2, hi=1.0, iters=14):
    """Bisection for the illumination exponent so the brightened image's mean luma hits target.

    Brightening is out = I * gain(T, gamma); smaller gamma -> brighter (monotone), so bisection works.
    Runs on a thumbnail, so it costs ~1 ms.
    """
    def mean_out(gm):
        return float(np.clip(y * retinex_gain(t, gm), 0, 1).mean())

    if mean_out(hi) >= target:
        return hi  # already bright enough: light touch
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_out(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def retinex_enhance(img, t, gamma):
    """Decompose I = R * T, compress the illumination (T**gamma) and recombine R * T**gamma.

    Reflectance (colour ratios + texture) is left untouched, so hue is preserved; the gain
    map T**(gamma-1) is returned for noise-aware post-processing and visualisation.
    """
    gain = retinex_gain(t, gamma)
    return img * gain[..., None], gain


def clahe(img, clip, weight=1.0, tiles=8):
    """CLAHE on the Lab lightness channel only (16-bit, so no banding); colour is unchanged.
    `weight` blends with the input to keep CLAHE from over-amplifying noise."""
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    l16 = (lab[..., 0] * 655.35).astype(np.uint16)
    eq = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tiles, tiles)).apply(l16)
    lab[..., 0] = (1 - weight) * lab[..., 0] + weight * eq.astype(np.float32) / 655.35
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)


def denoise(img, sigma):
    """Edge-preserving denoise in YCrCb.

    Luma: bilateral filter (range sigma tied to the noise level, so edges much stronger than
    noise survive). Chroma: guided filter steered by the clean luma -- colour noise is the
    most visible low-light artefact and chroma has little detail of its own.
    sigma is the noise std in 0..255 units.
    """
    s = max(float(sigma), 0.5) / 255.0
    ycc = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    for _ in range(2):  # two passes beat one (and guided-only / NLM) on the LOL training split
        y = cv2.bilateralFilter(y, 9, 3.0 * s, 4.0)
    c = guided_filter(y, ycc[..., 1:], 8, (2.0 * s) ** 2 + 1e-5)
    return cv2.cvtColor(np.dstack([y, c]).astype(np.float32), cv2.COLOR_YCrCb2RGB)


def gray_world_gains(img, lo=0.05, hi=0.95):
    """Per-channel white-balance gains from mid-tone pixels (ignores near-black and clipped)."""
    y = luminance(img)
    m = (y > lo) & (y < hi)
    means = img[m].mean(axis=0) if m.sum() > 100 else img.reshape(-1, 3).mean(axis=0)
    return np.clip(means.mean() / np.maximum(means, 1e-4), 0.7, 1.4).astype(np.float32)


def colour_correct(img, wb_strength, saturation):
    """Partial gray-world white balance + saturation adjustment (in HSV-free form: mix with luma)."""
    gains = 1 + wb_strength * (gray_world_gains(img) - 1)
    out = img * gains
    y = luminance(out)[..., None]
    return np.clip(y + saturation * (out - y), 0, None)


def enhance_details(img, beta, r=3):
    """Base/detail split with a guided filter; detail scaled by beta (>1 sharpens, <1 calms noise)."""
    y = luminance(img)
    base = guided_filter(y, y, r, 1e-3)
    y2 = base + beta * (y - base)
    return img * (np.maximum(y2, 0) / np.maximum(y, 1e-4))[..., None]


def protect_highlights(img, knee=0.82):
    """Soft-knee roll-off instead of hard clipping: luma above `knee` is compressed smoothly
    toward 1, and RGB is scaled by the same ratio so hue doesn't shift in bright areas."""
    y = luminance(img)
    k = 1 - knee
    y2 = np.where(y > knee, knee + k * np.tanh((y - knee) / k), y)
    out = img * (y2 / np.maximum(y, 1e-4))[..., None]
    return np.clip(out, 0, 1)


def adjust_exposure(img, factor):
    """Scale overall brightness by ~`factor` with a gamma on luma (black/white points and hue kept)."""
    if abs(factor - 1) < 1e-3:
        return img
    y = luminance(img)
    m = float(np.clip(y.mean(), 1e-3, 0.95))
    g = np.log(np.clip(m * factor, 0.02, 0.9)) / np.log(m)
    y2 = np.power(np.clip(y, 0, 1), g)
    return np.clip(img * (y2 / np.maximum(y, 1e-4))[..., None], 0, 1)


def histogram_equalize(img):
    """Baseline: global HE on luma (what a naive tool does)."""
    ycc = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return to_float(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB))
