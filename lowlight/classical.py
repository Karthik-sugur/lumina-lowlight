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


def retinex_gain(t, gamma, cap=GAIN_CAP):
    return np.minimum(t ** (gamma - 1.0), cap)


def solve_gamma(y, t, target, lo=0.2, hi=1.0, iters=14, cap=GAIN_CAP):
    """Bisection for the illumination exponent so the brightened image's mean luma hits target.

    Brightening is out = I * gain(T, gamma); smaller gamma -> brighter (monotone), so bisection works.
    Runs on a thumbnail, so it costs ~1 ms.
    """
    def mean_out(gm):
        return float(np.clip(y * retinex_gain(t, gm, cap), 0, 1).mean())

    if mean_out(hi) >= target:
        return hi  # already bright enough: light touch
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_out(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def retinex_enhance(img, t, gamma, cap=GAIN_CAP):
    """Decompose I = R * T, compress the illumination (T**gamma) and recombine R * T**gamma.

    Reflectance (colour ratios + texture) is left untouched, so hue is preserved; the gain
    map T**(gamma-1) is returned for noise-aware post-processing and visualisation.
    """
    gain = retinex_gain(t, gamma, cap)
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


# ---------------------------------------------------------------------------------------------
# Image-quality-aware additions: denoise BEFORE brightening, and noise-gated detail recovery.
# ---------------------------------------------------------------------------------------------

def core(d, t):
    """Soft-threshold ('coring'): detail smaller than the noise floor t is treated as noise and
    dropped, larger detail keeps its sign and loses t. This is what stops sharpening from
    amplifying grain or JPEG blocks and calling it 'recovered detail'."""
    return np.sign(d) * np.maximum(np.abs(d) - t, 0)


def block_grid(y):
    """[(ratio, k) for columns, rows]: strength of luma steps on an 8-px grid vs elsewhere, at the
    best of the 8 offsets k (the step between pixel k+8n and k+8n+1). ratio ~1 = no blocking."""
    out = []
    for axis in (1, 0):
        d = np.abs(np.diff(y, axis=axis)).mean(axis=1 - axis)
        if d.size < 32:
            return [(1.0, 0), (1.0, 0)]
        ratios = [d[k::8].mean() / (np.delete(d, np.s_[k::8]).mean() + 1e-6) for k in range(8)]
        k = int(np.argmax(ratios))
        out.append((float(ratios[k]), k))
    return out


def pre_denoise(img, sigma_in, deblock=False):
    """Edge-preserving denoise of the DARK input, before any brightening multiplies its noise.

    Chroma is smoothed with a guided filter steered by luma: in near-black pixels the dominant
    channel is often just noise, and Retinex would turn it into a saturated colour. With
    deblock=True, luma at the 8x8 block boundaries also gets a mild self-guided filter whose eps
    matches the quantisation step of a compressed dark JPEG, so block edges (1-2 levels) are
    flattened while real edges and the texture inside blocks survive. sigma_in: input noise std in 0..255 units."""
    ycc = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2YCrCb)
    y = ycc[..., 0]
    s = max(float(sigma_in), 0.5) / 255.0
    if deblock:  # only the pixels either side of each 8x8 block boundary: interior texture is kept
        (_, kx), (_, ky) = block_grid(y)
        m = np.zeros_like(y)
        for k, view in ((kx, m.T), (ky, m)):
            view[k::8] = 1
            view[k + 1::8] = 1
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 0.7) * 1.5, 0, 1)
        y = y + m * (guided_filter(y, y, 2, (2.5 / 255) ** 2 + (1.5 * s) ** 2) - y)
    c = guided_filter(y, ycc[..., 1:], 4, (4 * s) ** 2 + (2 / 255) ** 2)
    return np.clip(cv2.cvtColor(np.dstack([y, c]).astype(np.float32), cv2.COLOR_YCrCb2RGB), 0, 1)


def recover_detail(x, ref, noise_sigma, reinject, fine, mid, ref_sigma=None):
    """Adaptive detail recovery on luma (colour is untouched; RGB scaled by the luma ratio).

    1. Re-injection: detail present in `ref` (the image before noise removal) that noise removal
       took away, but only the part above the noise floor (coring), weighted by `reinject`.
    2. Two-scale unsharp masking: detail layers against Gaussian blurs (sigma 1 = fine texture,
       sigma 3 = local contrast), each cored at the residual noise, scaled by `fine` / `mid`.
    3. Halo clamp: the result may not overshoot the 3x3 local min/max of the unsharpened image by
       more than a small margin, which prevents bright/dark rims on high-contrast edges.
    noise_sigma is the residual noise of x, ref_sigma the (per-pixel) noise of ref, both in 0..255
    units: ref is cored at ITS noise level, so re-injection never brings the removed grain back.
    All strengths 0 = no-op."""
    y = luminance(np.clip(x, 0, 1)).astype(np.float32)
    t = 2.5 * max(float(noise_sigma), 0.3) / 255.0          # noise floor for coring
    y1 = y.copy()
    if reinject > 0 and ref is not None:
        yr = luminance(np.clip(ref, 0, 1)).astype(np.float32)
        t_ref = t if ref_sigma is None else 2.5 * np.maximum(ref_sigma, float(noise_sigma)) / 255.0
        d_ref = core(yr - guided_filter(yr, yr, 2, 1e-3), t_ref)
        d_out = y - guided_filter(y, y, 2, 1e-3)
        y1 = y1 + reinject * (d_ref - d_out) * (np.abs(d_ref) > np.abs(d_out))  # only where detail was lost
    for r, amt in ((2, fine), (6, mid)):
        if amt > 0:
            # Gaussian base, not an edge-preserving one: an edge-aware base keeps the edges, leaving
            # nothing to sharpen (validation: blurred inputs changed by 0.07 grey levels on average)
            y1 = y1 + amt * core(y1 - cv2.GaussianBlur(y1, (0, 0), r / 2), t)
    k = np.ones((3, 3), np.uint8)
    lo, hi = cv2.erode(y, k), cv2.dilate(y, k)
    y1 = np.clip(y1, lo - 0.04, hi + 0.04)
    return np.clip(x * (np.maximum(y1, 0) / np.maximum(y, 1e-4))[..., None], 0, 1)


def limit_saturation(img, max_colourfulness):
    """Scale chroma down (in Lab, hue kept) only if the mid-tone colourfulness (chroma / lightness)
    exceeds max_colourfulness; normal and colourful scenes are left alone."""
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    m = (lab[..., 0] > 3) & (lab[..., 0] < 97)
    if m.sum() < 100:
        return img
    c = float(np.hypot(lab[..., 1], lab[..., 2])[m].mean() / (lab[..., 0][m].mean() + 1e-3))
    if c <= max_colourfulness:
        return img
    lab[..., 1:] *= max_colourfulness / c
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0, 1)
