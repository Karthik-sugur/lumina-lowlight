"""Evaluation metrics. Full-reference (need ground truth): PSNR, SSIM, LPIPS, CIEDE2000, EPI.
No-reference (any image): NIQE, residual noise, clipping, colour cast, brightness."""
import cv2
import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .analysis import estimate_noise
from .classical import luminance

_iqa = {}


def _pyiqa(name):
    if name not in _iqa:
        import pyiqa  # heavy import, only when LPIPS/NIQE are asked for
        _iqa[name] = pyiqa.create_metric(name, device="cpu")
    return _iqa[name]


def _t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float()


def psnr(x, gt):
    return float(peak_signal_noise_ratio(gt, x, data_range=1.0))


def ssim(x, gt):
    # Wang et al. settings: 11x11 Gaussian (sigma 1.5), population covariance, RGB averaged
    return float(structural_similarity(gt, x, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


@torch.no_grad()
def lpips(x, gt):
    return float(_pyiqa("lpips")(_t(x), _t(gt)))


@torch.no_grad()
def niqe(x):
    return float(_pyiqa("niqe")(_t(x)))


def delta_e(x, gt):
    """Mean CIEDE2000 colour difference (lower = less colour shift)."""
    return float(deltaE_ciede2000(rgb2lab(gt), rgb2lab(x)).mean())


def epi(x, gt):
    """Edge Preservation Index: correlation of the Laplacian high-pass of output vs reference.
    1 = edges exactly preserved; drops with both blurring and added noise."""
    lx = cv2.Laplacian(luminance(x).astype(np.float32), cv2.CV_32F)
    lg = cv2.Laplacian(luminance(gt).astype(np.float32), cv2.CV_32F)
    lx, lg = lx - lx.mean(), lg - lg.mean()
    return float((lx * lg).sum() / (np.sqrt((lx * lx).sum() * (lg * lg).sum()) + 1e-12))


def noise(x):
    """Residual luma noise std in the output, 0..255 units (Immerkaer estimator)."""
    return estimate_noise(luminance(x).astype(np.float32))


def overexposed(x, ref=None):
    """% of pixels clipped (>= 0.98 in any channel); with a reference, only newly clipped ones."""
    m = x.max(axis=2) >= 0.98
    if ref is not None:
        m &= ref.max(axis=2) < 0.98
    return float(m.mean() * 100)


def colour_cast(x):
    """Chroma of the average mid-tone colour in Lab (0 = neutral)."""
    y = luminance(x)
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2Lab)
    m = (y > 0.05) & (y < 0.95)
    if m.sum() < 100:
        return 0.0
    return float(np.hypot(lab[..., 1][m].mean(), lab[..., 2][m].mean()))


def full_reference(x, gt):
    return dict(psnr=psnr(x, gt), ssim=ssim(x, gt), lpips=lpips(x, gt), delta_e=delta_e(x, gt),
                epi=epi(x, gt), noise=noise(x), overexposed=overexposed(x, gt),
                brightness_err=abs(float(luminance(x).mean() - luminance(gt).mean())))


def no_reference(x, with_niqe=True):
    d = dict(noise=noise(x), overexposed=overexposed(x), colour_cast=colour_cast(x),
             brightness=float(luminance(x).mean()))
    if with_niqe:
        d["niqe"] = niqe(x)
    return d
