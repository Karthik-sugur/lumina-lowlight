"""Zero-DCE++ (Li, Guo, Loy - TPAMI 2021): deep curve estimation for low-light enhancement.

The network predicts a per-pixel curve parameter map A in [-1, 1]; the image is brightened by
iterating the quadratic curve LE(x) = x + A*(x^2 - x) eight times. Curves are pixel-wise and
monotone, so the model can only re-tone pixels: it cannot invent content (a trust property we
rely on in the app). ~10k parameters, runs on CPU in milliseconds.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


class DSConv(nn.Module):
    """Depthwise-separable 3x3 conv (official name: CSDN_Tem)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.depth_conv = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.point_conv = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    def __init__(self, nf=32):
        super().__init__()
        self.e_conv1 = DSConv(3, nf)
        self.e_conv2 = DSConv(nf, nf)
        self.e_conv3 = DSConv(nf, nf)
        self.e_conv4 = DSConv(nf, nf)
        self.e_conv5 = DSConv(nf * 2, nf)
        self.e_conv6 = DSConv(nf * 2, nf)
        self.e_conv7 = DSConv(nf * 2, 3)

    def curve_map(self, x, down=1):
        """Estimate A on a `down`-times smaller copy (Zero-DCE++ trick), upsample to full size."""
        xs = F.interpolate(x, scale_factor=1 / down, mode="bilinear", align_corners=False) if down > 1 else x
        x1 = F.relu(self.e_conv1(xs))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        a = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False) if down > 1 else a

    def forward(self, x, down=1):
        a = self.curve_map(x, down)
        out = x
        for _ in range(8):
            out = out + a * (out * out - out)
        return out, a


_cache = {}


def load(name):
    """name: 'official' (published Zero-DCE++ weights, zero-shot) or 'dce' (our LOL fine-tune)."""
    if name not in _cache:
        file = "zero_dce_pp_official.pth" if name == "official" else f"{name}_finetuned.pth"
        m = ZeroDCEpp()
        m.load_state_dict(torch.load(WEIGHTS / file, map_location="cpu"))
        _cache[name] = m.eval()
    return _cache[name]


def pick_down(h, w, max_side=640):
    """Curve maps are smooth, so estimate them at <= max_side px and upsample (speed on CPU)."""
    return max(1, int(np.ceil(max(h, w) / max_side)))


@torch.no_grad()
def curve_params(model, img):
    """Curve map A (H, W, 3) as numpy; apply with apply_curve_np at any exposure scale."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    return model.curve_map(x, pick_down(*img.shape[:2]))[0].permute(1, 2, 0).numpy()


def apply_curve_np(img, a, scale=1.0, n=8):
    """Exposure-scaled curve. |scale*A| is clamped to 1 so every curve stays monotone in [0,1]
    (no tone reversals, no values pushed past white)."""
    c = np.clip(scale * a, -1, 1)
    out = img.copy()
    for _ in range(n):
        out = out + c * (out * out - out)
    return np.clip(out, 0, 1)
