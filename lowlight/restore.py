"""Noise-aware restoration U-Net (ours): the deep-learning restoration stage of the hybrid.

Brightening a dark photo multiplies its sensor noise by the local gain, so noise is strongly
non-uniform after enhancement. The classical illumination estimate tells us exactly that gain,
so we hand the network a per-pixel noise-level map (sigma_in * gain, FFDNet-style conditioning).
The network then knows *where* to denoise hard (lifted shadows) and where to keep texture (areas
that were already lit). It predicts a residual on top of its input, so it corrects rather than
repaints the image. ~140k parameters, CPU-friendly.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "restore_net.pth"
NOISE_NORM = 50.0  # noise map = sigma_255 * gain / NOISE_NORM, clipped to [0, 1]


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.ReLU(inplace=True))


class RestoreNet(nn.Module):
    def __init__(self, c=16, cond=True):
        super().__init__()
        self.cond = cond
        self.e0 = _block(4 if cond else 3, c)
        self.d1, self.e1 = nn.Conv2d(c, 2 * c, 3, 2, 1), _block(2 * c, 2 * c)
        self.d2, self.e2 = nn.Conv2d(2 * c, 4 * c, 3, 2, 1), _block(4 * c, 4 * c)
        self.u1, self.m1 = nn.Conv2d(4 * c, 2 * c, 1), _block(2 * c, 2 * c)
        self.u0, self.m0 = nn.Conv2d(2 * c, c, 1), _block(c, c)
        self.out = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, x, nmap):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 4, (-w) % 4
        inp = torch.cat([x, nmap], 1) if self.cond else x
        inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")
        z0 = self.e0(inp)
        z1 = self.e1(F.relu(self.d1(z0)))
        z2 = self.e2(F.relu(self.d2(z1)))
        y1 = self.m1(F.interpolate(self.u1(z2), size=z1.shape[-2:], mode="bilinear", align_corners=False) + z1)
        y0 = self.m0(F.interpolate(self.u0(y1), size=z0.shape[-2:], mode="bilinear", align_corners=False) + z0)
        return x + self.out(y0)[..., :h, :w]


_model = {}


def load(path=WEIGHTS):
    if path not in _model:
        state = torch.load(path, map_location="cpu")
        m = RestoreNet(cond="e0.0.weight" in state and state["e0.0.weight"].shape[1] == 4)
        m.load_state_dict(state)
        _model[path] = m.eval()
    return _model[path]


@torch.no_grad()
def restore(img, nmap, path=WEIGHTS):
    """img: float32 RGB [0,1] (brightened); nmap: HxW noise-level map in [0,1]."""
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
    n = torch.from_numpy(np.ascontiguousarray(nmap))[None, None].float()
    return load(path)(x, n)[0].permute(1, 2, 0).clamp(0, 1).numpy()
