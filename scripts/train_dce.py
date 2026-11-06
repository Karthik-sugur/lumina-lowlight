"""Fine-tune Zero-DCE++ on LOL our485 with a fixed, seeded train/val split (435 / 50).
LOL eval15, LOL-v2 and ExDark are never touched here: they are the unseen test sets.

  python scripts/train_dce.py        -> weights/dce_finetuned.pth, results/train_log_dce.json

Loss = L1 + 0.3*(1 - SSIM) against the normal-light reference, plus Zero-DCE's illumination
smoothness (TV on the curve map) and colour-constancy terms as regularisers.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight import classical as C  # noqa: E402
from lowlight import dce  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "data"))
LOL = DATA / "lol" / "lol_dataset"


def read(p):
    return C.to_float(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))


def split():
    path = ROOT / "results" / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    files = sorted((p.name for p in (LOL / "our485" / "low").glob("*.png")), key=lambda s: int(s[:-4]))
    random.Random(0).shuffle(files)
    s = {"seed": 0, "val": sorted(files[:50]), "train": sorted(files[50:])}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(s, indent=1))
    return s


def load(names):
    return ([read(LOL / "our485" / "low" / n) for n in names],
            [read(LOL / "our485" / "high" / n) for n in names])


def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def batch(data, bs, crop):
    low, high = data[:2]
    xs, ys = [], []
    for i in random.choices(range(len(low)), k=bs):
        h, w = low[i].shape[:2]
        r, c = random.randint(0, h - crop), random.randint(0, w - crop)
        x = to_t(low[i][r:r + crop, c:c + crop]).permute(2, 0, 1)
        y = to_t(high[i][r:r + crop, c:c + crop]).permute(2, 0, 1)
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        if random.random() < 0.5:
            x, y = x.flip(-2), y.flip(-2)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def ssim(x, y, c1=0.01 ** 2, c2=0.03 ** 2):
    g = torch.exp(-(torch.arange(11) - 5.0) ** 2 / (2 * 1.5 ** 2))
    k = (g[:, None] * g[None]) / g.sum() ** 2
    k = k.expand(3, 1, 11, 11)
    mu = lambda t: F.conv2d(t, k, groups=3)  # noqa: E731
    mx, my = mu(x), mu(y)
    sxx, syy, sxy = mu(x * x) - mx ** 2, mu(y * y) - my ** 2, mu(x * y) - mx * my
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def loss_fn(out, gt, a):
    tv = (a[..., 1:, :] - a[..., :-1, :]).pow(2).mean() + (a[..., 1:] - a[..., :-1]).pow(2).mean()
    m = out.mean(dim=(2, 3))
    colour = ((m[:, 0] - m[:, 1]) ** 2 + (m[:, 1] - m[:, 2]) ** 2 + (m[:, 0] - m[:, 2]) ** 2).mean()
    return F.l1_loss(out, gt) + 0.3 * (1 - ssim(out, gt)) + 500 * tv + 0.5 * colour


@torch.no_grad()
def validate(model, data):
    psnrs = []
    for low, high in zip(*data):
        out, _ = model(to_t(low).permute(2, 0, 1)[None])
        mse = float(((out[0].clamp(0, 1).permute(1, 2, 0).numpy() - high) ** 2).mean())
        psnrs.append(10 * np.log10(1 / mse))
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(0)
    torch.manual_seed(0)

    s = split()
    train, val = load(s["train"]), load(s["val"])
    model = dce.ZeroDCEpp()
    model.load_state_dict(torch.load(ROOT / "weights" / "zero_dce_pp_official.pth", map_location="cpu"))
    iters = len(s["train"]) // args.bs
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * iters)

    log = {"args": vars(args), "epochs": []}
    model.eval()
    best = validate(model, val)
    log["val_psnr_before_finetune"] = best
    print(f"official weights val PSNR {best:.2f}", flush=True)
    out_path = ROOT / "weights" / "dce_finetuned.pth"
    torch.save(model.state_dict(), out_path)
    for ep in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        for _ in range(iters):
            x, y = batch(train, args.bs, args.crop)
            out, a = model(x)
            loss = loss_fn(out, y, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        model.eval()
        v = validate(model, val)
        if v > best:
            best = v
            torch.save(model.state_dict(), out_path)
        log["epochs"].append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_psnr": v})
        print(f"ep {ep + 1} loss {np.mean(losses):.4f} val {v:.2f} best {best:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        log["best_val_psnr"] = best
        (ROOT / "results" / "train_log_dce.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
