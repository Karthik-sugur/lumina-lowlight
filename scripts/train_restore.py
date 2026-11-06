"""Train the noise-aware restoration U-Net (DL restoration stage of the hybrid).

Inputs are produced by the *exact* hybrid front-end (lowlight.pipeline.prepare + CLAHE/colour),
so training and inference see identical data. Same seeded LOL our485 split as train_dce.py.

Training pairs = the real LOL low/normal pairs PLUS one synthetic degradation of every LOL
*training* reference (exposure x0.02-x0.5 in linear light, Poisson shot + read noise; same physical
model as evaluate.degrade). LOL alone never shows the network the extreme noise of very dark
scenes, so without this it stops cleaning beyond its training range. Validation uses the 50 held-out
pairs, real and synthetic, and the checkpoint is chosen on their mean.

  python scripts/train_restore.py              # with classical noise-level map  -> restore_net.pth
  python scripts/train_restore.py --no-map     # ablation: no conditioning       -> restore_net_nomap.pth
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import degrade  # noqa: E402
from train_dce import LOL, ROOT, read, split, ssim  # noqa: E402

from lowlight import pipeline as P  # noqa: E402
from lowlight import restore  # noqa: E402

CACHE = Path(os.environ.get("DATA_DIR", "data")) / "cache" / "restore_inputs_v3.npz"
SYN = "#syn"


def synthetic_low(n):
    """One reproducible synthetic low-light version of reference n (exposure log-uniform 0.02-0.5)."""
    rng = np.random.default_rng(int(n[:-4]))
    k = float(np.exp(rng.uniform(np.log(0.02), np.log(0.5))))
    return degrade(read(LOL / "our485" / "high" / n), k, False, int(n[:-4]) + 10_000)


def build_cache(names):
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    f = dict(P.FULL)
    out = {}
    for i, n in enumerate([m for n in names for m in (n, n + SYN)]):
        low = synthetic_low(n[:-len(SYN)]) if n.endswith(SYN) else read(LOL / "our485" / "low" / n)
        x, local, d, p, viz, tm = P.prepare(low, f)
        x = P.contrast_colour(x, p, f, tm)
        nmap = np.clip(local / restore.NOISE_NORM, 0, 1)
        out[n] = np.dstack([x, nmap]).astype(np.float16)
        if i % 50 == 0:
            print("prepared", i, flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def tensors(arr, gt):
    """Kept compact in RAM (float16 inputs, uint8 targets); converted to float32 per crop."""
    a = torch.from_numpy(arr).permute(2, 0, 1)
    return a[:3], a[3:], torch.from_numpy((gt * 255 + 0.5).astype(np.uint8)).permute(2, 0, 1)


def f32(x, n, y):
    return x.float(), n.float(), y.float() / 255


def batch(items, bs, crop):
    xs, ns, ys = [], [], []
    for x, n, y in random.choices(items, k=bs):
        h, w = x.shape[-2:]
        r, c = random.randint(0, h - crop), random.randint(0, w - crop)
        x, n, y = f32(*(t[:, r:r + crop, c:c + crop] for t in (x, n, y)))
        if random.random() < 0.5:
            x, n, y = x.flip(-1), n.flip(-1), y.flip(-1)
        if random.random() < 0.5:
            x, n, y = x.flip(-2), n.flip(-2), y.flip(-2)
        xs.append(x)
        ns.append(n)
        ys.append(y)
    return torch.stack(xs), torch.stack(ns), torch.stack(ys)


@torch.no_grad()
def validate(model, items):
    out = []
    for t in items:
        x, n, y = f32(*t)
        out.append(10 * np.log10(1 / F.mse_loss(model(x[None], n[None]).clamp(0, 1), y[None]).item()))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(0)
    torch.manual_seed(0)

    s = split()
    cache = build_cache(s["train"] + s["val"])
    gts = {n: read(LOL / "our485" / "high" / n) for n in s["train"] + s["val"]}
    train = [tensors(cache[m], gts[n]) for n in s["train"] for m in (n, n + SYN)]
    val_real = [tensors(cache[n], gts[n]) for n in s["val"]]
    val_syn = [tensors(cache[n + SYN], gts[n]) for n in s["val"]]
    del cache

    model = restore.RestoreNet(cond=not args.no_map)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.iters, pct_start=0.05)
    name = "restore_net_nomap" if args.no_map else "restore_net"
    out_path = ROOT / "weights" / f"{name}.pth"
    log = {"args": vars(args), "params": sum(p.numel() for p in model.parameters()),
           "val_psnr_input": float(np.mean([10 * np.log10(1 / F.mse_loss(f32(*t)[0], f32(*t)[2]).item())
                                            for t in val_real])),
           "evals": []}
    print(name, "params", log["params"], "val PSNR of input", round(log["val_psnr_input"], 2), flush=True)
    best, t0, losses = 0.0, time.time(), []
    for it in range(1, args.iters + 1):
        model.train()
        x, n, y = batch(train, args.bs, args.crop)
        out = model(x, n)
        loss = F.l1_loss(out, y) + 0.2 * (1 - ssim(out.clamp(0, 1), y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if it % 250 == 0 or it == args.iters:
            model.eval()
            vr, vs = validate(model, val_real), validate(model, val_syn)
            v = (vr + vs) / 2
            if v > best:
                best = v
                torch.save(model.state_dict(), out_path)
            log["evals"].append({"iter": it, "train_loss": float(np.mean(losses)), "val_psnr": vr,
                                 "val_psnr_synthetic": vs})
            log["best_val_psnr"] = best
            losses = []
            print(f"{name} it {it} loss {log['evals'][-1]['train_loss']:.4f} val real {vr:.2f} syn {vs:.2f} best {best:.2f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            (ROOT / "results" / f"train_log_{name}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
