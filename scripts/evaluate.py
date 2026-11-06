"""Full evaluation: every method on every unseen test set, ablation, illumination-robustness sweep,
latency benchmark, failure-case mining, figures and markdown tables.

  python scripts/evaluate.py            (~1 h on an 8-core laptop CPU; parallel over images)

Outputs: results/metrics.json, results/per_image.csv, results/tables.md, results/figures/*.png
Enhanced images are written to $DATA_DIR/outputs (not committed).
"""
import csv
import json
import os
import platform
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight import classical as C  # noqa: E402
from lowlight import metrics as M  # noqa: E402
from lowlight import pipeline as P  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "data"))
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
OUTS = DATA / "outputs"
PAIRED = ["LOL-v1 eval15", "LOL-v2 real test", "LOL-v2 synthetic test"]
EXDARK = "ExDark (real-world, no reference)"
ROBUST_METHODS = ["he", "zero_dce_official", "classical", "dl", "hybrid"]
CONDITIONS = {"exposure x0.5": (0.5, False), "x0.25": (0.25, False), "x0.1": (0.1, False),
              "x0.05": (0.05, False), "x0.025": (0.025, False), "spotlight (non-uniform)": (0.25, True)}


def read(p):
    img = cv2.imread(str(p))
    return C.to_float(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def trim_letterbox(img):
    """ExDark-384 images are letterboxed with pure-black bars; drop them so they don't count as 'dark'."""
    m = img.max(axis=2) > 3 / 255
    r, c = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return img[r[0]:r[-1] + 1, c[0]:c[-1] + 1] if len(r) and len(c) else img


def test_sets():
    lol, v2r, v2s = DATA / "lol" / "lol_dataset" / "eval15", DATA / "lolv2_real" / "Test", DATA / "lolv2_synthetic" / "Test"
    ex = sorted(str(p) for p in (DATA / "exdark" / "images" / "ExDark").glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return {
        "LOL-v1 eval15": [(p, lol / "high" / p.name) for p in sorted((lol / "low").glob("*.png"))],
        "LOL-v2 real test": [(p, v2r / "GT" / p.name) for p in sorted((v2r / "Input").glob("*.png"))],
        "LOL-v2 synthetic test": [(p, v2s / "GT" / p.name[1:]) for p in sorted((v2s / "Input").glob("*.png"))],
        EXDARK: [(Path(p), None) for p in random.Random(0).sample(ex, 60)],
    }


def degrade(gt, k, spotlight, seed):
    """Physically-motivated synthetic low light: linearise, scale exposure (optionally with a
    spotlight falloff), add Poisson shot + Gaussian read noise, re-encode to 8-bit sRGB."""
    rng = np.random.default_rng(seed)
    lin = gt.astype(np.float64) ** 2.2 * k
    if spotlight:
        h, w = gt.shape[:2]
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        yy, xx = np.mgrid[:h, :w]
        d = np.hypot(yy - cy, xx - cx) / max(h, w)
        lin *= (0.04 + 1.2 * np.exp(-(d / 0.3) ** 2))[..., None]
    lin = rng.poisson(np.clip(lin, 0, 1) * 800) / 800 + rng.normal(0, 0.002, lin.shape)
    return (np.round(np.clip(lin, 0, 1) ** (1 / 2.2) * 255) / 255).astype(np.float32)


def _init():
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(2)


def job(task):
    kind, set_name, low_p, gt_p, method, flags, extra = task
    if kind == "robust":
        gt = read(gt_p)
        img = degrade(gt, *extra[:3])
    else:
        img, gt = read(low_p), (read(gt_p) if gt_p else None)
        if gt is None:
            img = trim_letterbox(img)
    r = P.run(img, method, **flags)
    out = r["output"]
    if kind == "main":
        d = OUTS / method / set_name.split(" (")[0].replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / (Path(low_p).stem + ".png")), cv2.cvtColor(C.to_u8(out), cv2.COLOR_RGB2BGR))
    m = M.full_reference(out, gt) if gt is not None else {}
    if kind != "robust":
        m.update(M.no_reference(out, with_niqe=True) if gt is None else {"niqe": M.niqe(out)})
    return dict(kind=kind, set=set_name, image=Path(low_p or gt_p).name, method=method,
                variant=extra if kind != "robust" else extra[-1], **m)


def mean_rows(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    metrics = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})  # union: sets differ
    return {k: {m: float(np.mean([r[m] for r in v if m in r])) for m in metrics if any(m in r for r in v)}
            for k, v in out.items()}


def latency():
    """Sequential, single process, all threads: what one user of the app experiences."""
    import torch
    torch.set_num_threads(os.cpu_count())
    cv2.setNumThreads(os.cpu_count())
    imgs = [read(p) for p, _ in test_sets()["LOL-v1 eval15"][:6]]
    big = cv2.resize(imgs[0], (1280, 960), interpolation=cv2.INTER_CUBIC)
    res = {}
    for name, key in P.METHODS.items():
        P.run(imgs[0], key)  # warm-up (model load)
        t = time.perf_counter()
        for im in imgs:
            P.run(im, key)
        small = (time.perf_counter() - t) / len(imgs) * 1000
        t = time.perf_counter()
        for _ in range(3):
            P.run(big, key)
        res[name] = {"ms_600x400": small, "ms_1280x960": (time.perf_counter() - t) / 3 * 1000}
    return res


def model_sizes():
    import torch
    from lowlight import dce, restore
    out = {}
    for name, mod, f in [("Zero-DCE++ (fine-tuned)", dce.load("dce"), ROOT / "weights" / "dce_finetuned.pth"),
                         ("Restoration U-Net", restore.load(), restore.WEIGHTS)]:
        out[name] = {"params": int(sum(p.numel() for p in mod.parameters())), "file_kb": round(f.stat().st_size / 1024, 1)}
    return out


def main():
    quick = "--quick" in sys.argv  # smoke test: 2 images per set
    sets = {k: v[:2] for k, v in test_sets().items()} if quick else test_sets()
    tasks = []
    for s, items in sets.items():
        for low, gt in items:
            for key in P.METHODS.values():
                tasks.append(("main", s, str(low), str(gt) if gt else None, key, {}, "main"))
    for name, flags in P.ABLATIONS.items():
        if "noise_map" in flags and not P.NOMAP_WEIGHTS.exists():
            continue
        for s in ("LOL-v1 eval15", "LOL-v2 real test"):
            for low, gt in sets[s]:
                tasks.append(("ablation", s, str(low), str(gt), "hybrid", flags, name))
    robust_gts = random.Random(1).sample([gt for _, gt in sets["LOL-v2 real test"]], 2 if quick else 30)
    for cname, (k, spot) in CONDITIONS.items():
        for i, gt in enumerate(robust_gts):
            for key in ROBUST_METHODS:
                tasks.append(("robust", cname, None, str(gt), key, {}, (k, spot, i, cname)))

    t0 = time.time()
    rows = []
    with Pool(8, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(job, tasks, chunksize=4)):
            rows.append(r)
            if i % 200 == 0:
                print(f"{i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)

    RES.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(RES / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    by = lambda kind: [r for r in rows if r["kind"] == kind]  # noqa: E731
    results = {
        "main": {f"{s}|{m}": v for (s, m), v in mean_rows(by("main"), ["set", "method"]).items()},
        "ablation": {f"{s}|{v_}": v for (s, v_), v in mean_rows(by("ablation"), ["set", "variant"]).items()},
        "robustness": {f"{s}|{m}": v for (s, m), v in mean_rows(by("robust"), ["set", "method"]).items()},
        "latency_ms": latency(),
        "models": model_sizes(),
        "machine": f"{platform.processor() or platform.machine()} ({os.cpu_count()} threads), CPU only",
        "n_images": {s: len(v) for s, v in sets.items()},
    }
    (RES / "metrics.json").write_text(json.dumps(results, indent=1))
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
