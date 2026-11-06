"""Full evaluation: every method on every unseen test set, ablation, illumination-robustness sweep,
latency benchmark, failure-case mining, figures and markdown tables.

  python scripts/evaluate.py            (~1 h on an 8-core laptop CPU; parallel over images)

Outputs: results/metrics.json, results/per_image.csv, results/tables.md, results/figures/*.png
Enhanced images are written to $DATA_DIR/outputs (not committed).
"""
import csv
import hashlib
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
from lowlight import analysis as A  # noqa: E402
from lowlight import classical as C  # noqa: E402
from lowlight import metrics as M  # noqa: E402
from lowlight import pipeline as P  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "data"))
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
OUTS = DATA / "outputs"
# LOL-v2 real re-packages most of LOL-v1: 86 of its 100 test inputs and 91 of its references are
# pixel-identical to LOL-v1 our485 images (our training/validation data). Only pairs with neither
# image in our485 are evaluated, so every test image is genuinely unseen.
V2R = "LOL-v2 real test (clean subset)"
PAIRED = ["LOL-v1 eval15", V2R, "LOL-v2 synthetic test"]
ABL_SETS = ("LOL-v1 eval15", "LOL-v2 synthetic test")  # enough pairs for a stable ablation
BEFORE = "previous pipeline (before quality-aware update)"  # also run on ExDark (before/after)
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


def pixel_hash(p):
    return hashlib.md5(cv2.imread(str(p), cv2.IMREAD_UNCHANGED).tobytes()).hexdigest()


def seen_hashes():
    """Pixel hashes of every LOL-v1 our485 image (training + validation split)."""
    d = DATA / "lol" / "lol_dataset" / "our485"
    return {pixel_hash(p) for p in list((d / "low").glob("*.png")) + list((d / "high").glob("*.png"))}


def test_sets():
    lol, v2r, v2s = DATA / "lol" / "lol_dataset" / "eval15", DATA / "lolv2_real" / "Test", DATA / "lolv2_synthetic" / "Test"
    ex = sorted(str(p) for p in (DATA / "exdark" / "images" / "ExDark").glob("*/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    seen = seen_hashes()
    return {
        "LOL-v1 eval15": [(p, lol / "high" / p.name) for p in sorted((lol / "low").glob("*.png"))],
        V2R: [(p, v2r / "GT" / p.name) for p in sorted((v2r / "Input").glob("*.png"))
              if pixel_hash(p) not in seen and pixel_hash(v2r / "GT" / p.name) not in seen],
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
    if gt is not None:  # 1 = as much fine detail as the reference; < 1 softer, > 1 crisper (or noisier)
        m["detail_ratio"] = A.sharpness_index(out) / max(A.sharpness_index(gt), 1e-6)
    if kind == "main" and method in ("classical", "dl", "hybrid"):
        import api
        m.update({f"q_{k}": float(v) for k, v in api.quality(out).items()})
    if kind != "robust":
        m.update(M.no_reference(out, with_niqe=True) if gt is None else {"niqe": M.niqe(out)})
    if gt is None:  # what the quality-aware update targets on real-world JPEGs
        m.update(colourfulness=A.colourfulness(out), blockiness=A.blockiness(out), sharpness=A.sharpness_index(out))
    return dict(kind=kind, set=set_name, image=Path(low_p or gt_p).name, method=method,
                variant=extra if kind != "robust" else extra[-1], **m)


def mean_rows(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    metrics = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})  # union: sets differ
    return {k: {m: float(np.mean([r[m] for r in v if m in r])) for m in metrics if any(m in r for r in v)}
            for k, v in out.items()}


def auto_mode(rows):
    """Auto mode = the API's per-image recommendation from no-reference checks only. Compared with
    always using one method and with the oracle (best PSNR per image, needs the ground truth)."""
    import api
    out = {}
    for s in {r["set"] for r in rows}:
        imgs = {}
        for r in rows:
            if r["set"] == s and r["method"] in ("classical", "dl", "hybrid"):
                imgs.setdefault(r["image"], {})[r["method"]] = r
        picks, chosen, oracle = {}, [], []
        for by_m in imgs.values():
            q = {m: {k[2:]: v for k, v in r.items() if k.startswith("q_")} for m, r in by_m.items()}
            m = api.recommend(q, A.TARGET_BRIGHTNESS)["method"]
            picks[m] = picks.get(m, 0) + 1
            chosen.append(by_m[m])
            if "psnr" in by_m[m]:
                oracle.append(max(by_m.values(), key=lambda r: r["psnr"]))
        metrics = [k for k in ("psnr", "ssim", "lpips", "niqe", "noise", "delta_e") if k in chosen[0]]
        out[s] = {"picks": picks, "auto": {k: float(np.mean([r[k] for r in chosen])) for k in metrics}}
        if oracle:
            out[s]["oracle_best_psnr"] = {k: float(np.mean([r[k] for r in oracle])) for k in metrics}
    return out


def latency():
    """Sequential, single process, all threads: what one user of the app experiences."""
    import torch
    torch.set_num_threads(os.cpu_count())
    cv2.setNumThreads(os.cpu_count())
    imgs = [read(p) for p, _ in test_sets()["LOL-v1 eval15"][:6]]
    big = cv2.resize(imgs[0], (1280, 960), interpolation=cv2.INTER_CUBIC)
    huge = cv2.resize(imgs[0], (2048, 1536), interpolation=cv2.INTER_CUBIC)
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
        mid = (time.perf_counter() - t) / 3 * 1000
        t = time.perf_counter()
        for _ in range(2):
            P.run_full(huge, key)  # what the app does for a 2048 px upload: enhance at 1280, return 2048
        res[name] = {"ms_600x400": small, "ms_1280x960": mid, "ms_2048x1536_full_size": (time.perf_counter() - t) / 2 * 1000}
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
        for s in ABL_SETS + ((EXDARK,) if name in ("Hybrid (full)", BEFORE) else ()):
            for low, gt in sets[s]:
                tasks.append(("ablation", s, str(low), str(gt) if gt else None, "hybrid", flags, name))
    robust_gts = random.Random(1).sample([gt for _, gt in sets["LOL-v2 synthetic test"]], 2 if quick else 30)
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
        "auto": auto_mode(by("main")),
        "latency_ms": latency(),
        "models": model_sizes(),
        "machine": f"{platform.processor() or platform.machine()} ({os.cpu_count()} threads), CPU only",
        "n_images": {s: len(v) for s, v in sets.items()},
    }
    (RES / "metrics.json").write_text(json.dumps(results, indent=1))
    print("done in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
