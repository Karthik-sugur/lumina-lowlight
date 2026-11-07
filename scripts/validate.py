"""Validation-only experiments behind the design decisions of the image-quality-aware strategy.
No test image is ever read here.

  python scripts/validate.py NAME '{"candidate": {"knobs": {"DETAIL_MID": 0.4}, "flags": {}}}' [--sets val,q70,q95,blur,fresh,fullsize] [--method hybrid]

"current" (the pipeline as it is) is always run too. Sets, all built from validation data:
  val       our485 validation split: 50 real low/normal-light pairs (results/split.json)
  q70       the same, 384 px wide, input saved as JPEG quality 70 (small compressed photos)
  q95       the same at full size, input saved as JPEG quality 95 (a typical phone JPEG)
  blur      validation references blurred (Gaussian sigma 1.5), then darkened x0.25 with shot and read
            noise, scored against the SHARP reference: does detail recovery recover detail?
  fresh     100 LOL-v2 synthetic *training* pairs used for nothing else (scripts/download_data.py)
  fullsize  val enhanced at 300 px and returned at 600 px by run_full (full-size detail transfer)

Decision rule, fixed before the runs it decides: a candidate replaces "current" only if its mean LPIPS
over the sets is lower, no set loses more than 0.10 dB PSNR, and residual noise on `val` rises < 10 %.
Among candidates that pass, the lowest mean LPIPS wins. Results: results/validation/NAME.json.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
DATA = Path(os.environ.get("DATA_DIR", "data"))
ALL_SETS = ["val", "q70", "q95", "blur", "fresh", "fullsize"]


def rd(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def jpeg(img, q, size=None):
    u8 = cv2.cvtColor((np.clip(img, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    if size:
        u8 = cv2.resize(u8, size, interpolation=cv2.INTER_AREA)
    dec = cv2.imdecode(cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def items(set_name):
    if set_name == "fresh":
        base = DATA / "lolv2_synthetic_trainval" / "Train"
        out = []
        for p in sorted((base / "Input").glob("*.png")):
            g = base / "GT" / p.name if (base / "GT" / p.name).exists() else base / "GT" / p.name[1:]
            if g.exists():
                out.append((str(p), str(g)))
        return out
    lol = DATA / "lol" / "lol_dataset" / "our485"
    names = json.loads((ROOT / "results" / "split.json").read_text())["val"]
    return [(str(lol / "low" / n), str(lol / "high" / n)) for n in names]


def load(set_name, low_p, gt_p, i):
    gt = rd(gt_p)
    if set_name == "q70":
        return jpeg(rd(low_p), 70, (384, 256)), cv2.resize(gt, (384, 256), interpolation=cv2.INTER_AREA)
    if set_name == "q95":
        return jpeg(rd(low_p), 95), gt
    if set_name == "blur":
        from evaluate import degrade
        return degrade(cv2.GaussianBlur(gt, (0, 0), 1.5), 0.25, False, i), gt
    return rd(low_p), gt


def job(task):
    set_name, low_p, gt_p, i, variants, method = task
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    from lowlight import analysis as A, classical as C, metrics as M, pipeline as P
    low, gt = load(set_name, low_p, gt_p, i)
    res = {}
    for name, spec in variants.items():
        saved = []
        for k, v in spec.get("knobs", {}).items():
            mod = next(m for m in (P, A, C) if hasattr(m, k))
            saved.append((mod, k, getattr(mod, k)))
            setattr(mod, k, v)
        try:
            if set_name == "fullsize":
                o = P.run_full(low, method, process_side=300, **spec.get("flags", {}))["output"]
            else:
                o = P.run(low, method, **spec.get("flags", {}))["output"]
        finally:
            for mod, k, v in saved:
                setattr(mod, k, v)
        res[name] = dict(psnr=M.psnr(o, gt), ssim=M.ssim(o, gt), lpips=M.lpips(o, gt), delta_e=M.delta_e(o, gt),
                         detail=A.sharpness_index(o) / max(A.sharpness_index(gt), 1e-6), noise=M.noise(o),
                         colour=A.colourfulness(o) / max(A.colourfulness(gt), 1e-6))
    return set_name, res


def decide(means, sets):
    cur = means["current"]
    lp = {v: float(np.mean([m[s]["lpips"] for s in sets])) for v, m in means.items()}
    verdict = {}
    for v, m in means.items():
        if v == "current":
            continue
        worst = min(m[s]["psnr"] - cur[s]["psnr"] for s in sets)
        noise_ok = "val" not in sets or m["val"]["noise"] <= 1.10 * cur["val"]["noise"]
        verdict[v] = dict(mean_lpips=lp[v], worst_psnr_change=worst,
                          passes=bool(lp[v] < lp["current"] and worst >= -0.10 and noise_ok))
    passing = [v for v in verdict if verdict[v]["passes"]]
    return {"current_mean_lpips": lp["current"], "candidates": verdict,
            "winner": min(passing, key=lambda v: lp[v]) if passing else "current"}


def main():
    name, variants = sys.argv[1], {"current": {}, **json.loads(sys.argv[2])}
    args = dict(a.split("=", 1) for a in sys.argv[3:] if a.startswith("--") and "=" in a)
    sets = args.get("--sets", ",".join(ALL_SETS)).split(",")
    method = args.get("--method", "hybrid")
    tasks = [(s, lo, gt, i, variants, method) for s in sets for i, (lo, gt) in enumerate(items(s))]
    with ProcessPoolExecutor(6) as ex:
        rows = list(ex.map(job, tasks, chunksize=2))
    means = {v: {s: {k: float(np.mean([r[v][k] for st, r in rows if st == s])) for k in rows[0][1]["current"]}
                 for s in sets} for v in variants}
    out = {"sets": {s: sum(st == s for st, _ in rows) for s in sets}, "method": method, "variants": variants,
           "means": means, "decision": decide(means, sets)}
    d = ROOT / "results" / "validation"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(out, indent=1))
    for v in variants:
        print(f"{v:22s} " + " | ".join(f"{s}: {means[v][s]['psnr']:.2f} dB, lpips {means[v][s]['lpips']:.4f}, "
                                       f"detail {means[v][s]['detail']:.2f}, noise {means[v][s]['noise']:.2f}" for s in sets))
    print("decision:", json.dumps(out["decision"], indent=1))


if __name__ == "__main__":
    main()
