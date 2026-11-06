"""Export data the static web UI (web/) needs, so the website never hand-copies numbers.

  python scripts/export_web_data.py

- web/benchmarks.json : straight from results/metrics.json (the evaluation's own aggregates)
- web/samples/        : demo photos from examples/ (unseen test images)
- web/samples/hero_after.jpg : the hybrid pipeline's real output for the first sample (homepage teaser)
"""
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight import classical as C  # noqa: E402
from lowlight import pipeline as P  # noqa: E402

WEB = ROOT / "web"
SAMPLES = ["01_lolv2_indoor_1", "02_lolv2_indoor_2", "03_exdark_bus_stop_streetlight",
           "04_exdark_dark_road_lamps", "05_exdark_neon_sign_coloured_lights", "08_exdark_dogs_near_black"]


def main():
    m = json.loads((ROOT / "results" / "metrics.json").read_text())
    names = {v: k for k, v in P.METHODS.items()}
    sets = list(m["n_images"])
    out = {
        "source": "results/metrics.json (scripts/evaluate.py)",
        "machine": m["machine"],
        "n_images": m["n_images"],
        "methods": [{"key": k, "name": names[k]} for k in P.METHODS.values()],
        "main": {s: {k: m["main"][f"{s}|{k}"] for k in P.METHODS.values()} for s in sets},
        "robustness": {},
        "ablation": {},
        "latency_ms": m["latency_ms"],
        "models": m["models"],
    }
    for key, v in m["robustness"].items():
        cond, method = key.split("|")
        out["robustness"].setdefault(cond, {})[method] = {"psnr": v["psnr"], "ssim": v["ssim"]}
    for key, v in m["ablation"].items():
        s, variant = key.split("|")
        out["ablation"].setdefault(s, {})[variant] = {k: v[k] for k in ("psnr", "ssim", "lpips", "niqe")}
    (WEB / "benchmarks.json").write_text(json.dumps(out, indent=1))

    (WEB / "samples").mkdir(parents=True, exist_ok=True)
    for n in SAMPLES:
        shutil.copy(ROOT / "examples" / f"{n}.jpg", WEB / "samples" / f"{n}.jpg")
    img = C.to_float(cv2.cvtColor(cv2.imread(str(ROOT / "examples" / f"{SAMPLES[0]}.jpg")), cv2.COLOR_BGR2RGB))
    hero = P.run(img, "hybrid")["output"]
    cv2.imwrite(str(WEB / "samples" / "hero_after.jpg"), cv2.cvtColor(C.to_u8(hero), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote web/benchmarks.json, samples and hero_after.jpg")


if __name__ == "__main__":
    main()
