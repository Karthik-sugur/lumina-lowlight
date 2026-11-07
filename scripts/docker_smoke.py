"""End-to-end checks of a running Lumina engine over HTTP, the way the web frontend calls it.

  python scripts/docker_smoke.py --base http://127.0.0.1:7860 [--prod https://darshuuuu-lumina-lowlight.hf.space] [--out DIR]

Checks: health, API contract (response keys), output sizes (full-size output up to 2048 px, larger inputs
capped with a note), PNG / greyscale / alpha inputs, rejected inputs (empty, not base64, not an image,
unsupported format, too small, over 15 MB), concurrent requests, and, with --prod, the same photos run on
the production engine compared pixel by pixel. Writes report.json to --out; exits 1 if a check fails.
"""
import argparse
import base64
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = {"input", "diagnosis", "params", "recoverability", "diagnostics", "strategy", "noticed", "actions",
            "grain_uncovered", "results", "recommendation", "intermediates", "timings_ms", "total_ms", "model"}
INPUT_KEYS = {"width", "height", "processed_width", "processed_height", "output_width", "output_height", "note", "quality"}
RESULT_KEYS = {"label", "image", "ms", "quality"}


def data_url(img, ext=".jpg", quality=95):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext == ".jpg" else []
    buf = cv2.imencode(ext, img, params)[1].tobytes()
    return f"data:image/{'jpeg' if ext == '.jpg' else ext[1:]};base64," + base64.b64encode(buf).decode()


def call(base, payload, timeout=300):
    """POST /gradio_api/call/restore, then read the server-sent events. Returns (ok, result_or_error, seconds)."""
    t = time.perf_counter()
    req = urllib.request.Request(f"{base}/gradio_api/call/restore", method="POST",
                                 data=json.dumps({"data": [payload, "Auto (balanced)", 1.0]}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        event_id = json.load(r)["event_id"]
    with urllib.request.urlopen(f"{base}/gradio_api/call/restore/{event_id}", timeout=timeout) as r:
        text = r.read().decode()
    for block in text.split("\n\n"):
        if block.startswith("event: complete"):
            return True, json.loads(block.split("data: ", 1)[1])[0], time.perf_counter() - t
        if block.startswith("event: error"):
            return False, block.split("data: ", 1)[-1].strip(), time.perf_counter() - t
    return False, f"no complete/error event: {text[:200]}", time.perf_counter() - t


def decode(url):
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def wait_ready(base, seconds):
    for _ in range(seconds // 2):
        try:
            with urllib.request.urlopen(f"{base}/gradio_api/info", timeout=10) as r:
                if r.status == 200 and "restore" in r.read().decode():
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--prod")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    report, failures = {"checks": {}}, []

    def check(name, cond, detail):
        report["checks"][name] = {"pass": bool(cond), "detail": detail}
        print(("PASS " if cond else "FAIL ") + name + ": " + json.dumps(detail, default=str)[:300], flush=True)
        if not cond:
            failures.append(name)

    check("health: /gradio_api/info lists the restore endpoint", wait_ready(a.base, 180), a.base)
    sample = cv2.imread(str(ROOT / "examples" / "01_lolv2_indoor_1.jpg"))
    exdark = cv2.imread(str(ROOT / "examples" / "03_exdark_bus_stop_streetlight.jpg"))
    big = cv2.resize(cv2.imread(str(ROOT / "examples" / "02_lolv2_stadium_seats.jpg")), (2048, 1365), interpolation=cv2.INTER_CUBIC)
    huge = cv2.resize(sample, (3000, 2000), interpolation=cv2.INTER_CUBIC)
    inputs = {"sample 600x400 JPEG": data_url(sample), "ExDark 384px JPEG": data_url(exdark), "2048x1365 JPEG": data_url(big)}

    # 1. contract and output size
    responses = {}
    for name, url in inputs.items():
        ok, r, s = call(a.base, url)
        responses[name] = r if ok else None
        if not ok:
            check(f"restore: {name}", False, r)
            continue
        h, w = decode(r["results"]["hybrid"]["image"]).shape[:2]
        exp = (2048, 1365) if name.startswith("2048") else (sample.shape[1], sample.shape[0]) if name.startswith("sample") else (exdark.shape[1], exdark.shape[0])
        check(f"restore: {name}", ok and set(r) == CONTRACT and set(r["results"]) == {"classical", "dl", "hybrid"}
              and all(set(v) == RESULT_KEYS for v in r["results"].values()) and set(r["input"]) == INPUT_KEYS
              and (w, h) == exp and all(decode(v["image"]).shape[:2] == (h, w) for v in r["results"].values()),
              {"seconds": round(s, 2), "engine_ms": r["total_ms"], "output": f"{w}x{h}", "expected": f"{exp[0]}x{exp[1]}",
               "processed": f"{r['input']['processed_width']}x{r['input']['processed_height']}", "note": r["input"]["note"][:90]})
    ok, r, s = call(a.base, data_url(huge))
    check("over 2048 px: capped to 2048 with a note", ok and r["input"]["output_width"] == 2048 and "3000x2000" in r["input"]["note"]
          and decode(r["results"]["hybrid"]["image"]).shape[:2] == (1365, 2048), r["input"] if ok else r)
    for name, img, ext in (("PNG", sample, ".png"), ("greyscale PNG", cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), ".png"),
                           ("RGBA PNG", cv2.cvtColor(sample, cv2.COLOR_BGR2BGRA), ".png"), ("WebP", sample, ".webp")):
        ok, r, s = call(a.base, data_url(img, ext))
        check(f"accepts {name}", ok and decode(r["results"]["hybrid"]["image"]).shape == (400, 600, 3), {"seconds": round(s, 2)} if ok else r)

    # 2. rejected inputs: each must come back as an error event, never a crash or a result
    noise = np.random.default_rng(0).integers(0, 256, (2600, 2600, 3), dtype=np.uint8)
    bad = {"empty": "", "not base64": "%%%not-base64%%%", "text, not an image": "data:image/png;base64," + base64.b64encode(b"hello").decode(),
           "PDF (unsupported format)": "data:application/pdf;base64," + base64.b64encode(b"%PDF-1.4\n1 0 obj<<>>endobj\n%%EOF").decode(),
           "12x12 (too small)": data_url(sample[:12, :12], ".png"), "over 15 MB": data_url(noise, ".png")}
    for name, payload in bad.items():
        ok, r, s = call(a.base, payload)
        check(f"rejects {name}", not ok, {"error": str(r)[:160], "payload_mb": round(len(payload) / 2 ** 20, 1), "seconds": round(s, 2)})
    gif = cv2.imencode(".gif", sample)[1].tobytes() if cv2.haveImageWriter(".gif") else None
    if gif:  # informational: the web UI only offers JPG/PNG/WebP; the engine accepts whatever OpenCV decodes
        ok, r, s = call(a.base, "data:image/gif;base64," + base64.b64encode(gif).decode())
        report["checks"]["info: GIF upload"] = {"pass": True, "detail": "accepted" if ok else f"rejected: {str(r)[:120]}"}
    ok, r, s = call(a.base, inputs["sample 600x400 JPEG"])
    check("still healthy after the rejected inputs", ok, {"seconds": round(s, 2)})

    # 3. concurrency: 4 simultaneous requests (the app queues >2 at a time)
    t = time.perf_counter()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda _: call(a.base, inputs["sample 600x400 JPEG"]), range(4)))
    check("4 concurrent requests all succeed", all(ok for ok, _, _ in res),
          {"wall_s": round(time.perf_counter() - t, 2), "each_s": [round(s, 2) for _, _, s in res]})

    # 4. container vs production: same inputs, pixel comparison of the returned images
    if a.prod:
        comp = {}
        if not wait_ready(a.prod, 240):
            report["production"] = "unreachable (sleeping Space?) - comparison skipped"
        for name, url in inputs.items():
            if responses.get(name) is None or report.get("production"):
                continue
            ok, r, s = call(a.prod, url)
            if not ok:
                comp[name] = f"production error: {r}"
                continue
            row = {}
            for m in ("classical", "dl", "hybrid"):
                x = decode(responses[name]["results"][m]["image"]).astype(np.float64)
                y = decode(r["results"][m]["image"]).astype(np.float64)
                mse = float(((x - y) ** 2).mean())
                row[m] = {"mean_abs_diff": round(float(np.abs(x - y).mean()), 3), "max_abs_diff": int(np.abs(x - y).max()),
                          "psnr_vs_production": "identical" if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)}
            row["same_keys"] = set(r) == set(responses[name])
            row["same_recommendation"] = r["recommendation"]["method"] == responses[name]["recommendation"]["method"]
            row["diagnosis_max_rel_diff"] = round(max(abs(responses[name]["diagnosis"][k] - v) / (abs(v) + 1e-6)
                                                      for k, v in r["diagnosis"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)), 4)
            row["engine_ms"] = {"container": responses[name]["total_ms"], "production": r["total_ms"]}
            comp[name] = row
        report["production_comparison"] = comp
        print(json.dumps(comp, indent=1), flush=True)
        worst = [v[m]["mean_abs_diff"] for v in comp.values() if isinstance(v, dict) for m in ("classical", "dl", "hybrid")]
        if worst:
            check("container output matches production (mean abs diff <= 1 grey level)", max(worst) <= 1.0, {"worst_mean_abs_diff": max(worst)})

    report["failures"] = failures
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"\n{len(report['checks']) - len(failures)} passed, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
