"""Turn results/metrics.json + per_image.csv into markdown tables, figures and failure-case panels.

  python scripts/report.py        (run after scripts/evaluate.py; takes seconds)
"""
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lowlight import analysis as A  # noqa: E402
from lowlight import classical as C  # noqa: E402
from lowlight import pipeline as P  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import trim_letterbox  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "data"))
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
NAMES = {v: k for k, v in P.METHODS.items()}
# validated categorical palette (dataviz reference instance), fixed per method = colour follows entity
COLOURS = dict(zip(P.METHODS.values(), ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
METRICS = [  # (key, label, higher_is_better, fmt) - order follows the problem statement
    ("psnr", "PSNR (dB) ↑", True, "{:.2f}"), ("ssim", "SSIM ↑", True, "{:.3f}"), ("lpips", "LPIPS ↓", False, "{:.3f}"),
    ("niqe", "NIQE ↓", False, "{:.2f}"), ("noise", "Residual noise σ ↓", False, "{:.2f}"),
    ("epi", "Edge preservation (EPI) ↑", True, "{:.3f}"), ("delta_e", "Colour shift ΔE2000 ↓", False, "{:.2f}"),
    ("overexposed", "New overexposure % ↓", False, "{:.2f}"), ("brightness_err", "Brightness error ↓", False, "{:.3f}"),
]


def table(rows, cols, bold_best=True):
    """rows: list of (label, {metric: value}); cols: subset of METRICS."""
    head = "| Method | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    best = {}
    for k, _, hib, _ in cols:
        vals = [r[1].get(k) for r in rows if r[1].get(k) is not None]
        best[k] = (max if hib else min)(vals) if vals else None
    lines = []
    for label, m in rows:
        cells = []
        for k, _, _, fmt in cols:
            v = m.get(k)
            s = "-" if v is None else fmt.format(v)
            cells.append(f"**{s}**" if bold_best and v is not None and np.isclose(v, best[k]) else s)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def bars(ax, groups, series, values, ylabel, fmt="{:.1f}", label_series=()):
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        xs = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w
        ys = [values[g][s] for g in groups]
        ax.bar(xs, ys, w * 0.9, color=COLOURS[s], label=NAMES[s], edgecolor=SURF, linewidth=1)
        if s in label_series:
            for x, y in zip(xs, ys):
                ax.annotate(fmt.format(y), (x, y), ha="center", va="bottom", fontsize=8, color=INK, xytext=(0, 2),
                            textcoords="offset points")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def cover(img, w, h):
    """Scale to cover a w x h cell and centre-crop, so every grid column lines up."""
    s = max(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(w, round(img.shape[1] * s)), max(h, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
    return r[y:y + h, x:x + w]


def read_input(s, name, src):
    if s in src:
        return cv2.imread(str(src[s] / name))
    p = next(DATA.joinpath("exdark", "images", "ExDark").glob(f"*/{name}"))
    return C.to_u8(trim_letterbox(C.to_float(cv2.imread(str(p)))))


def failures(rows, src, exdark):
    """Worst hybrid results: lowest PSNR on LOL-v2 real (paired) and highest NIQE on ExDark."""
    d = RES / "failures"
    d.mkdir(exist_ok=True)
    hyb = [r for r in rows if r["kind"] == "main" and r["method"] == "hybrid"]
    worst = sorted([r for r in hyb if r["set"] == "LOL-v2 real test"], key=lambda r: float(r["psnr"]))[:3]
    worst += sorted([r for r in hyb if r["set"] == exdark], key=lambda r: -float(r["niqe"]))[:3]
    lines = ["| Panel | Set | Image | PSNR | SSIM | NIQE | Diagnosis |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(worst, 1):
        s, name = r["set"], r["image"]
        folder = s.split(" (")[0].replace(" ", "_")
        inp = read_input(s, name, src)
        out = cv2.imread(str(DATA / "outputs" / "hybrid" / folder / f"{Path(name).stem}.png"))
        tiles = [inp, out]
        if s in src:
            tiles.append(cv2.imread(str(src[s].parent / "GT" / name)))
        h = 300
        tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
        cv2.imwrite(str(d / f"failure_{i}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rgb = C.to_float(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
        dg = A.diagnose(rgb, C.estimate_illumination(rgb))
        desc = (f"brightness {dg.brightness:.3f}, noise σ {dg.noise_sigma:.1f}, spread {dg.lighting_spread:.1f} stops, "
                f"cast {dg.colour_cast:.0f} ({dg.cast_hue}){', coloured lights' if dg.coloured_lights else ''}")
        f = lambda k: f"{float(r[k]):.2f}" if r.get(k) else "-"  # noqa: E731
        lines.append(f"| failure_{i}.jpg | {s} | {name} | {f('psnr')} | {f('ssim')} | {f('niqe')} | {desc} |")
    (d / "index.md").write_text("Panels: input | hybrid output | ground truth (if any)\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    m = json.loads((RES / "metrics.json").read_text())
    rows = list(csv.DictReader(open(RES / "per_image.csv")))
    FIG.mkdir(parents=True, exist_ok=True)
    main_ = {tuple(k.split("|")): v for k, v in m["main"].items()}
    sets = list(m["n_images"])
    paired, exdark = sets[:3], sets[3]
    keys = list(P.METHODS.values())
    lat = m["latency_ms"]
    out = ["# Benchmark results\n", f"_Machine: {m['machine']}. All test images are unseen (never used for training, "
           "validation or tuning). Best value per column in **bold**._\n"]

    # 1. main comparison, metrics in the order of the problem statement
    cols = [c for c in METRICS if c[0] in ("psnr", "ssim", "lpips", "niqe", "noise", "epi", "delta_e", "overexposed")]
    for s in paired:
        out.append(f"\n### {s} ({m['n_images'][s]} image pairs)\n")
        r = [(NAMES[k], {**main_[(s, k)], "latency": lat[NAMES[k]]["ms_600x400"]}) for k in keys]
        out.append(table(r, cols[:4] + [("latency", "Latency ms (600×400) ↓", False, "{:.0f}")] + cols[4:]))
    all_paired = {k: {c[0]: float(np.mean([main_[(s, k)][c[0]] for s in paired])) for c in cols} for k in keys}
    out.append("\n### Average over the three paired test sets (215 pairs, sets weighted equally)\n")
    out.append(table([(NAMES[k], all_paired[k]) for k in keys], cols))
    out.append(f"\n### {exdark} ({m['n_images'][exdark]} real-world night photos, no ground truth)\n")
    nr = [("niqe", "NIQE ↓", False, "{:.2f}"), ("noise", "Residual noise σ ↓", False, "{:.2f}"),
          ("overexposed", "Clipped pixels % ↓", False, "{:.2f}"), ("colour_cast", "Colour cast ↓", False, "{:.1f}"),
          ("brightness", "Mean brightness", True, "{:.2f}")]
    out.append(table([(NAMES[k], main_[(exdark, k)]) for k in keys], nr, bold_best=False))
    out.append("\n_Lower NIQE = more natural statistics. Brightness is reported, not ranked._\n")

    # 2. ablation
    abl = {tuple(k.split("|")): v for k, v in m["ablation"].items()}
    names = [n for n in P.ABLATIONS if ("LOL-v1 eval15", n) in abl]
    out.append("\n## Ablation study (hybrid, one component removed at a time)\n")
    for s in ("LOL-v1 eval15", "LOL-v2 real test"):
        full = abl[(s, "Hybrid (full)")]
        r = []
        for n in names:
            v = dict(abl[(s, n)])
            v["dpsnr"] = v["psnr"] - full["psnr"]
            r.append((n, v))
        acols = [("psnr", "PSNR ↑", True, "{:.2f}"), ("dpsnr", "ΔPSNR vs full", True, "{:+.2f}"),
                 ("ssim", "SSIM ↑", True, "{:.3f}"), ("lpips", "LPIPS ↓", False, "{:.3f}"),
                 ("niqe", "NIQE ↓", False, "{:.2f}"), ("noise", "Noise σ ↓", False, "{:.2f}"),
                 ("delta_e", "ΔE ↓", False, "{:.2f}"), ("overexposed", "Overexp. % ↓", False, "{:.2f}")]
        out.append(f"\n### {s}\n")
        out.append(table(r, acols))

    # 3. robustness
    rob = {tuple(k.split("|")): v for k, v in m["robustness"].items()}
    conds = list(dict.fromkeys(k[0] for k in rob))
    rkeys = [k for k in keys if (conds[0], k) in rob]
    out.append("\n## Robustness across illumination conditions (synthetic, 30 unseen LOL-v2 references)\n")
    out.append("_Exposure scaled in linear light, plus Poisson shot and Gaussian read noise. Spotlight = strongly non-uniform lighting._\n\n")
    out.append("| Method | " + " | ".join(conds) + " |\n|---|" + "---|" * len(conds) + "\n")
    for k in rkeys:
        best = lambda c: max(rob[(c, kk)]["psnr"] for kk in rkeys)  # noqa: E731
        out.append(f"| {NAMES[k]} | " + " | ".join(
            (f"**{rob[(c, k)]['psnr']:.2f}** / {rob[(c, k)]['ssim']:.3f}" if np.isclose(rob[(c, k)]["psnr"], best(c))
             else f"{rob[(c, k)]['psnr']:.2f} / {rob[(c, k)]['ssim']:.3f}") for c in conds) + " |\n")
    out.append("\n_Cells: PSNR (dB) / SSIM._\n")

    # 4. efficiency
    out.append("\n## Efficiency (CPU only)\n\n| Method | 600×400 (ms) | 1280×960 (ms) |\n|---|---|---|\n")
    for k in keys:
        out.append(f"| {NAMES[k]} | {lat[NAMES[k]]['ms_600x400']:.0f} | {lat[NAMES[k]]['ms_1280x960']:.0f} |\n")
    out.append("\n| Model | Parameters | File size |\n|---|---|---|\n")
    for n, v in m["models"].items():
        out.append(f"| {n} | {v['params']:,} | {v['file_kb']} KB |\n")
    (RES / "tables.md").write_text("".join(out), encoding="utf-8")

    # ---------- figures ----------
    groups = [s.replace(" test", "").replace(" ", "\n", 1) for s in paired]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (k, lab) in zip(axs, [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓")]):
        vals = {g: {mk: main_[(s, mk)][k] for mk in keys} for g, s in zip(groups, paired)}
        bars(ax, groups, keys, vals, lab, "{:.1f}" if k == "psnr" else "{:.2f}", label_series=("hybrid",))
        ax.set_title(lab, loc="left", fontsize=11)
        ax.set_ylabel("")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=6, fontsize=9)
    fig.suptitle("Method comparison on unseen paired test sets", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "1_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in rkeys:
        ys = [rob[(c, k)]["psnr"] for c in conds]
        ax.plot(range(len(conds)), ys, "-o", color=COLOURS[k], lw=2, ms=6, label=NAMES[k],
                markeredgecolor=SURF, markeredgewidth=1.5)
        if k == "hybrid":
            for x, y in enumerate(ys):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(conds)), conds, rotation=15)
    ax.set_ylabel("PSNR (dB) ↑")
    ax.set_title("Robustness: darker inputs and uneven lighting", loc="left", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "2_robustness_illumination.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    full = np.mean([abl[(s, "Hybrid (full)")]["psnr"] for s in ("LOL-v1 eval15", "LOL-v2 real test")])
    items = [(n, np.mean([abl[(s, n)]["psnr"] for s in ("LOL-v1 eval15", "LOL-v2 real test")]) - full) for n in names[1:]]
    items.sort(key=lambda t: t[1])
    ax.barh([n for n, _ in items], [d for _, d in items], color="#2a78d6", height=0.6)
    for i, (_, d) in enumerate(items):
        ax.annotate(f"{d:+.2f}", (d, i), xytext=(-4 if d < 0 else 4, 0), textcoords="offset points",
                    ha="right" if d < 0 else "left", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    ax.margins(x=0.18)
    ax.set_xlabel("ΔPSNR (dB) when removed — avg. of LOL-v1 eval15 and LOL-v2 real")
    ax.set_title("Ablation: what each component contributes", loc="left", fontsize=12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "3_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logs = {n: json.loads((RES / f"train_log_{n}.json").read_text()) for n in ("dce", "restore_net", "restore_net_nomap")
            if (RES / f"train_log_{n}.json").exists()}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    if "dce" in logs:
        e = logs["dce"]["epochs"]
        axs[0].plot([0] + [x["epoch"] for x in e], [logs["dce"]["val_psnr_before_finetune"]] + [x["val_psnr"] for x in e],
                    "-o", color="#2a78d6", lw=2, ms=5)
        axs[0].set_title("Zero-DCE++ fine-tuning (validation PSNR)", loc="left", fontsize=11)
        axs[0].set_xlabel("epoch (0 = official weights)")
    for n, c, lab in (("restore_net", "#2a78d6", "with noise-level map"), ("restore_net_nomap", "#eb6834", "without map")):
        if n in logs:
            e = logs[n]["evals"]
            axs[1].plot([0] + [x["iter"] for x in e], [logs[n]["val_psnr_input"]] + [x["val_psnr"] for x in e],
                        "-o", color=c, lw=2, ms=5, label=lab)
    axs[1].set_title("Restoration U-Net (validation PSNR)", loc="left", fontsize=11)
    axs[1].set_xlabel("iteration (0 = input to the network)")
    axs[1].legend()
    for ax in axs:
        ax.set_ylabel("PSNR (dB)")
    fig.tight_layout()
    fig.savefig(FIG / "4_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # visual grid: 4 unseen examples x methods
    picks = [("LOL-v1 eval15", "1.png"), ("LOL-v2 real test", "00748.png"), ("LOL-v2 synthetic test", None), (exdark, None)]
    src = {"LOL-v1 eval15": DATA / "lol" / "lol_dataset" / "eval15" / "low",
           "LOL-v2 real test": DATA / "lolv2_real" / "Test" / "Input",
           "LOL-v2 synthetic test": DATA / "lolv2_synthetic" / "Test" / "Input"}
    grid_rows = []
    for s, name in picks:
        cand = sorted({r["image"] for r in rows if r["kind"] == "main" and r["set"] == s})
        name = name if name in cand else cand[len(cand) // 3]
        stem = Path(name).stem
        folder = s.split(" (")[0].replace(" ", "_")
        tiles = [read_input(s, name, src)]
        for k in ("he", "zero_dce_official", "classical", "dl", "hybrid"):
            tiles.append(cv2.imread(str(DATA / "outputs" / k / folder / f"{stem}.png")))
        grid_rows.append([cover(t, 300, 220) for t in tiles])
    canvas = [np.hstack(r) for r in grid_rows]
    width = canvas[0].shape[1]
    header = np.full((36, width, 3), 252, np.uint8)
    x = 0
    for lab, t in zip(["Input", "Hist. equalisation", "Zero-DCE++ (official)", "Classical (ours)", "DL only (ours)", "Hybrid (ours)"], grid_rows[0]):
        cv2.putText(header, lab, (x + 6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (11, 11, 11), 1, cv2.LINE_AA)
        x += t.shape[1]
    cv2.imwrite(str(FIG / "5_visual_comparison.jpg"), np.vstack([header] + canvas), [cv2.IMWRITE_JPEG_QUALITY, 90])
    failures(rows, src, exdark)
    print("wrote", RES / "tables.md", "and", len(list(FIG.glob("*"))), "figures")


if __name__ == "__main__":
    main()
