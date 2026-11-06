"""Smoke + property tests.  Run:  python -m pytest -q   (or  python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowlight import analysis as A  # noqa: E402
from lowlight import classical as C  # noqa: E402
from lowlight import pipeline as P  # noqa: E402

rng = np.random.default_rng(0)


def dark_image(h=120, w=160, level=0.05):
    base = np.linspace(0.2, 1.0, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    base[h // 3:, w // 2:] *= 0.3  # an edge + darker region
    return np.clip(base * level + rng.normal(0, 0.004, (h, w, 3)), 0, 1).astype(np.float32)


def test_guided_filter_keeps_flat_and_edges():
    flat = np.full((50, 50), 0.4, np.float32)
    assert np.allclose(C.guided_filter(flat, flat, 4, 1e-3), 0.4, atol=1e-4)
    step = np.zeros((50, 50), np.float32)
    step[:, 25:] = 1
    out = C.guided_filter(step, step, 4, 1e-4)
    assert out[:, :20].max() < 0.05 and out[:, 30:].min() > 0.95  # edge not smeared


def test_gamma_solver_hits_target():
    img = dark_image()
    t = C.estimate_illumination(img)
    y = C.luminance(img)
    g = C.solve_gamma(y, t, 0.4)
    assert abs(np.clip(y * t ** (g - 1), 0, 1).mean() - 0.4) < 0.02


def test_bright_image_gets_light_touch():
    img = np.clip(dark_image(level=1.0) + 0.3, 0, 1)
    t = C.estimate_illumination(img)
    assert C.solve_gamma(C.luminance(img), t, A.TARGET_BRIGHTNESS) == 1.0


def test_all_methods_valid_output():
    img = dark_image()
    for key in P.METHODS.values():
        out = P.run(img, key)["output"]
        assert out.shape == img.shape and out.dtype == np.float32, key
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1, key
        assert out.mean() > img.mean(), key  # every method brightens a dark image


def test_hybrid_brightens_and_respects_intent():
    img = dark_image()
    auto = P.run(img, "hybrid")["output"].mean()
    natural = P.run(img, "hybrid", preset="Natural night mood")["output"].mean()
    assert auto > 4 * img.mean() and natural < auto


def test_ablation_flags_run():
    img = dark_image(64, 64)
    for flags in P.ABLATIONS.values():
        if "noise_map" in flags and not P.NOMAP_WEIGHTS.exists():
            continue
        assert P.run(img, "hybrid", **flags)["output"].shape == img.shape


def test_tiny_and_extreme_inputs():
    for img in (np.zeros((32, 32, 3), np.float32), np.ones((40, 33, 3), np.float32), dark_image(33, 47, 0.005)):
        out = P.run(img, "hybrid")["output"]
        assert np.isfinite(out).all() and out.shape == img.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


# ---------- image-quality-aware strategy ----------
def textured(h=96, w=128, seed=1, amp=0.6, tint=(1.0, 0.95, 0.85)):
    """Grey block texture with a mild warm tint (a natural, low-colourfulness scene)."""
    r = np.random.default_rng(seed)
    import cv2
    t = cv2.resize(r.random((h // 4, w // 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return np.clip((0.5 + amp * (t[..., None] - 0.5)) * np.array(tint, np.float32), 0, 1)


def test_detail_recovery_zero_strength_is_noop_and_never_sharpens_pure_noise():
    x = textured()
    assert np.allclose(C.recover_detail(x, x, 1.0, 0, 0, 0), x, atol=1e-5)
    flat_noisy = np.clip(0.5 + rng.normal(0, 3 / 255, (96, 128, 3)), 0, 1).astype(np.float32)
    out = C.recover_detail(flat_noisy, None, 3.0, 0, 0.5, 0.25)  # residual noise sigma = 3/255
    assert A.estimate_noise(C.luminance(out)) < A.estimate_noise(C.luminance(flat_noisy)) * 1.1  # coring


def test_detail_recovery_sharpens_soft_clean_image_without_halos():
    import cv2
    x = textured(amp=0.08)                      # fine, low-contrast texture
    soft = cv2.GaussianBlur(x, (0, 0), 1.2)
    out = C.recover_detail(soft, None, 0.5, 0, 0.5, 0.25)
    assert A.sharpness_index(out) > A.sharpness_index(soft) * 1.1
    y, yo = C.luminance(soft), C.luminance(out)
    k = np.ones((3, 3), np.uint8)
    assert (yo <= cv2.dilate(y, k) + 0.04 + 1e-4).all() and (yo >= cv2.erode(y, k) - 0.04 - 1e-4).all()


def test_saturation_ceiling_only_touches_oversaturated():
    natural = textured()
    assert np.allclose(C.limit_saturation(natural, 0.55), natural, atol=1e-6)
    garish = textured(tint=(1.0, 0.15, 0.8))
    assert A.colourfulness(garish) > 0.55 and A.colourfulness(C.limit_saturation(garish, 0.55)) <= 0.56


def test_blockiness_detects_jpeg_blocks():
    import cv2
    x = cv2.GaussianBlur(textured(256, 256), (0, 0), 2)
    ok, buf = cv2.imencode(".jpg", C.to_u8(x), [cv2.IMWRITE_JPEG_QUALITY, 15])
    jpg = C.to_float(cv2.imdecode(buf, 1))
    assert A.blockiness(x) < A.BLOCKY < A.blockiness(jpg)


def test_run_full_keeps_input_size_and_reports_it():
    big = np.clip(dark_image(300, 400) + 0.02 * textured(300, 400), 0, 1).astype(np.float32)
    r = P.run_full(big, "hybrid", process_side=200)
    assert r["output"].shape == big.shape and r["output_size"] == (400, 300) and r["processed_size"] == (200, 150)
    small = P.run_full(big[:100, :150], "classical", process_side=200)
    assert small["processed_size"] == small["output_size"] == (150, 100)


def test_strategy_and_labels_are_consistent():
    img = dark_image(level=0.02)
    r = P.run(img, "hybrid")
    d, p = r["diagnosis"], r["params"]
    lab = A.quality_labels(d, p, p.sharp_restored, p.residual_noise, A.recoverability(d, p)[0])
    assert lab["illumination"]["label"].startswith("Low")
    st = A.strategy(d, p, P.FULL)
    assert "Extremely dark" in st["cases"] and any(o["stage"] == "Detail recovery" for o in st["operations"])
