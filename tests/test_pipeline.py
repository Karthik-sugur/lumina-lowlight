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
