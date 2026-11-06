"""Integration tests for the JSON API the web frontend uses (api.py, gr.api "restore").

Runs the real pipelines in-process (no server needed). The HTTP + CORS path is exercised by
the frontend checks described in docs/FRONTEND.md.
"""
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api  # noqa: E402

rng = np.random.default_rng(1)


def data_url(img_u8, ext=".jpg"):
    ok, buf = cv2.imencode(ext, img_u8)
    mime = "jpeg" if ext == ".jpg" else ext[1:]
    return f"data:image/{mime};base64," + base64.b64encode(buf.tobytes()).decode()


def dark_scene(h=120, w=160, level=12):
    img = np.full((h, w, 3), level, np.uint8)
    img[h // 4: h // 2, w // 4: w // 2] = level * 3
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def decode(url):
    raw = base64.b64decode(url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def test_restore_returns_all_three_real_outputs():
    res = api.restore(data_url(dark_scene()))
    assert set(res["results"]) == {"classical", "dl", "hybrid"}
    for r in res["results"].values():
        out = decode(r["image"])
        assert out.shape == (120, 160, 3)
        assert out.mean() > dark_scene().mean()          # actually brightened
        assert r["ms"] >= 0 and {"brightness", "noise", "clipped_pct", "colour_cast", "sharpness",
                                 "colourfulness", "blockiness"} == set(r["quality"])
    assert res["recommendation"]["method"] in res["results"]
    assert res["recoverability"]["level"] in {"High", "Medium", "Low"}
    assert res["noticed"] and res["actions"] and res["intermediates"]
    assert all(decode(i["image"]) is not None for i in res["intermediates"])


def test_outputs_differ_between_methods():
    res = api.restore(data_url(dark_scene()))
    imgs = {k: decode(v["image"]).astype(int) for k, v in res["results"].items()}
    assert np.abs(imgs["classical"] - imgs["dl"]).mean() > 1
    assert np.abs(imgs["hybrid"] - imgs["dl"]).mean() > 1


def test_large_image_keeps_its_size_with_note():
    res = api.restore(data_url(dark_scene(1500, 2000)))
    assert res["input"]["processed_width"] == api.MAX_SIDE          # enhanced at <= 1280 px ...
    assert (res["input"]["output_width"], res["input"]["output_height"]) == (2000, 1500)  # ... returned full size
    assert decode(res["results"]["hybrid"]["image"]).shape == (1500, 2000, 3)
    assert "returned at 2000x1500" in res["input"]["note"]


def test_huge_image_is_capped_and_says_so():
    res = api.restore(data_url(dark_scene(1200, 3000)))
    assert res["input"]["output_width"] == api.OUTPUT_MAX and "3000x1200" in res["input"]["note"]


def test_diagnostics_and_strategy_are_reported():
    res = api.restore(data_url(dark_scene()))
    assert {"illumination", "noise", "sharpness", "colour", "recoverability"} <= set(res["diagnostics"])
    assert res["diagnostics"]["illumination"]["label"].startswith("Low")
    assert res["strategy"]["cases"] and all({"stage", "decision", "why"} <= set(o) for o in res["strategy"]["operations"])


def test_png_and_greyscale_inputs():
    grey = cv2.cvtColor(dark_scene(), cv2.COLOR_BGR2GRAY)
    res = api.restore(data_url(grey, ".png"))
    assert decode(res["results"]["hybrid"]["image"]).shape == (120, 160, 3)


def test_intent_changes_brightness():
    url = data_url(dark_scene())
    natural = api.restore(url, "Natural night mood")["results"]["hybrid"]["quality"]["brightness"]
    bright = api.restore(url, "Maximum visibility")["results"]["hybrid"]["quality"]["brightness"]
    assert bright > natural


@pytest.mark.parametrize("bad", ["", "data:image/png;base64,aGVsbG8=", "%%%not-base64%%%",
                                 data_url(np.zeros((10, 10, 3), np.uint8))])
def test_bad_inputs_raise_friendly_errors(bad):
    with pytest.raises(api.APIError):
        api.restore(bad)


def test_recommendation_is_not_hardwired():
    q = {"classical": {"brightness": 0.42, "noise": 1.0, "clipped_pct": 0.0, "colour_cast": 2.0},
         "dl": {"brightness": 0.2, "noise": 6.0, "clipped_pct": 0.0, "colour_cast": 9.0},
         "hybrid": {"brightness": 0.6, "noise": 1.5, "clipped_pct": 3.0, "colour_cast": 4.0}}
    assert api.recommend(q, 0.42)["method"] == "classical"


def test_origin_allow_list(monkeypatch):
    monkeypatch.setattr(api, "ALLOWED_ORIGINS", ["https://*.vercel.app"])
    assert api.origin_allowed("https://lumina-t76.vercel.app")
    assert not api.origin_allowed("https://evil.example.com")
    assert not api.origin_allowed(None)
    monkeypatch.setattr(api, "ALLOWED_ORIGINS", [])
    assert api.origin_allowed("https://anything.example")


def test_auto_keeps_hybrid_unless_clearly_beaten():
    base = {"brightness": 0.42, "noise": 1.0, "clipped_pct": 0.0, "colour_cast": 2.0}
    q = {"classical": dict(base, noise=0.5), "dl": dict(base, noise=3.0), "hybrid": dict(base, noise=1.0)}
    r = api.recommend(q, 0.42)   # classical only 0.03 better: not a clear win
    assert r["method"] == "hybrid" and "kept" in r["why"]
