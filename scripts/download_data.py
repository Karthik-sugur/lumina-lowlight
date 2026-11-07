"""Download every dataset + the pretrained Zero-DCE++ weights used in this project.

Usage:  python scripts/download_data.py          (data goes to $DATA_DIR, default ./data)
"""
import os
import random
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DATA = Path(os.environ.get("DATA_DIR", "data"))
ROOT = Path(__file__).resolve().parents[1]
ZERO_DCE_PP = ("https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/"
               "Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth")


def fetch_zip(repo, filename, dest):
    if dest.exists():
        return
    z = hf_hub_download(repo, filename, repo_type="dataset", local_dir=DATA / "_downloads")
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    print("extracted", dest)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # LOL-v1: our485 (train/val) + eval15 (test)  -- the official P1 dataset
    fetch_zip("geekyrakshit/LoL-Dataset", "lol_dataset.zip", DATA / "lol")
    # LOL-v2 test splits only: unseen paired data (real camera + synthetic)
    for name in ("real", "synthetic"):
        snapshot_download(f"okhater/lolv2-{name}", repo_type="dataset",
                          allow_patterns=["Test/*"], local_dir=DATA / f"lolv2_{name}")
    # Fresh validation data for scripts/validate.py: 100 LOL-v2 synthetic *training* pairs (seed 7),
    # used for no training and no test; disjoint from LOL-v1 and from every test set (pixel-hash checked)
    files = HfApi().list_repo_files("okhater/lolv2-synthetic", repo_type="dataset")
    gts = {f for f in files if f.startswith("Train/GT/")}
    pick = random.Random(7).sample(sorted(f for f in files if f.startswith("Train/Input/")), 100)
    pairs = [p for f in pick for p in (f, f"Train/GT/{f.split('/')[-1]}") if f"Train/GT/{f.split('/')[-1]}" in gts]
    snapshot_download("okhater/lolv2-synthetic", repo_type="dataset", allow_patterns=pairs,
                      local_dir=DATA / "lolv2_synthetic_trainval")
    # ExDark: unseen real-world night photos, no ground truth (no-reference metrics)
    fetch_zip("hieupth/exdark_384", "images.zip", DATA / "exdark")
    # Official Zero-DCE++ weights (starting point for fine-tuning)
    w = ROOT / "weights" / "zero_dce_pp_official.pth"
    if not w.exists():
        w.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ZERO_DCE_PP, w)
    print("done ->", DATA.resolve())


if __name__ == "__main__":
    main()
