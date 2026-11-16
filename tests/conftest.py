"""Synthetic ImageNet-like fixture. NOT real ImageNet data.

Each of our ten classes gets a distinctive base colour, so a wrong
wnid->label assignment shows up as a wrong mean colour after preprocessing.
Distractor validation images (other WordNet IDs) must never reach TEST.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from subset.classes import CLASSES

N_TRAIN = 20  # images per class in the fake train folders
N_VAL = 5     # official validation images per class in the fake val folder
N_DISTRACTOR = 12


def class_color(label: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(label / 10.0, 0.9, 0.9)
    return int(r * 255), int(g * 255), int(b * 255)


def make_image(label: int, seed: int, mode: str = "RGB", size: tuple[int, int] = (90, 70)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array(class_color(label), dtype=np.int16)
    arr = np.clip(base + rng.integers(-12, 13, size=(size[1], size[0], 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    if mode == "L":
        return img.convert("L")
    if mode == "CMYK":
        return img.convert("CMYK")
    if mode == "RGBA":
        return img.convert("RGBA")
    if mode == "P":
        return img.convert("P")
    return img


@pytest.fixture()
def fake_imagenet(tmp_path: Path):
    """Returns (root, layout kwargs). Layout mimics Kaggle's ILSVRC/Data/CLS-LOC."""
    root = tmp_path / "imagenet"
    cls_loc = root / "ILSVRC" / "Data" / "CLS-LOC"
    modes = ["RGB", "RGB", "L", "CMYK", "RGBA", "P"]
    sizes = [(90, 70), (50, 120), (64, 64), (30, 30), (200, 150)]
    for c in CLASSES:
        folder = cls_loc / "train" / c.wnid
        folder.mkdir(parents=True)
        for i in range(N_TRAIN):
            mode = modes[i % len(modes)]
            img = make_image(c.label, 1000 * c.label + i, mode, sizes[i % len(sizes)])
            if mode in ("RGBA", "P"):
                img.save(folder / f"{c.wnid}_{i:04d}.PNG".replace("PNG", "JPEG"), format="PNG")  # PNG disguised as JPEG
            else:
                img.save(folder / f"{c.wnid}_{i:04d}.JPEG", format="JPEG")

    val_dir = cls_loc / "val"
    val_dir.mkdir(parents=True)
    rows = ["ImageId,PredictionString"]
    k = 1
    for c in CLASSES:
        for j in range(N_VAL):
            name = f"ILSVRC2012_val_{k:08d}"
            make_image(c.label, 5_000_000 + k).save(val_dir / f"{name}.JPEG", format="JPEG")
            rows.append(f"{name},{c.wnid} 1 2 30 40")
            k += 1
    for j in range(N_DISTRACTOR):
        name = f"ILSVRC2012_val_{k:08d}"
        make_image(3, 9_000_000 + k).save(val_dir / f"{name}.JPEG", format="JPEG")
        rows.append(f"{name},n99999999 1 2 30 40")
        k += 1
    (root / "LOC_val_solution.csv").write_text("\n".join(rows) + "\n")
    return root
