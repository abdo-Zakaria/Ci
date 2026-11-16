"""Deterministic image conversion: any PIL mode -> 64x64 RGB uint8.

Method (documented choice; the brief leaves it open):
  1. Decode with Pillow. EXIF orientation is NOT applied (matches the standard
     ImageNet convention and keeps the pipeline simple and deterministic).
  2. Convert to RGB:
       - grayscale / palette / 1-bit / 16-bit  -> Pillow's convert("RGB")
       - CMYK / YCCK                           -> Pillow's convert("RGB")
         (plain conversion, no ICC profile handling)
       - images with alpha                     -> composited on a white background
  3. Resize so the SHORTER side equals ``size`` (aspect ratio preserved) with
     bicubic resampling. Pillow's resize scales its kernel support when
     downsampling, so this is antialiased.
  4. Deterministic center crop to ``size`` x ``size``.
There is no randomness anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class SourceInfo:
    width: int
    height: int
    mode: str


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow image to RGB (alpha composited on white)."""
    if img.mode == "RGB":
        return img
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return img.convert("RGB")


def resize_center_crop(img: Image.Image, size: int, resample: str = "bicubic") -> Image.Image:
    """Resize shortest side to ``size`` keeping aspect ratio, then center crop."""
    width, height = img.size
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {img.size}")
    scale = size / min(width, height)
    new_w = max(size, int(width * scale + 0.5))
    new_h = max(size, int(height * scale + 0.5))
    resized = img.resize((new_w, new_h), resample=RESAMPLING[resample])
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))


def process_image(img: Image.Image, size: int = 64, resample: str = "bicubic") -> np.ndarray:
    """PIL image -> uint8 array of shape (size, size, 3)."""
    arr = np.asarray(resize_center_crop(to_rgb(img), size, resample), dtype=np.uint8)
    if arr.shape != (size, size, 3):  # defensive; should be unreachable
        raise ValueError(f"unexpected processed shape {arr.shape}")
    return np.ascontiguousarray(arr)


def load_and_process(path: Path, size: int = 64, resample: str = "bicubic") -> tuple[np.ndarray, SourceInfo]:
    """Decode a file and return (processed array, source info).

    Raises on any decode/convert failure; the caller records the rejection.
    """
    with Image.open(path) as img:
        img.load()
        info = SourceInfo(width=img.width, height=img.height, mode=img.mode)
        return process_image(img, size, resample), info
