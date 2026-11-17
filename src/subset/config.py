"""Build configuration. Defaults follow the project brief (section 3) and card."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SubsetConfig:
    image_size: int = 64
    # Brief: "up to 1,300 training images per class".
    max_train_per_class: int = 1300
    # Brief: "Hold out a stratified VAL split of 10 %".
    val_fraction: float = 0.10
    # Brief: TEST is the 50 official validation images of each class.
    test_per_class: int = 50
    split_seed: int = 42
    # Pillow resampling filter used for the shortest-side resize.
    resample: str = "bicubic"
    # The array file must stay under this many bytes (brief: "under 200 MB").
    max_array_bytes: int = 200 * 1000 * 1000

    def to_dict(self) -> dict:
        return asdict(self)
