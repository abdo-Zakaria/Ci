"""Deterministic, class-stratified TRAIN/VAL split."""

from __future__ import annotations

import numpy as np


def val_count(n: int, val_fraction: float) -> int:
    """Number of VAL images for a class with ``n`` eligible images.

    The brief gives no rounding rule, so we use round-half-up of
    ``n * val_fraction`` (1300 -> 130, 1299 -> 130, 1294 -> 129).
    """
    return int(n * val_fraction + 0.5 + 1e-9)


def stratified_split(
    filenames: list[str], label: int, val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split one class's filenames into (train, val), both sorted.

    The input is sorted first so the result never depends on directory
    listing order. The shuffle is seeded per class with ``[seed, label]``.
    """
    ordered = sorted(filenames)
    n_val = val_count(len(ordered), val_fraction)
    rng = np.random.default_rng([seed, label])
    perm = rng.permutation(len(ordered))
    val_positions = set(perm[:n_val].tolist())
    val = [f for i, f in enumerate(ordered) if i in val_positions]
    train = [f for i, f in enumerate(ordered) if i not in val_positions]
    return train, val
