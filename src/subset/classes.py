"""Single source of truth for the team's ten classes and label numbers.

Source: Team T09 Assignment Card (the card's class names are used verbatim).
Labels are the project's own 0-9 numbers, NOT ImageNet's 0-999 indices, and are
always derived from the WordNet ID, never from folder order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WNID_PATTERN = re.compile(r"^n\d{8}$")


@dataclass(frozen=True)
class ImageNetClass:
    label: int
    wnid: str
    name: str


CLASSES: tuple[ImageNetClass, ...] = (
    ImageNetClass(0, "n01807496", "partridge"),
    ImageNetClass(1, "n02102318", "cocker spaniel"),
    ImageNetClass(2, "n02102480", "Sussex spaniel"),
    ImageNetClass(3, "n02256656", "cicada"),
    ImageNetClass(4, "n02268443", "dragonfly"),
    ImageNetClass(5, "n02859443", "boathouse"),
    ImageNetClass(6, "n03876231", "paintbrush"),
    ImageNetClass(7, "n03947888", "pirate"),
    ImageNetClass(8, "n04592741", "wing"),
    ImageNetClass(9, "n07684084", "French loaf"),
)

WNID_TO_LABEL: dict[str, int] = {c.wnid: c.label for c in CLASSES}
LABEL_TO_WNID: dict[int, str] = {c.label: c.wnid for c in CLASSES}
WNID_TO_NAME: dict[str, str] = {c.wnid: c.name for c in CLASSES}
LABEL_TO_NAME: dict[int, str] = {c.label: c.name for c in CLASSES}
NUM_CLASSES = len(CLASSES)


def _self_check() -> None:
    assert [c.label for c in CLASSES] == list(range(NUM_CLASSES)), "labels must be 0..9 in order"
    assert len(WNID_TO_LABEL) == NUM_CLASSES, "duplicate WordNet IDs in class table"
    assert all(WNID_PATTERN.match(c.wnid) for c in CLASSES), "malformed WordNet ID"


_self_check()
