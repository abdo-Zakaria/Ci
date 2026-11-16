"""End-to-end subset construction.

Index order of the shared image array (stable and documented):
  1. all train-folder images (these become TRAIN or VAL), ordered by
     (label, file name);
  2. then all official-validation images (TEST), ordered by (label, file name).
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import PIL

from .classes import CLASSES, LABEL_TO_NAME, WNID_TO_LABEL, WNID_TO_NAME
from .config import SubsetConfig
from .preprocess import load_and_process
from .sources import SourceLayout, collect_test_files, list_train_files
from .split import stratified_split

ARRAY_FILES = ("images.npy", "labels.npy", "train_idx.npy", "val_idx.npy", "test_idx.npy")
MANIFEST_FILE = "manifest.csv"
METADATA_FILE = "metadata.json"
MANIFEST_COLUMNS = [
    "index", "split", "label", "wnid", "class_name",
    "filename", "source_path", "orig_width", "orig_height", "orig_mode",
]


@dataclass
class Entry:
    wnid: str
    filename: str
    path: Path
    origin: str  # "train" (-> TRAIN or VAL) or "val" (-> TEST)

    @property
    def label(self) -> int:
        return WNID_TO_LABEL[self.wnid]

    @property
    def source_path(self) -> str:
        return f"train/{self.wnid}/{self.filename}" if self.origin == "train" else f"val/{self.filename}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_train_entries(
    layout: SourceLayout, config: SubsetConfig, exclusions: dict[str, str], applied: set[str]
) -> list[Entry]:
    """Eligible training files: sorted by name, minus exclusions, first N per class."""
    entries: list[Entry] = []
    for cls in CLASSES:
        names = list_train_files(layout.train_dir, cls.wnid)
        kept = []
        for name in names:
            if name in exclusions:
                applied.add(name)
            else:
                kept.append(name)
        for name in kept[: config.max_train_per_class]:
            entries.append(Entry(cls.wnid, name, layout.train_dir / cls.wnid / name, "train"))
    return entries


def select_test_entries(
    layout: SourceLayout, exclusions: dict[str, str], applied: set[str]
) -> tuple[list[Entry], list[str]]:
    found, problems = collect_test_files(layout, [c.wnid for c in CLASSES])
    entries: list[Entry] = []
    for cls in CLASSES:
        for path in found[cls.wnid]:
            if path.name in exclusions:
                applied.add(path.name)
                continue
            entries.append(Entry(cls.wnid, path.name, path, "val"))
    return entries, problems


def process_entries(entries: list[Entry], config: SubsetConfig, progress: bool = True):
    """Decode + preprocess. Returns (kept entries, arrays, infos, rejected)."""
    kept, arrays, infos, rejected = [], [], [], []
    for i, entry in enumerate(entries, 1):
        try:
            arr, info = load_and_process(entry.path, config.image_size, config.resample)
        except Exception as exc:  # recorded, never silently dropped
            rejected.append({"source_path": entry.source_path, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        kept.append(entry)
        arrays.append(arr)
        infos.append(info)
        if progress and i % 1000 == 0:
            print(f"  processed {i}/{len(entries)}", file=sys.stderr, flush=True)
    return kept, arrays, infos, rejected


def build_subset(
    layout: SourceLayout,
    out_dir: Path,
    config: SubsetConfig = SubsetConfig(),
    exclusions: dict[str, str] | None = None,
    progress: bool = True,
) -> dict:
    """Build the dataset into ``out_dir`` and return the metadata dict."""
    exclusions = exclusions or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    applied: set[str] = set()

    train_entries = select_train_entries(layout, config, exclusions, applied)
    test_entries, source_problems = select_test_entries(layout, exclusions, applied)
    if progress:
        print(f"selected {len(train_entries)} train-folder and {len(test_entries)} val images", file=sys.stderr)

    kept_all, arrays, infos, rejected = process_entries(train_entries + test_entries, config, progress)

    # Split AFTER dropping undecodable files so VAL is 10 % of what actually exists.
    pool = [e for e in kept_all if e.origin == "train"]
    tests = [e for e in kept_all if e.origin == "val"]
    ordered = pool + tests  # already (label, filename) ordered by construction
    position = {id(e): i for i, e in enumerate(kept_all)}

    split_of: dict[int, str] = {}
    for cls in CLASSES:
        cls_entries = {e.filename: e for e in pool if e.wnid == cls.wnid}
        train_names, val_names = stratified_split(
            list(cls_entries), cls.label, config.val_fraction, config.split_seed
        )
        for n in train_names:
            split_of[id(cls_entries[n])] = "train"
        for n in val_names:
            split_of[id(cls_entries[n])] = "val"
    for e in tests:
        split_of[id(e)] = "test"

    images = np.stack([arrays[position[id(e)]] for e in ordered]).astype(np.uint8, copy=False)
    info_by_id = {id(e): infos[position[id(e)]] for e in ordered}
    labels = np.array([e.label for e in ordered], dtype=np.int64)
    splits = np.array([split_of[id(e)] for e in ordered])
    idx = {s: np.flatnonzero(splits == s).astype(np.int64) for s in ("train", "val", "test")}

    np.save(out_dir / "images.npy", images)
    np.save(out_dir / "labels.npy", labels)
    for s in ("train", "val", "test"):
        np.save(out_dir / f"{s}_idx.npy", idx[s])

    with open(out_dir / MANIFEST_FILE, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(MANIFEST_COLUMNS)
        for i, e in enumerate(ordered):
            info = info_by_id[id(e)]
            writer.writerow([
                i, split_of[id(e)], e.label, e.wnid, WNID_TO_NAME[e.wnid],
                e.filename, e.source_path, info.width, info.height, info.mode,
            ])

    counts = {s: {LABEL_TO_NAME[c.label]: int(((splits == s) & (labels == c.label)).sum()) for c in CLASSES}
              for s in ("train", "val", "test")}
    metadata = {
        "config": config.to_dict(),
        "classes": [{"label": c.label, "wnid": c.wnid, "name": c.name} for c in CLASSES],
        "array": {"shape": list(images.shape), "dtype": str(images.dtype),
                  "bytes": int((out_dir / "images.npy").stat().st_size)},
        "totals": {s: int(len(idx[s])) for s in idx} | {"all": int(len(images))},
        "per_class_counts": counts,
        "source_modes": dict(sorted(Counter(info_by_id[id(e)].mode for e in ordered).items())),
        "non_rgb_converted": int(sum(info_by_id[id(e)].mode != "RGB" for e in ordered)),
        "rejected_files": rejected,
        "exclusions_applied": sorted(applied),
        "exclusions_not_found": sorted(set(exclusions) - applied),
        "source_problems": source_problems,
        "preprocessing": ("decode -> RGB (alpha on white; CMYK/gray via Pillow) -> resize shortest side "
                          f"to {config.image_size} ({config.resample}) -> center crop; no EXIF rotation"),
        "split_method": ("per class: sort file names, permute with numpy default_rng([seed, label]), "
                         "first round-half-up(n*val_fraction) -> VAL, rest -> TRAIN"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
        "sha256": {name: sha256_file(out_dir / name) for name in (*ARRAY_FILES, MANIFEST_FILE)},
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
