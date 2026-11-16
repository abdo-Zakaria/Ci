"""Independent validation of a built subset directory.

Re-reads the files from disk (it does not trust build-time state) and checks the
requirements of the brief and card. Errors mean the dataset must not be used.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .build import ARRAY_FILES, MANIFEST_COLUMNS, MANIFEST_FILE, METADATA_FILE, sha256_file
from .classes import CLASSES, LABEL_TO_NAME, NUM_CLASSES, WNID_TO_LABEL, WNID_TO_NAME
from .config import SubsetConfig
from .split import val_count


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)  # split -> class name -> n
    checks_run: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def check(self, condition: bool, message: str) -> bool:
        self.checks_run += 1
        if not condition:
            self.errors.append(message)
        return condition

    def format_summary(self) -> str:
        names = [c.name for c in CLASSES]
        lines = [f"{'class':<16}{'TRAIN':>8}{'VAL':>8}{'TEST':>8}"]
        for n in names:
            lines.append(f"{n:<16}" + "".join(f"{self.counts.get(s, {}).get(n, 0):>8}" for s in ("train", "val", "test")))
        totals = {s: sum(self.counts.get(s, {}).values()) for s in ("train", "val", "test")}
        lines.append(f"{'TOTAL':<16}" + "".join(f"{totals[s]:>8}" for s in ("train", "val", "test")))
        lines.append(f"checks run: {self.checks_run}   errors: {len(self.errors)}   warnings: {len(self.warnings)}")
        lines += [f"ERROR: {e}" for e in self.errors] + [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_dataset(out_dir: Path, config: SubsetConfig = SubsetConfig(), exclusions: dict[str, str] | None = None) -> ValidationReport:
    rep = ValidationReport()
    out_dir = Path(out_dir)
    exclusions = exclusions or {}

    missing = [n for n in (*ARRAY_FILES, MANIFEST_FILE, METADATA_FILE) if not (out_dir / n).is_file()]
    if not rep.check(not missing, f"missing output files: {missing}"):
        return rep

    images = np.load(out_dir / "images.npy", mmap_mode="r")
    labels = np.load(out_dir / "labels.npy")
    idx = {s: np.load(out_dir / f"{s}_idx.npy") for s in ("train", "val", "test")}
    metadata = json.loads((out_dir / METADATA_FILE).read_text())
    with open(out_dir / MANIFEST_FILE, newline="") as fh:
        reader = csv.DictReader(fh)
        rep.check(reader.fieldnames == MANIFEST_COLUMNS, f"manifest columns {reader.fieldnames} != {MANIFEST_COLUMNS}")
        rows = list(reader)
    n = len(rows)

    # --- image array -------------------------------------------------------
    rep.check(images.ndim == 4 and images.shape[1:] == (config.image_size, config.image_size, 3),
              f"images shape {images.shape} is not (N, {config.image_size}, {config.image_size}, 3)")
    rep.check(images.dtype == np.uint8, f"images dtype {images.dtype} is not uint8")
    rep.check(images.shape[0] == n, f"{images.shape[0]} images but {n} manifest rows")
    size = (out_dir / "images.npy").stat().st_size
    rep.check(size < config.max_array_bytes, f"images.npy is {size} bytes, limit {config.max_array_bytes}")

    # --- classes and labels -----------------------------------------------
    pairs = {(int(r["label"]), r["wnid"]) for r in rows}
    rep.check(pairs == {(c.label, c.wnid) for c in CLASSES}, f"manifest classes {sorted(pairs)} differ from the card")
    rep.check(len(pairs) == NUM_CLASSES, f"expected exactly {NUM_CLASSES} classes, found {len(pairs)}")
    rep.check(all(WNID_TO_LABEL.get(r["wnid"]) == int(r["label"]) and WNID_TO_NAME.get(r["wnid"]) == r["class_name"] for r in rows),
              "a manifest row has a wnid/label/name that disagrees with the card")
    rep.check(labels.shape == (n,) and labels.dtype == np.int64, f"labels.npy shape/dtype {labels.shape}/{labels.dtype} wrong")
    if labels.shape == (n,):
        rep.check(bool(np.array_equal(labels, np.array([WNID_TO_LABEL[r["wnid"]] for r in rows]))),
                  "labels.npy does not match the wnid->label mapping of the manifest")
    rep.check(all(int(r["index"]) == i for i, r in enumerate(rows)), "manifest 'index' column is not 0..N-1 in order")
    rep.check(all(r["source_path"].startswith(("train/", "val/")) for r in rows), "manifest source_path has unexpected prefix")
    rep.check(len({r["source_path"] for r in rows}) == n, "duplicate source files in manifest")

    # --- index files -------------------------------------------------------
    for s, a in idx.items():
        rep.check(a.ndim == 1 and np.issubdtype(a.dtype, np.integer), f"{s}_idx.npy must be a 1-D integer array")
        rep.check(len(np.unique(a)) == len(a), f"{s}_idx.npy contains duplicate indices")
        rep.check(len(a) == 0 or (a.min() >= 0 and a.max() < n), f"{s}_idx.npy has out-of-range indices")
    sets = {s: set(a.tolist()) for s, a in idx.items()}
    rep.check(not (sets["train"] & sets["val"]), "TRAIN and VAL overlap")
    rep.check(not (sets["train"] & sets["test"]), "TRAIN and TEST overlap")
    rep.check(not (sets["val"] & sets["test"]), "VAL and TEST overlap")
    rep.check(sets["train"] | sets["val"] | sets["test"] == set(range(n)), "some stored images belong to no split")
    for s in idx:
        rep.check(all(rows[i]["split"] == s for i in sets[s] if i < n), f"manifest 'split' column disagrees with {s}_idx.npy")
    rep.check(all((rows[i]["source_path"].startswith("val/")) for i in sets["test"] if i < n), "TEST contains non-validation source files")
    rep.check(all((rows[i]["source_path"].startswith("train/")) for s in ("train", "val") for i in sets[s] if i < n),
              "TRAIN/VAL contain non-training source files")
    train_names = {rows[i]["filename"] for s in ("train", "val") for i in sets[s] if i < n}
    test_names = {rows[i]["filename"] for i in sets["test"] if i < n}
    rep.check(not (train_names & test_names), "a file name appears in both the train pool and TEST")

    # --- class distribution -----------------------------------------------
    per = {s: {c.label: 0 for c in CLASSES} for s in idx}
    for s, a in idx.items():
        for i in a.tolist():
            if i < n:
                per[s][int(rows[i]["label"])] += 1
    rep.counts = {s: {LABEL_TO_NAME[k]: v for k, v in d.items()} for s, d in per.items()}
    for c in CLASSES:
        pool = per["train"][c.label] + per["val"][c.label]
        rep.check(pool > 0, f"class {c.label} ({c.name}) has no training-folder images")
        rep.check(pool <= config.max_train_per_class, f"class {c.label} uses {pool} > {config.max_train_per_class} train images")
        rep.check(per["val"][c.label] == val_count(pool, config.val_fraction),
                  f"class {c.label} VAL count {per['val'][c.label]} != expected {val_count(pool, config.val_fraction)} for {pool} images")
        rep.check(per["test"][c.label] == config.test_per_class,
                  f"class {c.label} ({c.name}) has {per['test'][c.label]} TEST images, expected {config.test_per_class} "
                  "(official validation data incomplete, or an image was excluded/rejected)")
        if pool < config.max_train_per_class:
            rep.warnings.append(f"class {c.label} ({c.name}) has only {pool} eligible train images (< {config.max_train_per_class})")

    # --- exclusions, rejects, hashes ---------------------------------------
    stored = {r["filename"] for r in rows}
    rep.check(not (stored & set(exclusions)), f"excluded files present in the dataset: {sorted(stored & set(exclusions))}")
    for name in metadata.get("exclusions_not_found", []):
        rep.warnings.append(f"exclusion entry {name} matched no source file")
    for item in metadata.get("rejected_files", []):
        rep.warnings.append(f"rejected (undecodable) file {item['source_path']}: {item['reason']}")
    for item in metadata.get("source_problems", []):
        rep.warnings.append(f"source problem: {item}")
    for name, digest in metadata.get("sha256", {}).items():
        rep.check(sha256_file(out_dir / name) == digest, f"sha256 mismatch for {name}")
    rep.check(metadata.get("totals", {}).get("all") == n, "metadata total disagrees with manifest")
    return rep
