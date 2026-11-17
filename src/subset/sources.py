"""Locating raw ImageNet files, official validation ground truth, exclusions.

Nothing here modifies the raw data; everything is read-only.

Supported official-validation label sources (verify against your copy!):
  1. Kaggle ``LOC_val_solution.csv`` (header ``ImageId,PredictionString``; the
     first token of PredictionString is the WordNet ID).
  2. A two-column file ``<image name>,<wnid>`` (comma, tab or space separated).
  3. A validation directory already arranged as ``val/<wnid>/<image>.JPEG``.
The devkit's integer-only ``ILSVRC2012_validation_ground_truth.txt`` is
deliberately NOT supported: its integers are ILSVRC2012 IDs that need meta.mat
to translate, and silently assuming sorted-WordNet-ID order would mislabel images.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .classes import WNID_PATTERN

IMAGE_SUFFIXES = {".jpeg", ".jpg"}
_ROOT_CANDIDATES = ("ILSVRC/Data/CLS-LOC", "Data/CLS-LOC", "CLS-LOC", ".")
_LABEL_FILE_CANDIDATES = ("LOC_val_solution.csv", "val_labels.csv", "val_ground_truth.csv")


class SourceError(RuntimeError):
    """Raised when the raw data is missing or inconsistent."""


@dataclass(frozen=True)
class SourceLayout:
    train_dir: Path
    val_dir: Path
    val_labels: Path | None  # None when val_dir is already arranged by WordNet ID


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def resolve_source_layout(
    root: Path | None,
    train_dir: Path | None = None,
    val_dir: Path | None = None,
    val_labels: Path | None = None,
) -> SourceLayout:
    """Find train/val directories and the validation label file."""
    if root is not None:
        root = Path(root)
        if not root.is_dir():
            raise SourceError(f"ImageNet root does not exist: {root}")
        for sub in _ROOT_CANDIDATES:
            base = root / sub
            if (base / "train").is_dir():
                train_dir = train_dir or base / "train"
                val_dir = val_dir or base / "val"
                break
        if val_labels is None:
            for name in _LABEL_FILE_CANDIDATES:
                if (root / name).is_file():
                    val_labels = root / name
                    break
    if train_dir is None or not Path(train_dir).is_dir():
        raise SourceError(
            f"Could not find the ImageNet train directory (looked under {root} in "
            f"{list(_ROOT_CANDIDATES)}). Pass --train-dir explicitly."
        )
    if val_dir is None or not Path(val_dir).is_dir():
        raise SourceError("Could not find the ImageNet val directory. Pass --val-dir explicitly.")
    return SourceLayout(Path(train_dir), Path(val_dir), Path(val_labels) if val_labels else None)


def list_train_files(train_dir: Path, wnid: str) -> list[str]:
    """Sorted image file names in ``train/<wnid>/`` (deterministic order)."""
    class_dir = train_dir / wnid
    if not class_dir.is_dir():
        raise SourceError(f"Missing training folder for {wnid}: {class_dir}")
    names = sorted(p.name for p in class_dir.iterdir() if _is_image(p))
    if not names:
        raise SourceError(f"Training folder for {wnid} contains no images: {class_dir}")
    return names


def read_val_ground_truth(path: Path) -> dict[str, str]:
    """Parse a validation label file into ``{image stem: wnid}``.

    Raises SourceError for unsupported or malformed formats instead of guessing.
    """
    rows: list[list[str]] = []
    with open(path, newline="") as fh:
        text = fh.read()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise SourceError(f"Validation label file is empty: {path}")

    first = re.split(r"[,\t ]+", lines[0].strip())
    if first[0].lower() == "imageid":  # Kaggle LOC_val_solution.csv
        reader = csv.reader(lines)
        next(reader)
        mapping: dict[str, str] = {}
        for row in reader:
            if len(row) < 2 or not row[1].strip():
                raise SourceError(f"Malformed row in {path}: {row}")
            wnid = row[1].split()[0]
            if not WNID_PATTERN.match(wnid):
                raise SourceError(f"Unexpected PredictionString start {wnid!r} in {path}")
            stem = Path(row[0]).stem if "." in row[0] else row[0]
            if stem in mapping:
                raise SourceError(f"Duplicate image id {stem} in {path}")
            mapping[stem] = wnid
        return mapping

    if all(re.fullmatch(r"\d+", ln.strip()) for ln in lines[:20]):
        raise SourceError(
            f"{path} looks like the devkit's integer ground-truth list. Its integers are "
            "ILSVRC2012 IDs that require meta.mat to map to WordNet IDs; refusing to guess. "
            "Use Kaggle's LOC_val_solution.csv or a two-column '<image>,<wnid>' file."
        )

    mapping = {}
    for ln in lines:
        parts = [p for p in re.split(r"[,\t ]+", ln.strip()) if p]
        if len(parts) != 2 or not WNID_PATTERN.match(parts[1]):
            if parts and parts[0].lower() in {"image", "filename", "file", "name"}:
                continue  # header
            raise SourceError(f"Unrecognised validation label line in {path}: {ln!r}")
        stem = Path(parts[0]).stem if "." in parts[0] else parts[0]
        if stem in mapping:
            raise SourceError(f"Duplicate image id {stem} in {path}")
        mapping[stem] = parts[1]
    return mapping


def collect_test_files(layout: SourceLayout, wnids: list[str]) -> tuple[dict[str, list[Path]], list[str]]:
    """Official validation images belonging to ``wnids``.

    Returns ({wnid: sorted paths}, problems). A problem is e.g. a file listed in the
    ground truth but absent on disk. Nothing is fabricated or substituted.
    """
    problems: list[str] = []
    found: dict[str, list[Path]] = {w: [] for w in wnids}
    val_dir = layout.val_dir

    if layout.val_labels is None:
        subdirs = [w for w in wnids if (val_dir / w).is_dir()]
        if not subdirs:
            raise SourceError(
                f"No validation label file given and {val_dir} has no <wnid>/ subfolders for "
                "our classes. Provide --val-labels (e.g. LOC_val_solution.csv)."
            )
        for w in wnids:
            if w not in subdirs:
                problems.append(f"val folder missing for {w}")
                continue
            found[w] = sorted((p for p in (val_dir / w).iterdir() if _is_image(p)), key=lambda p: p.name)
        return found, problems

    truth = read_val_ground_truth(layout.val_labels)
    on_disk = {p.stem: p for p in val_dir.iterdir() if _is_image(p)}
    wanted = set(wnids)
    for stem, wnid in sorted(truth.items()):
        if wnid not in wanted:
            continue
        if stem not in on_disk:
            problems.append(f"{stem} is listed for {wnid} in {layout.val_labels.name} but is missing in {val_dir}")
            continue
        found[wnid].append(on_disk[stem])
    for w in wnids:
        found[w].sort(key=lambda p: p.name)
    return found, problems


def load_exclusions(path: Path | None) -> dict[str, str]:
    """Read the persistent exclusion list: ``<original file name>  # reason`` per line."""
    exclusions: dict[str, str] = {}
    if path is None or not Path(path).is_file():
        return exclusions
    for raw in Path(path).read_text().splitlines():
        body, _, reason = raw.partition("#")
        body = body.strip()
        if not body:
            continue
        exclusions[body.split()[0]] = reason.strip()
    return exclusions
