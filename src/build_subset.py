#!/usr/bin/env python3
"""Build the team-T09 64x64 ImageNet subset from raw ILSVRC-2012 files.

Example (Kaggle notebook with the ImageNet competition data attached):
    python src/build_subset.py \
        --imagenet-root /kaggle/input/imagenet-object-localization-challenge \
        --out /kaggle/working/subset
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from subset.build import build_subset
from subset.config import SubsetConfig
from subset.sources import SourceError, load_exclusions, resolve_source_layout
from subset.validate import validate_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
LIGHT_FILES = ("labels.npy", "train_idx.npy", "val_idx.npy", "test_idx.npy", "manifest.csv", "metadata.json")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagenet-root", type=Path, help="folder containing ILSVRC/Data/CLS-LOC (Kaggle layout)")
    p.add_argument("--train-dir", type=Path, help="override: folder containing <wnid>/ training folders")
    p.add_argument("--val-dir", type=Path, help="override: folder with official validation images")
    p.add_argument("--val-labels", type=Path, help="override: LOC_val_solution.csv or '<image>,<wnid>' file")
    p.add_argument("--out", type=Path, required=True, help="output folder for images.npy and all dataset files (not committed)")
    p.add_argument("--light-out", type=str, default=str(REPO_ROOT / "results" / "dataset"),
                   help="also copy the small files (indices, labels, manifest, metadata) here; pass '' to skip")
    p.add_argument("--exclusions", type=Path, default=REPO_ROOT / "configs" / "exclusions.txt")
    p.add_argument("--seed", type=int, default=SubsetConfig.split_seed)
    p.add_argument("--check-reproducible", action="store_true", help="build a second time and compare sha256 of every output")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = SubsetConfig(split_seed=args.seed)
    try:
        layout = resolve_source_layout(args.imagenet_root, args.train_dir, args.val_dir, args.val_labels)
        exclusions = load_exclusions(args.exclusions)
        print(f"train dir: {layout.train_dir}\nval dir:   {layout.val_dir}\nval labels: {layout.val_labels}\n"
              f"exclusions loaded: {len(exclusions)}", file=sys.stderr)
        metadata = build_subset(layout, args.out, config, exclusions, progress=not args.quiet)
    except SourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = validate_dataset(args.out, config, exclusions)
    print(report.format_summary())
    print(f"images.npy: shape={metadata['array']['shape']} dtype={metadata['array']['dtype']} bytes={metadata['array']['bytes']}")
    print(f"non-RGB sources converted: {metadata['non_rgb_converted']}  modes: {metadata['source_modes']}")

    if args.check_reproducible:
        with tempfile.TemporaryDirectory() as tmp:
            second = build_subset(layout, Path(tmp), config, exclusions, progress=False)
        same = second["sha256"] == metadata["sha256"]
        print(f"reproducibility (second build, all file hashes identical): {same}")
        if not same:
            return 3

    if args.light_out and report.ok:
        light = Path(args.light_out)
        light.mkdir(parents=True, exist_ok=True)
        for name in LIGHT_FILES:
            shutil.copy2(args.out / name, light / name)
        print(f"copied lightweight files to {light}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
