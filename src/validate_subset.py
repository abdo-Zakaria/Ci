#!/usr/bin/env python3
"""Validate an already-built subset folder: python src/validate_subset.py --data data/subset"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from subset.config import SubsetConfig
from subset.sources import load_exclusions
from subset.validate import validate_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--exclusions", type=Path, default=REPO_ROOT / "configs" / "exclusions.txt")
    args = p.parse_args(argv)
    report = validate_dataset(args.data, SubsetConfig(), load_exclusions(args.exclusions))
    print(report.format_summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
