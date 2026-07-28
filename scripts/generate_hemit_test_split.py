#!/usr/bin/env python3
"""Reproduce the released HEMIT paper-test tile selection.

The source HEMIT test tiles are 1024x1024 crops made with a 512-pixel WSI
stride. Retaining tiles with even row and column indices makes the retained
1024x1024 tiles non-overlapping. The released exclusion list removes the six
empty tiles omitted from the paper evaluation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATCH_PATTERN = re.compile(r"_patch_(\d+)_(\d+)\.[^.]+$")


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def patch_indices(name: str) -> tuple[int, int]:
    match = PATCH_PATTERN.search(name)
    if match is None:
        raise ValueError(f"Cannot parse HEMIT patch indices from {name!r}")
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-test-ids", type=Path, required=True, help="One original 1024x1024 test filename per line.")
    parser.add_argument("--output", type=Path, required=True, help="Output file containing the selected sample IDs.")
    parser.add_argument(
        "--excluded-ids",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "splits" / "hemit" / "hemit_test_empty_excluded_ids.txt",
        help="Released empty-tile exclusion list.",
    )
    args = parser.parse_args()

    excluded = set(read_ids(args.excluded_ids))
    selected = []
    for sample_id in read_ids(args.all_test_ids):
        row, col = patch_indices(Path(sample_id).name)
        if row % 2 == 0 and col % 2 == 0 and Path(sample_id).name not in excluded:
            selected.append(Path(sample_id).name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")
    print(f"Wrote {len(selected)} non-overlapping, non-empty HEMIT test IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
