#!/usr/bin/env python3
"""Extract medical eval accuracies from TEMP medical eval outputs.

Scans ``$TEMP_OUTPUT_ROOT/eval/m23k-llama-factory*`` for ``metrics.json``
(written by ``TEMP/src/eval/score.py``) and writes a sibling ``.txt`` with a
tab-separated header row and accuracy row.

For OpenThoughts-Math / evalchemy JSON, use ``extract_accuracies-math.py``.

Usage:
    python3 TEMP/src/train/extract_accuracies.py [eval_dir]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

# Paper medical suite order (10 sets; average reported in the paper).
COLUMN_ORDER = [
    "MedMCQA_validation",
    "MedQA_USLME_test",
    "PubMedQA_test",
    "MMLU-Pro_Medical_test",
    "GPQA_Medical_test",
    "Lancet",
    "MedBullets_op4",
    "MedBullets_op5",
    "MedXpertQA",
    "NEJM",
]


def extract_accuracies(metrics_path: Path) -> Optional[Dict[str, float]]:
    """Parse ``metrics.json``: ``{source: {accuracy: float, ...}, ...}``."""
    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error processing {metrics_path}: {e}")
        return None

    if not isinstance(data, dict):
        return None

    accuracies: Dict[str, float] = {}
    for source, stats in data.items():
        if isinstance(stats, dict) and "accuracy" in stats:
            accuracies[source] = float(stats["accuracy"])
        elif isinstance(stats, (int, float)):
            accuracies[source] = float(stats)

    return accuracies or None


def create_txt_output(metrics_path: Path, accuracies: Dict[str, float]) -> None:
    headers = []
    values = []
    for col in COLUMN_ORDER:
        if col in accuracies:
            headers.append(col)
            values.append(str(accuracies[col]))
    for name in sorted(accuracies.keys()):
        if name not in COLUMN_ORDER:
            headers.append(name)
            values.append(str(accuracies[name]))

    txt_path = metrics_path.with_suffix(".txt")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        f.write("\t".join(values) + "\n")
    print(f"Created: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract medical eval accuracies")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=None,
        help="Directory to scan for metrics.json "
        "(default: $TEMP_OUTPUT_ROOT/eval/m23k-llama-factory*)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_root = Path(os.environ.get("TEMP_OUTPUT_ROOT", repo_root / "outputs"))

    if args.base_dir is not None:
        search_roots = [Path(args.base_dir)]
    else:
        search_roots = sorted(output_root.glob("eval/m23k-llama-factory*"))
        if not search_roots:
            search_roots = [output_root / "eval" / "m23k-llama-factory"]

    json_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            print(f"Warning: directory {root} does not exist")
            continue
        json_files.extend(root.rglob("metrics.json"))

    if not json_files:
        print("No metrics.json files found")
        return

    print(f"Found {len(json_files)} metrics.json file(s)")
    processed = skipped = 0
    for metrics_path in json_files:
        txt_path = metrics_path.with_suffix(".txt")
        if txt_path.exists():
            skipped += 1
            continue
        print(f"\nProcessing: {metrics_path}")
        accuracies = extract_accuracies(metrics_path)
        if accuracies:
            create_txt_output(metrics_path, accuracies)
            processed += 1
        else:
            print(f"Skipped: No accuracy data in {metrics_path}")
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: {processed} file(s) processed, {skipped} file(s) skipped")


if __name__ == "__main__":
    main()
