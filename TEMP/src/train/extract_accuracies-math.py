#!/usr/bin/env python3
"""
Extract accuracy results from JSON evaluation files and output tab-separated text files.

This script recursively finds JSON files under an OpenThoughts-math eval directory,
parses the accuracy results, and creates tab-separated text files with columns:
AMC23, AIME24, AIME25, MATH500 (plus any extras).

The output file is created in the same directory as the input JSON file with a .txt extension.

Usage:
    python3 TEMP/src/train/extract_accuracies-math.py [eval_dir]

Output Format:
    AMC23	AIME24	AIME25	MATH500
    0.123	0.742	0.434	0.276
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


def extract_accuracies(json_path: Path) -> Optional[Dict[str, float]]:
    """
    Extract accuracy values from a JSON evaluation file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary mapping dataset names to accuracy values, or None if parsing fails
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        if 'results' not in data:
            return None

        results = data['results']
        accuracies = {}

        # Extract accuracy for each dataset
        for dataset_name, dataset_results in results.items():
            if isinstance(dataset_results, dict):
                if "accuracy" in dataset_results:
                    accuracies[dataset_name] = dataset_results["accuracy"]
                elif "accuracy_avg" in dataset_results:
                    accuracies[dataset_name] = dataset_results["accuracy_avg"]

        return accuracies if accuracies else None

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing {json_path}: {e}")
        return None


def create_txt_output(json_path: Path, accuracies: Dict[str, float]) -> None:
    """
    Create a tab-separated text file with accuracy results.

    Args:
        json_path: Path to the original JSON file
        accuracies: Dictionary of dataset names to accuracy values
    """
    # Paper math tasks (AMC23, AIME24, AIME25, MATH500)
    column_order = [
        'AMC23',
        'AIME24',
        'AIME25',
        'MATH500'
    ]

    # Build header and values lists
    headers = []
    values = []

    for col in column_order:
        if col in accuracies:
            headers.append(col)
            values.append(str(accuracies[col]))

    # Add any remaining datasets not in the predefined order
    for dataset_name in sorted(accuracies.keys()):
        if dataset_name not in column_order:
            headers.append(dataset_name)
            values.append(str(accuracies[dataset_name]))

    # Create output file path (same directory, same name but .txt extension)
    txt_path = json_path.with_suffix('.txt')

    # Write tab-separated output
    with open(txt_path, 'w') as f:
        f.write('\t'.join(headers) + '\n')
        f.write('\t'.join(values) + '\n')

    print(f"Created: {txt_path}")


def main():
    """Main function to process JSON files under a given eval directory."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract math eval accuracies from evalchemy JSON outputs")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=None,
        help="Directory to scan for JSON results (default: $TEMP_OUTPUT_ROOT/eval/openthoughts-math)",
    )
    args = parser.parse_args()

    if args.base_dir is not None:
        base_dir = Path(args.base_dir)
    else:
        repo_root = Path(__file__).resolve().parents[3]
        output_root = Path(os.environ.get("TEMP_OUTPUT_ROOT", repo_root / "outputs"))
        base_dir = output_root / "eval" / "openthoughts-math"

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    # Find all JSON files recursively
    json_files = list(base_dir.rglob('*.json'))

    if not json_files:
        print(f"No JSON files found in {base_dir}")
        return

    print(f"Found {len(json_files)} JSON file(s) under {base_dir}")

    processed_count = 0
    skipped_count = 0

    for json_path in json_files:
        # skip if the txt file already exists
        txt_path = json_path.with_suffix('.txt')
        if txt_path.exists():
            skipped_count += 1
            continue

        print(f"\nProcessing: {json_path}")

        accuracies = extract_accuracies(json_path)

        if accuracies:
            create_txt_output(json_path, accuracies)
            processed_count += 1
        else:
            print(f"Skipped: No accuracy data found in {json_path}")
            skipped_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {processed_count} file(s) processed, {skipped_count} file(s) skipped")


if __name__ == '__main__':
    main()
