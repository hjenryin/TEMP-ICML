#!/usr/bin/env python3
"""Build TEMP ``m23k-prd-sharegpt`` JSONL from the public M23k HF release.

Default source: ``UCSC-VLAA/m23k-tokenized`` (prompt, reasoning,
distilled_answer_string, source).

Assistant content format::

  <|im_start|>think
  {reasoning}
  <|im_start|>answer
  {distilled_answer_string}
  <|im_end|>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def pd_to_sharegpt(item: dict, domain: str = "medical") -> dict:
    prompt = item["prompt"]
    reasoning = item.get("reasoning")
    distilled = item.get("distilled_answer_string")
    if not reasoning or not distilled:
        raise KeyError("Need both reasoning and distilled_answer_string")

    assistant = (
        f"<|im_start|>think\n{reasoning.strip()}\n"
        f"<|im_start|>answer\n{distilled.strip()}\n"
        f"<|im_end|>\n"
    )

    out = {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": prompt,
        "domain": domain,
    }
    if "source" in item and item["source"] is not None:
        out["source"] = item["source"]
    return out


def _iter_rows(args: argparse.Namespace):
    if args.input is not None:
        with args.input.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    ds = load_dataset(args.hf_dataset, split=args.split)
    for row in ds:
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="UCSC-VLAA/m23k-tokenized",
        help="HF dataset id (used when --input is omitted)",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional local JSONL with the same fields as the HF release",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output ShareGPT JSONL")
    parser.add_argument("--domain", type=str, default="medical")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for item in _iter_rows(args):
            fout.write(json.dumps(pd_to_sharegpt(item, args.domain), ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"Processed {n}...")

    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
