#!/usr/bin/env python3
"""Build TEMP ``op114kmath-prd-sharegpt`` JSONL from OpenThoughts-Math.

Default source: ``open-r1/OpenThoughts-114k-math``. Keep rows with
``correct=True``, rename ``problem`` → ``prompt``, normalize ShareGPT turns
from ``from``/``value`` to ``role``/``content``, and set ``domain=math``.

Preserves HF fields used downstream (``messages``, ``solution``, ``system``,
``source``, ``correct``, ``generated_token_count``, ``conversations``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _normalize_conversations(conversations: list) -> list:
    out = []
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        if "role" in msg and "content" in msg:
            role, content = msg["role"], msg["content"]
        else:
            role = msg.get("from", "")
            content = msg.get("value", "")
            if role in ("human", "user"):
                role = "user"
            elif role in ("gpt", "assistant"):
                role = "assistant"
        out.append({"role": role, "content": content})
    return out


def row_to_record(item: dict) -> dict:
    prompt = item.get("prompt")
    if prompt is None:
        prompt = item.get("problem")
    if prompt is None:
        raise KeyError("Need prompt or problem")

    conversations = item.get("conversations")
    if conversations is None:
        raise KeyError("Need conversations")

    record = {
        "system": item.get("system"),
        "conversations": _normalize_conversations(conversations),
        "messages": item.get("messages"),
        "solution": item.get("solution"),
        "prompt": prompt,
        "source": item.get("source"),
        "correct": item.get("correct"),
        "generated_token_count": item.get("generated_token_count"),
        "domain": "math",
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="open-r1/OpenThoughts-114k-math",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading {args.hf_dataset} ({args.split})...")
    ds = load_dataset(args.hf_dataset, split=args.split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for row in tqdm(ds, total=len(ds)):
            if not row.get("correct", False):
                continue
            fout.write(json.dumps(row_to_record(row), ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} rows to {args.output}")


if __name__ == "__main__":
    main()
