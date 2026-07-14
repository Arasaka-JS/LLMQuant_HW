#!/usr/bin/env python3
"""Convert the downloaded pile-10k parquet shard to AutoRound JSONL input."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Input parquet file containing a text column")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    args = parser.parse_args()

    table = pq.read_table(args.source, columns=["text"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.output.open("w", encoding="utf-8") as output_file:
        for row in table.to_pylist():
            text = row.get("text")
            if not text:
                continue
            output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} samples to {args.output}")


if __name__ == "__main__":
    main()
