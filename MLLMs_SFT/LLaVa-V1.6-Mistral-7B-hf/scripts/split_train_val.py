#!/usr/bin/env python3
"""Split an SFT training JSON into train / val files by ratio."""
import json
import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the original training JSON")
    parser.add_argument("--train-output", required=True, help="Path to the output train JSON")
    parser.add_argument("--val-output", required=True, help="Path to the output val JSON")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    val_size = max(1, int(len(data) * args.val_ratio))
    val_indices = set(indices[:val_size])

    train_data = [data[i] for i in range(len(data)) if i not in val_indices]
    val_data = [data[i] for i in range(len(data)) if i in val_indices]

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(args.val_output, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"Original data: {len(data)} samples")
    print(f"Train set:     {len(train_data)} samples -> {args.train_output}")
    print(f"Val set:       {len(val_data)} samples -> {args.val_output}")


if __name__ == "__main__":
    main()
