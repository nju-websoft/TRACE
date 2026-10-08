#!/usr/bin/env python3
"""
Convert LLaVA-format training JSON to SWIFT messages format.

LLaVA format (input):
  {"id": "...", "image": "path.png",
   "conversations": [{"from": "user", "value": "<image>\n..."}, {"from": "assistant", "value": "..."}]}

SWIFT format (output):
  {"messages": [{"role": "user", "content": "<image>\n..."}, {"role": "assistant", "content": "..."}],
   "images": ["path.png"]}

Usage:
    python convert_to_swift_format.py --input data.json --output data_swift.json
"""
import argparse
import json


def convert(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    swift_data = []
    for item in data:
        # Build messages
        messages = []
        for turn in item["conversations"]:
            role = "user" if turn["from"] in ("user", "human") else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        # Build images list
        img = item.get("image", "")
        if isinstance(img, list):
            images = img
        elif img:
            images = [img]
        else:
            images = []

        entry = {"messages": messages}
        if images:
            entry["images"] = images

        swift_data.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(swift_data, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(swift_data)} samples: {input_path} → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
