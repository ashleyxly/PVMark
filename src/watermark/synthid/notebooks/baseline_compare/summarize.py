from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from common import summarize_numbers, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize baseline detection and timing outputs.")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--detection", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rate(records: list[dict[str, Any]], expected_watermarked: bool) -> dict[str, Any]:
    subset = [r for r in records if bool(r.get("watermarked")) is expected_watermarked]
    valid = [r for r in subset if r.get("detected") is not None]
    positives = [r for r in valid if bool(r.get("detected"))]
    return {
        "count": len(subset),
        "valid_count": len(valid),
        "positive_count": len(positives),
        "positive_rate": (len(positives) / len(valid)) if valid else None,
    }


def main() -> None:
    args = parse_args()
    gen = load(args.generation)
    det = load(args.detection)
    gen_records = gen["records"]
    det_records = det["records"]

    summary = {
        "method": gen["metadata"].get("method"),
        "model_name_or_path": gen["metadata"].get("model_name_or_path"),
        "num_generation_records": len(gen_records),
        "num_detection_records": len(det_records),
        "effectiveness": {
            "wm_detection": rate(det_records, True),
            "uwm_false_positive": rate(det_records, False),
        },
        "efficiency": {
            "generation_time_sec": summarize_numbers(
                [r.get("generation_time_sec") for r in gen_records]
            ),
            "detection_time_sec": summarize_numbers(
                [r.get("detection_time_sec") for r in det_records]
            ),
            "completion_token_count": summarize_numbers(
                [r.get("completion_token_count") for r in gen_records]
            ),
        },
        "score": summarize_numbers([r.get("score") for r in det_records if r.get("score") is not None]),
    }

    write_json(args.output, {"metadata": {"generation": args.generation, "detection": args.detection}, "summary": summary})


if __name__ == "__main__":
    main()

