from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize PDW WET/WDT from generation and detection records. "
            "PDW writes wall-clock generation/detection timings during the main run; "
            "this script makes the length-aware efficiency report explicit."
        )
    )
    parser.add_argument(
        "--generations",
        default="tests/baseline_comparison/pdw_gpt2/generations.json",
        help="PDW generations.json produced by pdw_experiment.py.",
    )
    parser.add_argument(
        "--detection",
        default="tests/baseline_comparison/pdw_gpt2/detection.json",
        help="PDW detection.json produced by pdw_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default="tests/baseline_comparison/pdw_gpt2/pdw_efficiency_from_records.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p90": None,
        }
    xs = sorted(float(v) for v in values)
    p90_index = int(0.9 * (len(xs) - 1))
    return {
        "count": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": xs[0],
        "max": xs[-1],
        "p90": xs[p90_index],
    }


def grouped_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {"watermarked": [], "unwatermarked": []}
    for record in records:
        key = "watermarked" if bool(record.get("watermarked")) else "unwatermarked"
        groups[key].append(record)
    return groups


def generation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    for record in records:
        metadata = record.get("method_metadata") or {}
        if metadata.get("generation_error"):
            continue
        if not finite(record.get("generation_time_sec")):
            continue
        if not finite(record.get("completion_token_count")):
            continue
        if float(record["completion_token_count"]) <= 0:
            continue
        valid.append(record)

    times = [float(record["generation_time_sec"]) for record in valid]
    token_counts = [float(record["completion_token_count"]) for record in valid]
    sec_per_token = [
        float(record["generation_time_sec"]) / float(record["completion_token_count"])
        for record in valid
    ]
    return {
        "num_records": len(records),
        "num_valid_records": len(valid),
        "generation_time_sec": summarize(times),
        "completion_token_count": summarize(token_counts),
        "sec_per_token": summarize(sec_per_token),
        "ms_per_token": summarize([1000.0 * x for x in sec_per_token]),
    }


def detection_summary(
    records: list[dict[str, Any]],
    token_counts_by_key: dict[tuple[int, bool], float],
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    sec_per_token: list[float] = []
    for record in records:
        if record.get("error"):
            continue
        if not finite(record.get("detection_time_sec")):
            continue
        valid.append(record)
        sample_id = record.get("sample_id")
        key = (int(sample_id), bool(record.get("watermarked"))) if sample_id is not None else None
        token_count = token_counts_by_key.get(key) if key is not None else None
        if token_count and token_count > 0:
            sec_per_token.append(float(record["detection_time_sec"]) / token_count)

    times = [float(record["detection_time_sec"]) for record in valid]
    return {
        "num_records": len(records),
        "num_valid_records": len(valid),
        "detection_time_sec": summarize(times),
        "sec_per_token": summarize(sec_per_token),
        "ms_per_token": summarize([1000.0 * x for x in sec_per_token]),
    }


def main() -> None:
    args = parse_args()
    generation_payload = read_json(args.generations)
    detection_payload = read_json(args.detection)
    generation_records = generation_payload.get("records", [])
    detection_records = detection_payload.get("records", [])

    token_counts_by_key: dict[tuple[int, bool], float] = {}
    for record in generation_records:
        if record.get("sample_id") is None or not finite(record.get("completion_token_count")):
            continue
        token_counts_by_key[(int(record["sample_id"]), bool(record.get("watermarked")))] = float(
            record["completion_token_count"]
        )

    gen_groups = grouped_records(generation_records)
    det_groups = grouped_records(detection_records)
    output = {
        "metadata": {
            "generations": str(args.generations),
            "detection": str(args.detection),
            "definition": (
                "PDW WET is the recorded asymmetric generation wall-clock time. "
                "PDW WDT is the recorded public verification wall-clock time. "
                "Because PDW watermarked outputs are variable-length, report both "
                "per-sample and per-token summaries; for a 200-token comparison, "
                "use the unwatermarked detection group."
            ),
            "generation_metadata": generation_payload.get("metadata", {}),
            "detection_metadata": detection_payload.get("metadata", {}),
        },
        "wet": {
            "watermarked": generation_summary(gen_groups["watermarked"]),
            "unwatermarked_plain": generation_summary(gen_groups["unwatermarked"]),
        },
        "wdt": {
            "watermarked": detection_summary(det_groups["watermarked"], token_counts_by_key),
            "unwatermarked_plain": detection_summary(
                det_groups["unwatermarked"], token_counts_by_key
            ),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
