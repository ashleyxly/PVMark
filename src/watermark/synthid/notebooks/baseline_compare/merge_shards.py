from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from common import read_json, sort_records, write_json
from summarize import rate as detection_rate
from summarize_robustness import rate as robustness_rate
from common import summarize_numbers


PIPELINE_FILES = [
    "generations.json",
    "detection.json",
    "attacks.json",
    "attack_detection.json",
    "ppl.json",
    "attack_ppl.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge baseline shard directories.")
    parser.add_argument("--shard-root", required=True, help="Directory containing shard_00, shard_01, ...")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    return parser.parse_args()


def key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("sample_id"),
        bool(record.get("watermarked")),
        record.get("attack") or "",
    )


def merge_payloads(paths: list[Path]) -> dict[str, Any] | None:
    metadata: dict[str, Any] | None = None
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    sources: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        payload = read_json(path)
        if metadata is None:
            metadata = dict(payload.get("metadata", {}))
        sources.append(str(path))
        for record in payload.get("records", []):
            merged[key(record)] = record
    if metadata is None:
        return None
    metadata["merged_from"] = sources
    metadata["num_records"] = len(merged)
    return {"metadata": metadata, "records": sort_records(list(merged.values()))}


def summarize_detection(generation_path: Path, detection_path: Path, output_path: Path) -> None:
    generation = read_json(generation_path)
    detection = read_json(detection_path)
    gen_records = generation.get("records", [])
    det_records = detection.get("records", [])
    summary = {
        "method": generation.get("metadata", {}).get("method"),
        "model_name_or_path": generation.get("metadata", {}).get("model_name_or_path"),
        "num_generation_records": len(gen_records),
        "num_detection_records": len(det_records),
        "effectiveness": {
            "wm_detection": detection_rate(det_records, True),
            "uwm_false_positive": detection_rate(det_records, False),
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
        "score": summarize_numbers(
            [r.get("score") for r in det_records if r.get("score") is not None]
        ),
    }
    write_json(
        output_path,
        {
            "metadata": {"generation": str(generation_path), "detection": str(detection_path)},
            "summary": summary,
        },
    )


def summarize_robustness(attack_detection_path: Path, attack_ppl_path: Path, output_json: Path, output_csv: Path) -> None:
    detection = read_json(attack_detection_path)
    det_records = detection.get("records", [])
    ppl_records = read_json(attack_ppl_path).get("records", []) if attack_ppl_path.exists() else []
    ppl_by_attack: dict[tuple[str, bool], list[float]] = {}
    for record in ppl_records:
        attack = record.get("attack")
        ppl = record.get("ppl")
        if attack is None or ppl is None:
            continue
        ppl_by_attack.setdefault((str(attack), bool(record.get("watermarked"))), []).append(float(ppl))

    by_attack: dict[str, list[dict[str, Any]]] = {}
    for record in det_records:
        attack = record.get("attack")
        if attack is None:
            continue
        by_attack.setdefault(str(attack), []).append(record)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for attack, records in sorted(by_attack.items()):
        wm_rate = robustness_rate(records, True)
        uwm_rate = robustness_rate(records, False)
        wm_ppl = summarize_numbers(ppl_by_attack.get((attack, True), []))
        uwm_ppl = summarize_numbers(ppl_by_attack.get((attack, False), []))
        summary[attack] = {
            "wm_detection": wm_rate,
            "uwm_false_positive": uwm_rate,
            "detection_time_sec": summarize_numbers([r.get("detection_time_sec") for r in records]),
            "wm_ppl": wm_ppl,
            "uwm_ppl": uwm_ppl,
        }
        rows.append(
            {
                "attack": attack,
                "wm_count": wm_rate["count"],
                "wm_valid_count": wm_rate["valid_count"],
                "wm_error_count": wm_rate.get("error_count", 0),
                "wm_detect_rate": wm_rate["positive_rate"],
                "uwm_count": uwm_rate["count"],
                "uwm_valid_count": uwm_rate["valid_count"],
                "uwm_error_count": uwm_rate.get("error_count", 0),
                "uwm_false_positive_rate": uwm_rate["positive_rate"],
                "mean_detection_time_sec": summary[attack]["detection_time_sec"]["mean"],
                "wm_mean_ppl": wm_ppl["mean"],
                "uwm_mean_ppl": uwm_ppl["mean"],
            }
        )

    write_json(
        output_json,
        {
            "metadata": {
                "detection": str(attack_detection_path),
                "ppl": str(attack_ppl_path) if attack_ppl_path.exists() else None,
                "method": detection.get("metadata", {}).get("method"),
            },
            "summary": summary,
        },
    )
    if rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = [shard_root / f"shard_{idx:02d}" for idx in range(args.num_shards)]

    for filename in PIPELINE_FILES:
        payload = merge_payloads([shard_dir / filename for shard_dir in shard_dirs])
        if payload is not None:
            write_json(output_dir / filename, payload)

    if (output_dir / "generations.json").exists() and (output_dir / "detection.json").exists():
        summarize_detection(output_dir / "generations.json", output_dir / "detection.json", output_dir / "summary.json")
    if (output_dir / "attack_detection.json").exists():
        summarize_robustness(
            output_dir / "attack_detection.json",
            output_dir / "attack_ppl.json",
            output_dir / "robustness_summary.json",
            output_dir / "robustness_summary.csv",
        )


if __name__ == "__main__":
    main()
