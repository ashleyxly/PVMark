from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import summarize_numbers, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize attacked detection outputs by attack.")
    parser.add_argument("--detection", required=True, help="Attack detection JSON.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--ppl", default=None, help="Optional attacked PPL JSON.")
    return parser.parse_args()


def load(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rate(records: list[dict[str, Any]], watermarked: bool) -> dict[str, Any]:
    subset = [r for r in records if bool(r.get("watermarked")) is watermarked]
    # For robustness, a detector failure on an attacked text should count as a
    # missed detection rather than silently shrinking the denominator.
    valid = subset
    positives = [r for r in valid if bool(r.get("detected"))]
    return {
        "count": len(subset),
        "valid_count": len(valid),
        "error_count": len([r for r in subset if r.get("error")]),
        "positive_count": len(positives),
        "positive_rate": (len(positives) / len(valid)) if valid else None,
    }


def build_ppl_index(path: str | None) -> dict[tuple[str, bool], list[float]]:
    if not path:
        return {}
    payload = load(path)
    index: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for record in payload.get("records", []):
        ppl = record.get("ppl")
        attack = record.get("attack")
        if ppl is None or attack is None:
            continue
        index[(str(attack), bool(record.get("watermarked")))].append(float(ppl))
    return index


def main() -> None:
    args = parse_args()
    payload = load(args.detection)
    records = payload.get("records", [])
    ppl_index = build_ppl_index(args.ppl)

    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        attack = record.get("attack")
        if attack is None:
            continue
        by_attack[str(attack)].append(record)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for attack, attack_records in sorted(by_attack.items()):
        wm_rate = rate(attack_records, True)
        uwm_rate = rate(attack_records, False)
        wm_ppl = summarize_numbers(ppl_index.get((attack, True), []))
        uwm_ppl = summarize_numbers(ppl_index.get((attack, False), []))
        attack_summary = {
            "wm_detection": wm_rate,
            "uwm_false_positive": uwm_rate,
            "detection_time_sec": summarize_numbers(
                [r.get("detection_time_sec") for r in attack_records]
            ),
            "wm_ppl": wm_ppl,
            "uwm_ppl": uwm_ppl,
        }
        summary[attack] = attack_summary
        rows.append(
            {
                "attack": attack,
                "wm_count": wm_rate["count"],
                "wm_valid_count": wm_rate["valid_count"],
                "wm_detect_rate": wm_rate["positive_rate"],
                "uwm_count": uwm_rate["count"],
                "uwm_valid_count": uwm_rate["valid_count"],
                "uwm_false_positive_rate": uwm_rate["positive_rate"],
                "mean_detection_time_sec": attack_summary["detection_time_sec"]["mean"],
                "wm_mean_ppl": wm_ppl["mean"],
                "uwm_mean_ppl": uwm_ppl["mean"],
            }
        )

    write_json(
        args.output_json,
        {
            "metadata": {
                "detection": args.detection,
                "ppl": args.ppl,
                "method": payload.get("metadata", {}).get("method"),
            },
            "summary": summary,
        },
    )

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
