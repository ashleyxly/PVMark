from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import summarize_numbers, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize UPV network detector robustness, detector scores, and PPL."
    )
    parser.add_argument("--normal-detection", default=None)
    parser.add_argument("--normal-ppl", default=None)
    parser.add_argument("--attack-detection", required=True)
    parser.add_argument("--attack-ppl", default=None)
    parser.add_argument("--diagnostics", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def load(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rate(records: list[dict[str, Any]], watermarked: bool) -> dict[str, Any]:
    subset = [r for r in records if bool(r.get("watermarked")) is watermarked]
    valid = [r for r in subset if r.get("score") is not None and r.get("detected") is not None]
    positives = [r for r in valid if bool(r.get("detected"))]
    return {
        "count": len(subset),
        "valid_count": len(valid),
        "error_count": len(subset) - len(valid),
        "positive_count": len(positives),
        "positive_rate": len(positives) / len(valid) if valid else None,
    }


def index_ppl(records: list[dict[str, Any]]) -> dict[tuple[str, bool], list[float]]:
    by_key: dict[tuple[str, bool], list[float]] = defaultdict(list)
    for record in records:
        ppl = record.get("ppl")
        if ppl is None:
            continue
        attack = str(record.get("attack") or "normal")
        by_key[(attack, bool(record.get("watermarked")))].append(float(ppl))
    return by_key


def summarize_subset(
    records: list[dict[str, Any]],
    ppl_by_key: dict[tuple[str, bool], list[float]],
    attack: str,
) -> dict[str, Any]:
    wm = [r for r in records if bool(r.get("watermarked"))]
    uwm = [r for r in records if not bool(r.get("watermarked"))]
    return {
        "wm_detection": rate(records, True),
        "uwm_false_positive": rate(records, False),
        "wm_score": summarize_numbers([r.get("score") for r in wm]),
        "uwm_score": summarize_numbers([r.get("score") for r in uwm]),
        "wm_ppl": summarize_numbers(ppl_by_key.get((attack, True), [])),
        "uwm_ppl": summarize_numbers(ppl_by_key.get((attack, False), [])),
    }


def main() -> None:
    args = parse_args()
    normal_detection = load(args.normal_detection)
    normal_ppl = load(args.normal_ppl)
    attack_detection = load(args.attack_detection)
    attack_ppl = load(args.attack_ppl)
    diagnostics = load(args.diagnostics)

    normal_ppl_by_key = index_ppl(normal_ppl.get("records", []))
    attack_ppl_by_key = index_ppl(attack_ppl.get("records", []))

    output: dict[str, Any] = {
        "metadata": {
            "normal_detection": args.normal_detection,
            "normal_ppl": args.normal_ppl,
            "attack_detection": args.attack_detection,
            "attack_ppl": args.attack_ppl,
            "diagnostics": args.diagnostics,
            "score_definition": (
                "UPV network detector classifier probability in [0, 1]; "
                "threshold > 0.5 is treated as watermarked."
            ),
            "ppl_evaluator": attack_ppl.get("metadata", {}).get(
                "evaluator", normal_ppl.get("metadata", {}).get("evaluator")
            ),
        },
        "normal": {},
        "robustness": {},
    }

    if normal_detection:
        normal_records = normal_detection.get("records", [])
        output["normal"] = summarize_subset(normal_records, normal_ppl_by_key, "normal")

    attack_records = attack_detection.get("records", [])
    by_attack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in attack_records:
        attack = record.get("attack")
        if attack is not None:
            by_attack[str(attack)].append(record)

    rows: list[dict[str, Any]] = []
    diag_summary = diagnostics.get("summary", {}) if diagnostics else {}
    for attack, records in sorted(by_attack.items()):
        summary = summarize_subset(records, attack_ppl_by_key, attack)
        if attack in diag_summary:
            summary["diagnostics"] = {
                "wm_mean_edit_ratio": diag_summary[attack].get("wm", {}).get("mean_edit_ratio"),
                "wm_median_edit_ratio": diag_summary[attack].get("wm", {}).get("median_edit_ratio"),
                "wm_median_len_ratio": diag_summary[attack].get("wm", {}).get("median_len_ratio"),
            }
        output["robustness"][attack] = summary
        rows.append(
            {
                "attack": attack,
                "wm_detect_rate": summary["wm_detection"]["positive_rate"],
                "uwm_false_positive_rate": summary["uwm_false_positive"]["positive_rate"],
                "wm_score_mean": summary["wm_score"]["mean"],
                "wm_score_median": summary["wm_score"]["median"],
                "uwm_score_mean": summary["uwm_score"]["mean"],
                "uwm_score_median": summary["uwm_score"]["median"],
                "wm_ppl_mean": summary["wm_ppl"]["mean"],
                "wm_ppl_median": summary["wm_ppl"]["median"],
                "uwm_ppl_mean": summary["uwm_ppl"]["mean"],
                "uwm_ppl_median": summary["uwm_ppl"]["median"],
                "wm_mean_edit_ratio": summary.get("diagnostics", {}).get("wm_mean_edit_ratio"),
                "wm_median_edit_ratio": summary.get("diagnostics", {}).get("wm_median_edit_ratio"),
            }
        )

    write_json(args.output_json, output)
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
