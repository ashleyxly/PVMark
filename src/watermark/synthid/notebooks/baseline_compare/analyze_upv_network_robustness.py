from __future__ import annotations
import os

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose UPV network robustness outputs.")
    parser.add_argument("--attacks", default="tests/baseline_comparison/upv_gpt2/attacks.json")
    parser.add_argument(
        "--detection",
        default="tests/baseline_comparison/upv_network_detector_gpt2_eli5/attack_detection_network_z1.json",
    )
    parser.add_argument("--model-name-or-path", default=os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
    parser.add_argument(
        "--output",
        default="tests/baseline_comparison/upv_network_detector_gpt2_eli5/robustness_diagnostics_network_z1.json",
    )
    return parser.parse_args()


def levenshtein_ratio(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (0 if x == y else 1)))
        prev = curr
    return prev[-1] / max(len(a), len(b), 1)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    detected_rows = [r for r in rows if r["detected"] is not None]
    positive_rows = [r for r in detected_rows if r["detected"]]
    nonempty_changed = [r for r in rows if (not r["empty"]) and (not r["unchanged"])]
    positive_nonempty_changed = [r for r in nonempty_changed if r["detected"]]

    bins = [
        ("edit_lt_0p05", lambda r: r["edit_ratio"] < 0.05),
        ("edit_0p05_0p2", lambda r: 0.05 <= r["edit_ratio"] < 0.2),
        ("edit_0p2_0p4", lambda r: 0.2 <= r["edit_ratio"] < 0.4),
        ("edit_ge_0p4", lambda r: r["edit_ratio"] >= 0.4),
    ]
    by_bin = {}
    for name, pred in bins:
        subset = [r for r in rows if pred(r)]
        by_bin[name] = {
            "count": len(subset),
            "positive_count": sum(1 for r in subset if r["detected"]),
            "positive_rate": (
                sum(1 for r in subset if r["detected"]) / len(subset) if subset else None
            ),
        }

    return {
        "count": len(rows),
        "empty_count": sum(1 for r in rows if r["empty"]),
        "unchanged_count": sum(1 for r in rows if r["unchanged"]),
        "mean_orig_len": mean_or_none([r["orig_len"] for r in rows]),
        "mean_attacked_len": mean_or_none([r["attacked_len"] for r in rows]),
        "mean_len_ratio": mean_or_none([r["len_ratio"] for r in rows]),
        "median_len_ratio": median_or_none([r["len_ratio"] for r in rows]),
        "mean_edit_ratio": mean_or_none([r["edit_ratio"] for r in rows]),
        "median_edit_ratio": median_or_none([r["edit_ratio"] for r in rows]),
        "positive_count": len(positive_rows),
        "positive_rate": len(positive_rows) / len(detected_rows) if detected_rows else None,
        "nonempty_changed_count": len(nonempty_changed),
        "nonempty_changed_positive_count": len(positive_nonempty_changed),
        "nonempty_changed_positive_rate": (
            len(positive_nonempty_changed) / len(nonempty_changed) if nonempty_changed else None
        ),
        "by_edit_ratio_bin": by_bin,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    with open(args.attacks, "r", encoding="utf-8") as f:
        attacks = json.load(f)["records"]
    with open(args.detection, "r", encoding="utf-8") as f:
        detection = json.load(f)["records"]

    det_by_key = {
        (r["sample_id"], bool(r["watermarked"]), r["attack"]): r for r in detection
    }
    rows: list[dict[str, Any]] = []
    for record in attacks:
        original_ids = tokenizer(record.get("original_text", ""), add_special_tokens=False)[
            "input_ids"
        ]
        for attack_name, text in record.get("attacks", {}).items():
            attacked_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            det = det_by_key.get((record["sample_id"], bool(record["watermarked"]), attack_name))
            rows.append(
                {
                    "sample_id": record["sample_id"],
                    "watermarked": bool(record["watermarked"]),
                    "attack": attack_name,
                    "detected": det.get("detected") if det else None,
                    "score": det.get("score") if det else None,
                    "empty": text == "",
                    "unchanged": original_ids == attacked_ids,
                    "orig_len": len(original_ids),
                    "attacked_len": len(attacked_ids),
                    "len_ratio": len(attacked_ids) / max(len(original_ids), 1),
                    "edit_ratio": levenshtein_ratio(original_ids, attacked_ids),
                }
            )

    summary: dict[str, Any] = {}
    for attack_name in sorted({r["attack"] for r in rows}):
        summary[attack_name] = {}
        attack_rows = [r for r in rows if r["attack"] == attack_name]
        summary[attack_name]["all"] = summarize(attack_rows)
        summary[attack_name]["wm"] = summarize([r for r in attack_rows if r["watermarked"]])
        summary[attack_name]["uwm"] = summarize([r for r in attack_rows if not r["watermarked"]])

    out = {
        "metadata": {
            "attacks": args.attacks,
            "detection": args.detection,
            "model_name_or_path": args.model_name_or_path,
        },
        "summary": summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
