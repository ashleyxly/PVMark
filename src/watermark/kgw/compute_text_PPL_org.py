from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import evaluate

from path_config import PPL_MODEL, RESULT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PPL for clean KGW/hash-KGW generation outputs.")
    parser.add_argument("--input-dir", default=str(RESULT_DIR / "pile_dataset_test"))
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-dir", default=str(RESULT_DIR / "PPL_org"))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--ppl-model", default=PPL_MODEL)
    parser.add_argument("--model-label", default="opt")
    parser.add_argument("--dataset-label", default="pile")
    parser.add_argument("--hash-types", default="3,4,5")
    parser.add_argument("--hash-methods", default="2,4", help="Fixed variants only: 2 and 4.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_int_list(value: str, allowed: set[int] | None = None) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if allowed is not None:
        unknown = [item for item in result if item not in allowed]
        if unknown:
            raise SystemExit(f"Unsupported values {unknown}; valid values are {sorted(allowed)}")
    return result


def find_files(directory: Path, pattern: str) -> list[Path]:
    matches: list[Path] = []
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(Path(root) / filename)
    return sorted(matches)


def extract_generation_texts(files: list[Path]) -> tuple[list[str], list[str]]:
    with_watermark: list[str] = []
    without_watermark: list[str] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with_watermark.extend(data.get("output_with_watermark", []))
        without_watermark.extend(data.get("output_without_watermark", []))
    return with_watermark, without_watermark


def compute_ppl(metric, texts: list[str], args: argparse.Namespace) -> dict[str, Any]:
    filtered = [text for text in texts if isinstance(text, str) and text.strip()]
    if not filtered:
        return {"perplexities": [], "mean_perplexity": None}
    return metric.compute(
        model_id=args.ppl_model,
        batch_size=args.batch_size,
        add_start_token=True,
        predictions=filtered,
        device=args.device,
        local_files_only=args.local_files_only,
    )


def evaluate_files(files: list[Path], args: argparse.Namespace, metric) -> dict[str, Any]:
    with_watermark, without_watermark = extract_generation_texts(files)
    return {
        "output_with_watermark_ppl": compute_ppl(metric, with_watermark, args),
        "output_without_watermark_ppl": compute_ppl(metric, without_watermark, args),
        "metadata": {
            "input_files": [str(path) for path in files],
            "ppl_model": args.ppl_model,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def output_path(args: argparse.Namespace, hash_type: int, hash_method: int) -> Path:
    if args.output_file:
        return Path(args.output_file)
    return Path(args.output_dir) / (
        f"PPL_model_{args.model_label}_dataset_{args.dataset_label}"
        f"_hash_type_{hash_type}_hash_method_{hash_method}.json"
    )


def main() -> None:
    args = parse_args()
    metric = evaluate.load("perplexity", module_type="metric")
    if args.input_file:
        files = [Path(args.input_file)]
        write_json(output_path(args, 0, 2), evaluate_files(files, args, metric))
        return

    input_dir = Path(args.input_dir)
    for hash_type in parse_int_list(args.hash_types):
        for hash_method in parse_int_list(args.hash_methods, {2, 4}):
            pattern = f"*hash_type_{hash_type}_hash_method_{hash_method}*.json"
            files = find_files(input_dir, pattern)
            if not files:
                print(f"No files found for hash_type={hash_type}, hash_method={hash_method}.")
                continue
            write_json(output_path(args, hash_type, hash_method), evaluate_files(files, args, metric))


if __name__ == "__main__":
    main()
