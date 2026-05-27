from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import evaluate

from path_config import MARKLLM_ROOT, PPL_MODEL, RESULT_DIR, ppl_result_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PPL for attacked hash-based KGW texts.")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--ppl-model", default=PPL_MODEL)
    parser.add_argument("--model-label", default="opt")
    parser.add_argument("--dataset-label", default="c4")
    parser.add_argument("--hash-types", default="3,4,5")
    parser.add_argument("--hash-methods", default="2,4", help="Fixed variants only: 2 and 4.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def default_input_dir() -> Path:
    if MARKLLM_ROOT:
        return Path(MARKLLM_ROOT) / "test_result"
    return RESULT_DIR / "markllm_attacks"


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


def extract_attack_texts(files: list[Path]) -> tuple[list[str], list[str], list[str], list[str]]:
    original_text: list[str] = []
    word_deletion: list[str] = []
    synonym_substitution: list[str] = []
    context_aware_synonym_substitution: list[str] = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        original_text.extend(data.get("original_text", []))
        word_deletion.extend(data.get("word_deletion", []))
        synonym_substitution.extend(data.get("synonym_substitution", []))
        context_aware_synonym_substitution.extend(data.get("context_aware_synonym_substitution", []))
    return original_text, word_deletion, synonym_substitution, context_aware_synonym_substitution


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
    original, attack1, attack2, attack3 = extract_attack_texts(files)
    return {
        "original_text": compute_ppl(metric, original, args),
        "attack1_text": compute_ppl(metric, attack1, args),
        "attack2_text": compute_ppl(metric, attack2, args),
        "attack3_text": compute_ppl(metric, attack3, args),
        "metadata": {
            "input_files": [str(path) for path in files],
            "ppl_model": args.ppl_model,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    metric = evaluate.load("perplexity", module_type="metric")
    if args.input_file:
        files = [Path(args.input_file)]
        output = Path(args.output_file) if args.output_file else ppl_result_file(args.model_label, args.dataset_label, 0, 2)
        write_json(output, evaluate_files(files, args, metric))
        return

    input_dir = Path(args.input_dir) if args.input_dir else default_input_dir()
    for hash_type in parse_int_list(args.hash_types):
        for hash_method in parse_int_list(args.hash_methods, {2, 4}):
            pattern = f"model_{args.model_label}_dataset_{args.dataset_label}_hash_type_{hash_type}_hash_method_{hash_method}.json"
            files = find_files(input_dir, pattern)
            if not files:
                print(f"No files found for hash_type={hash_type}, hash_method={hash_method}.")
                continue
            output = ppl_result_file(args.model_label, args.dataset_label, hash_type, hash_method)
            write_json(output, evaluate_files(files, args, metric))


if __name__ == "__main__":
    main()
