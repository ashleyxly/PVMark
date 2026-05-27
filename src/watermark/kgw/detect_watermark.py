from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from path_config import GEN_MODEL, MARKLLM_ROOT, RESULT_DIR, detection_result_file
from watermark_processor import WatermarkDetector

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect watermarks in hash-based KGW attack outputs.")
    parser.add_argument("--model_name_or_path", default=GEN_MODEL)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--model-label", default="opt")
    parser.add_argument("--dataset-label", default="c4")
    parser.add_argument("--hash-types", default="3,4,5")
    parser.add_argument("--hash-methods", default="2,4", help="Fixed variants only: 2 and 4.")
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--seeding_scheme", default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--normalizers", default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    parser.add_argument(
        "--hash_type",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="Used only with --input-file.",
    )
    parser.add_argument(
        "--hash_method",
        type=int,
        default=2,
        choices=[2, 4],
        help="Used only with --input-file. Fixed variants only.",
    )
    parser.add_argument("--skip_model_load", type=str2bool, default=False)
    return parser.parse_args()


def parse_int_list(value: str, allowed: set[int] | None = None) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if allowed is not None:
        unknown = [item for item in result if item not in allowed]
        if unknown:
            raise SystemExit(f"Unsupported values {unknown}; valid values are {sorted(allowed)}")
    return result


def default_input_dir() -> Path:
    if MARKLLM_ROOT:
        return Path(MARKLLM_ROOT) / "test_result"
    return RESULT_DIR / "markllm_attacks"


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


def load_model(args: argparse.Namespace):
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return tokenizer, device


def make_detector(args: argparse.Namespace, tokenizer, device: str, hash_type: int, hash_method: int) -> WatermarkDetector:
    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
        hash_type=hash_type,
        hash_method=hash_method,
    )


def detect_batch(
    texts: list[str],
    args: argparse.Namespace,
    device: str,
    tokenizer,
    detector: WatermarkDetector,
) -> tuple[list[Any], float, int, int]:
    z_scores: list[Any] = []
    total_z = 0.0
    detected = 0
    total = 0
    for text in tqdm(texts):
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=args.prompt_max_length,
        ).to(device)
        decoded = tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=True)[0]
        if len(decoded) <= detector.min_prefix_len:
            z_scores.append(None)
            continue
        score = detector.detect(decoded)
        z_value = float(score["z_score"])
        z_scores.append(z_value)
        total_z += z_value
        total += 1
        if z_value > args.detection_z_threshold:
            detected += 1
    return z_scores, total_z, detected, total


def evaluate_file(
    files: list[Path],
    args: argparse.Namespace,
    tokenizer,
    device: str,
    hash_type: int,
    hash_method: int,
) -> dict[str, Any]:
    detector = make_detector(args, tokenizer, device, hash_type, hash_method)
    original, attack1, attack2, attack3 = extract_attack_texts(files)
    attack1_scores, attack1_total_z, attack1_count, attack1_total = detect_batch(attack1, args, device, tokenizer, detector)
    attack2_scores, attack2_total_z, attack2_count, attack2_total = detect_batch(attack2, args, device, tokenizer, detector)
    attack3_scores, attack3_total_z, attack3_count, attack3_total = detect_batch(attack3, args, device, tokenizer, detector)
    original_scores, original_total_z, original_count, original_total = detect_batch(original, args, device, tokenizer, detector)
    return {
        "org_total_z_scores": original_total_z,
        "org_count_over_threshold": original_count,
        "org_total_count": original_total,
        "attack1_total_z_scores": attack1_total_z,
        "attack1_count_over_threshold": attack1_count,
        "attack1_total_count": attack1_total,
        "attack2_total_z_scores": attack2_total_z,
        "attack2_count_over_threshold": attack2_count,
        "attack2_total_count": attack2_total,
        "attack3_total_z_scores": attack3_total_z,
        "attack3_count_over_threshold": attack3_count,
        "attack3_total_count": attack3_total,
        "original_text_z_score": original_scores,
        "attack1_text_z_score": attack1_scores,
        "attack2_text_z_score": attack2_scores,
        "attack3_text_z_score": attack3_scores,
        "metadata": {
            "hash_type": hash_type,
            "hash_method": hash_method,
            "input_files": [str(path) for path in files],
            "threshold": args.detection_z_threshold,
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.normalizers = [item for item in args.normalizers.split(",") if item]
    if args.skip_model_load:
        print("Skipping tokenizer load.")
        return

    tokenizer, device = load_model(args)
    input_dir = Path(args.input_dir) if args.input_dir else default_input_dir()

    if args.input_file:
        files = [Path(args.input_file)]
        output = Path(args.output_file) if args.output_file else detection_result_file(args.model_label, args.dataset_label, args.hash_type, args.hash_method)
        write_json(output, evaluate_file(files, args, tokenizer, device, args.hash_type, args.hash_method))
        return

    for hash_type in parse_int_list(args.hash_types):
        for hash_method in parse_int_list(args.hash_methods, {2, 4}):
            pattern = f"model_{args.model_label}_dataset_{args.dataset_label}_hash_type_{hash_type}_hash_method_{hash_method}.json"
            files = find_files(input_dir, pattern)
            if not files:
                print(f"No files found for hash_type={hash_type}, hash_method={hash_method}.")
                continue
            output = detection_result_file(args.model_label, args.dataset_label, hash_type, hash_method)
            write_json(output, evaluate_file(files, args, tokenizer, device, hash_type, hash_method))


if __name__ == "__main__":
    main()
