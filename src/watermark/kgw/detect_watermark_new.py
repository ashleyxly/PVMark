from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from path_config import GEN_MODEL, RESULT_DIR
from watermark_processor_org_scheme import WatermarkDetector

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
    parser = argparse.ArgumentParser(description="Detect original KGW attack outputs.")
    parser.add_argument("--model_name_or_path", default=GEN_MODEL)
    parser.add_argument(
        "--input-file",
        default=str(RESULT_DIR / "Org_scheme" / "Normal" / "Attack" / "opt_c4_attack.json"),
    )
    parser.add_argument(
        "--output-file",
        default=str(RESULT_DIR / "Org_scheme" / "Normal" / "Attack" / "Detect" / "Detect_opt_c4_attack.json"),
    )
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--seeding_scheme", default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--normalizers", default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    parser.add_argument("--skip_model_load", type=str2bool, default=False)
    return parser.parse_args()


def load_texts(path: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        list(data.get("original_text", [])),
        list(data.get("word_deletion", [])),
        list(data.get("synonym_substitution", [])),
        list(data.get("context_aware_synonym_substitution", [])),
    )


def load_model(args: argparse.Namespace):
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return tokenizer, device


def make_detector(args: argparse.Namespace, tokenizer, device: str) -> WatermarkDetector:
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


def safe_mean(total: float, count: int) -> float | None:
    return total / count if count else None


def main() -> None:
    args = parse_args()
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.normalizers = [item for item in args.normalizers.split(",") if item]
    if args.skip_model_load:
        print(f"Input file: {args.input_file}")
        return

    tokenizer, device = load_model(args)
    detector = make_detector(args, tokenizer, device)
    original, attack1, attack2, attack3 = load_texts(Path(args.input_file))
    attack1_scores, attack1_total_z, attack1_count, attack1_total = detect_batch(attack1, args, device, tokenizer, detector)
    attack2_scores, attack2_total_z, attack2_count, attack2_total = detect_batch(attack2, args, device, tokenizer, detector)
    attack3_scores, attack3_total_z, attack3_count, attack3_total = detect_batch(attack3, args, device, tokenizer, detector)
    original_scores, original_total_z, original_count, original_total = detect_batch(original, args, device, tokenizer, detector)

    result = {
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
        "averate_org_z_score": safe_mean(original_total_z, original_total),
        "averate_attack1_z_score": safe_mean(attack1_total_z, attack1_total),
        "averate_attack2_z_score": safe_mean(attack2_total_z, attack2_total),
        "averate_attack3_z_score": safe_mean(attack3_total_z, attack3_total),
        "averate_org_z_score_2": safe_mean(original_total_z, original_count),
        "averate_attack1_z_score_2": safe_mean(attack1_total_z, attack1_count),
        "averate_attack2_z_score_2": safe_mean(attack2_total_z, attack2_count),
        "averate_attack3_z_score_2": safe_mean(attack3_total_z, attack3_count),
        "original_text_z_score": original_scores,
        "attack1_text_z_score": attack1_scores,
        "attack2_text_z_score": attack2_scores,
        "attack3_text_z_score": attack3_scores,
        "metadata": {
            "input_file": args.input_file,
            "threshold": args.detection_z_threshold,
        },
    }

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
