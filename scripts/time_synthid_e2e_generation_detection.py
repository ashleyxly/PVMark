from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NOTEBOOKS_DIR))
import test_detect_time as synthid_bench  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from common import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    DEFAULT_GPT2_PATH,
    ensure_dir,
    load_eli5_prompts,
    set_seed,
    summarize_numbers,
    write_json,
)

from synthid_text import logits_processing, synthid_mixin  # noqa: E402


def runtime_environment() -> dict[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-sample end-to-end generation/detection timing for SynthID-style schemes."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--scheme", choices=["original", "hash"], required=True)
    parser.add_argument("--hash-type", type=int, default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--score-type", choices=["mean", "weighted_mean", "both"], default="weighted_mean")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device(device_arg)


def clear_hash_caches() -> None:
    logits_processing.compute_keys_use_LCG_from_rustlib.cache_clear()
    logits_processing.invoke_sample_g_values_use_LCG_from_rustlib.cache_clear()
    logits_processing.compute_ngram_keys_use_LCG_from_rustlib.cache_clear()
    logits_processing.compute_keys_use_hash_from_rustlib.cache_clear()
    logits_processing.invoke_sample_g_values_use_hash_from_rustlib.cache_clear()
    logits_processing.compute_ngram_keys_use_hash_from_rustlib.cache_clear()


def configure_backend(args: argparse.Namespace) -> str:
    if args.scheme == "original":
        logits_processing.RUST_LIB = False
        logits_processing.IS_LCG = True
        clear_hash_caches()
        return "original_synthid_non_hash_lcg"
    if args.hash_type is None:
        raise ValueError("--hash-type is required when --scheme hash")
    logits_processing.RUST_LIB = True
    logits_processing.IS_LCG = False
    logits_processing.HASH_TYPE = int(args.hash_type)
    clear_hash_caches()
    return f"pvmark_hash_type_{args.hash_type}"


def synthid_config(device: torch.device) -> dict[str, Any]:
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_tokenizer(path: str) -> Any:
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_watermarked_model(path: str) -> Any:
    model = synthid_mixin.SynthIDGPT2LMHeadModel.from_pretrained(path, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    return model


def load_plain_model(path: str) -> Any:
    model = transformers.GPT2LMHeadModel.from_pretrained(path, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    return model


def generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int, float]:
    synchronize(device)
    start = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(model.device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            return_dict_in_generate=False,
        )
    completion_ids = output[:, prompt_len:]
    completion_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
    synchronize(device)
    elapsed = time.perf_counter() - start
    return completion_text, int(completion_ids.shape[-1]), elapsed


def detect_once(
    tokenizer: Any,
    detector_processor: Any,
    text: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Any, int, float]:
    synchronize(device)
    start = time.perf_counter()
    token_ids = tokenizer(text or " ", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    score = synthid_bench.run_detection(
        token_ids,
        detector_processor,
        tokenizer.eos_token_id,
        args.score_type,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    return score, int(token_ids.shape[-1]), elapsed


def summarize_records(records: list[dict[str, Any]], watermarked: bool) -> dict[str, Any]:
    subset = [r for r in records if bool(r["watermarked"]) == watermarked]
    return {
        "count": len(subset),
        "generation_time_sec": summarize_numbers(r["generation_time_sec"] for r in subset),
        "detection_time_sec": summarize_numbers(r["detection_time_sec"] for r in subset),
        "generation_tokens": summarize_numbers(r["generation_token_count"] for r in subset),
        "detection_tokens": summarize_numbers(r["detection_token_count"] for r in subset),
        "generation_ms_per_token": summarize_numbers(
            1000.0 * r["generation_time_sec"] / r["generation_token_count"]
            for r in subset
            if r["generation_token_count"] > 0
        ),
        "detection_ms_per_token": summarize_numbers(
            1000.0 * r["detection_time_sec"] / r["detection_token_count"]
            for r in subset
            if r["detection_token_count"] > 0
        ),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    backend_label = configure_backend(args)
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_name_or_path)
    prompts = load_eli5_prompts(args.dataset_path, args.limit)

    detector_processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    watermarked_model = load_watermarked_model(args.model_name_or_path)
    plain_model = load_plain_model(args.model_name_or_path)

    records: list[dict[str, Any]] = []
    for item in prompts:
        for watermarked, model in ((True, watermarked_model), (False, plain_model)):
            set_seed(args.seed + int(item["sample_id"]) * 17 + (1 if watermarked else 0))
            text, gen_tokens, gen_time = generate_once(model, tokenizer, item["prompt"], args, device)
            score, det_tokens, det_time = detect_once(
                tokenizer, detector_processor, text, args, device
            )
            records.append(
                {
                    **item,
                    "scheme": backend_label,
                    "watermarked": watermarked,
                    "completion_text": text,
                    "generation_token_count": gen_tokens,
                    "detection_token_count": det_tokens,
                    "generation_time_sec": gen_time,
                    "detection_time_sec": det_time,
                    "score": score.tolist() if hasattr(score, "tolist") else score,
                }
            )

    output = {
        "metadata": {
            "scheme": backend_label,
            "hash_type": args.hash_type,
            "model_name_or_path": args.model_name_or_path,
            "dataset_path": args.dataset_path,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "score_type": args.score_type,
            "definition": (
                "End-to-end generation excludes model loading but includes prompt "
                "tokenization, 200-token generation, and decoding. End-to-end "
                "detection includes text tokenization and SynthID detector core."
            ),
            "runtime_environment": runtime_environment(),
        },
        "summary": {
            "watermarked": summarize_records(records, True),
            "unwatermarked_plain": summarize_records(records, False),
        },
        "records": records,
    }
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "synthid_e2e_generation_detection.json", output)
    print(out_dir / "synthid_e2e_generation_detection.json")


if __name__ == "__main__":
    main()
