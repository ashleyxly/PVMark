from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from baseline_eval.benchmark_efficiency import (
    HASH_METHODS,
    HASH_TYPES,
    clear_hash_caches,
    clear_processor_caches,
    finite_float,
    make_hash_kgw,
    make_original_kgw,
)
from baseline_eval.common import (
    DEFAULT_DATASET,
    DEFAULT_GENERATION_MODEL,
    WallTimer,
    count_tokens,
    decode_new_tokens,
    ensure_dir,
    load_c4_records,
    load_causal_lm_and_tokenizer,
    prepare_prompt,
    set_seed,
    write_json,
)


DEFAULT_OUTPUT_DIR = "test_result/two_layer_efficiency_20260522/e2e_kgw_hash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end generation/detection timing for KGW and hash KGW.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--force-max-new-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-max-length", type=int, default=1848)
    parser.add_argument("--warmup-new-tokens", type=int, default=4)
    parser.add_argument("--schemes", default="original,hash")
    parser.add_argument(
        "--hash-variants",
        default="3:2,3:4,4:2,5:2,5:4",
        help="Comma-separated hash_type:hash_method pairs.",
    )
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--seeding-scheme", default="simple_1")
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--original-hash-key", type=int, default=15485863)
    parser.add_argument("--hash-kgw-hash-key", type=int, default=2023)
    parser.add_argument("--sampling-temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_scheme_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def parse_hash_variants(value: str) -> list[tuple[int, int]]:
    variants: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        hash_type = int(left)
        hash_method = int(right)
        if hash_type not in HASH_TYPES:
            raise ValueError(f"unknown hash_type={hash_type}")
        if hash_method not in HASH_METHODS:
            raise ValueError(f"unknown hash_method={hash_method}")
        variants.append((hash_type, hash_method))
    return variants


def score_detector(
    detector: Any,
    text: str,
    ids: torch.Tensor,
    device: str,
    clear_cache_fn: Any | None = None,
) -> dict[str, Any]:
    if clear_cache_fn:
        clear_cache_fn()
    synchronize(device)
    with WallTimer() as raw_timer:
        raw_score = detector.detect(text=text)
    if clear_cache_fn:
        clear_cache_fn()
    synchronize(device)
    with WallTimer() as ids_timer:
        ids_score = detector.detect(tokenized_text=ids.squeeze(0))
    return {
        "raw_text_ms": raw_timer.elapsed * 1000.0,
        "token_ids_ms": ids_timer.elapsed * 1000.0,
        "raw_score": raw_score,
        "ids_score": ids_score,
    }


def detector_prediction(score: dict[str, Any]) -> bool | None:
    prediction = score.get("prediction")
    if isinstance(prediction, bool):
        return prediction
    return None


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    processor: Any | None,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    prompt_max_length: int,
    sampling_temp: float,
    top_k: int,
    force_max_new_tokens: bool,
) -> tuple[torch.Tensor, str, float]:
    tokd_input = prepare_prompt(tokenizer, prompt, device, prompt_max_length)
    prompt_len = tokd_input["input_ids"].shape[-1]
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": top_k,
        "temperature": sampling_temp,
    }
    if force_max_new_tokens:
        gen_kwargs["min_new_tokens"] = max_new_tokens
    if processor is not None:
        gen_kwargs["logits_processor"] = LogitsProcessorList([processor])

    set_seed(seed)
    with WallTimer() as timer, torch.inference_mode():
        output = model.generate(**tokd_input, **gen_kwargs)
    new_ids = output[:, prompt_len:]
    return new_ids, decode_new_tokens(tokenizer, new_ids), timer.elapsed


def warmup_scheme(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    processor: Any,
    prompt: str,
    args: argparse.Namespace,
) -> None:
    if args.warmup_new_tokens <= 0:
        return
    generate_one(
        model=model,
        tokenizer=tokenizer,
        device=device,
        processor=None,
        prompt=prompt,
        seed=args.seed,
        max_new_tokens=args.warmup_new_tokens,
        prompt_max_length=args.prompt_max_length,
        sampling_temp=args.sampling_temp,
        top_k=args.top_k,
        force_max_new_tokens=False,
    )
    generate_one(
        model=model,
        tokenizer=tokenizer,
        device=device,
        processor=processor,
        prompt=prompt,
        seed=args.seed,
        max_new_tokens=args.warmup_new_tokens,
        prompt_max_length=args.prompt_max_length,
        sampling_temp=args.sampling_temp,
        top_k=args.top_k,
        force_max_new_tokens=False,
    )


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    wm_gen = [
        sample["generation_time_with_watermark_sec"] / sample["token_count_with_watermark"] * 1000.0
        for sample in samples
        if sample["token_count_with_watermark"]
    ]
    plain_gen = [
        sample["generation_time_without_watermark_sec"] / sample["token_count_without_watermark"] * 1000.0
        for sample in samples
        if sample["token_count_without_watermark"]
    ]
    overhead = [
        (sample["generation_time_with_watermark_sec"] - sample["generation_time_without_watermark_sec"])
        / sample["token_count_with_watermark"]
        * 1000.0
        for sample in samples
        if sample["token_count_with_watermark"]
    ]
    return {
        "num_samples": len(samples),
        "avg_token_count_with_watermark": mean([float(sample["token_count_with_watermark"]) for sample in samples]),
        "avg_token_count_without_watermark": mean([float(sample["token_count_without_watermark"]) for sample in samples]),
        "watermarked_generation_ms_per_token": stats(wm_gen),
        "plain_generation_ms_per_token": stats(plain_gen),
        "end_to_end_overhead_ms_per_token": stats(overhead),
        "watermarked_detection_raw_text_ms": stats([sample["detection_with_watermark"]["raw_text_ms"] for sample in samples]),
        "watermarked_detection_token_ids_ms": stats([sample["detection_with_watermark"]["token_ids_ms"] for sample in samples]),
        "plain_detection_raw_text_ms": stats([sample["detection_without_watermark"]["raw_text_ms"] for sample in samples]),
        "plain_detection_token_ids_ms": stats([sample["detection_without_watermark"]["token_ids_ms"] for sample in samples]),
        "watermarked_detected": sum(1 for sample in samples if sample.get("prediction_with_watermark")),
        "plain_false_positive": sum(1 for sample in samples if sample.get("prediction_without_watermark")),
    }


def run_scheme(
    *,
    scheme_name: str,
    category: str,
    processor: Any,
    detector: Any,
    model: Any,
    tokenizer: Any,
    device: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    hash_type: int | None = None,
    hash_method: int | None = None,
) -> dict[str, Any]:
    print(f"Running end-to-end {scheme_name}", flush=True)
    if records and args.warmup_new_tokens > 0:
        warmup_scheme(
            model=model,
            tokenizer=tokenizer,
            device=device,
            processor=processor,
            prompt=records[0]["input_text"],
            args=args,
        )

    samples: list[dict[str, Any]] = []
    for row in tqdm(records, desc=scheme_name):
        if category == "hash_kgw":
            clear_hash_caches()
            clear_processor_caches(processor)
        sample_id = int(row["id"])
        sample_seed = int(args.seed) + sample_id

        plain_ids, plain_text, plain_elapsed = generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            processor=None,
            prompt=row["input_text"],
            seed=sample_seed,
            max_new_tokens=args.max_new_tokens,
            prompt_max_length=args.prompt_max_length,
            sampling_temp=args.sampling_temp,
            top_k=args.top_k,
            force_max_new_tokens=args.force_max_new_tokens,
        )
        if category == "hash_kgw":
            clear_hash_caches()
            clear_processor_caches(processor)
        wm_ids, wm_text, wm_elapsed = generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            processor=processor,
            prompt=row["input_text"],
            seed=sample_seed,
            max_new_tokens=args.max_new_tokens,
            prompt_max_length=args.prompt_max_length,
            sampling_temp=args.sampling_temp,
            top_k=args.top_k,
            force_max_new_tokens=args.force_max_new_tokens,
        )

        clear_detection_cache = clear_hash_caches if category == "hash_kgw" else None
        plain_detection = score_detector(detector, plain_text, plain_ids, device, clear_detection_cache)
        wm_detection = score_detector(detector, wm_text, wm_ids, device, clear_detection_cache)
        samples.append(
            {
                "id": sample_id,
                "input_text": row["input_text"],
                "reference_text_removed": row.get("reference_text_removed", ""),
                "output_without_watermark": plain_text,
                "output_with_watermark": wm_text,
                "token_count_without_watermark": int(plain_ids.numel()),
                "token_count_with_watermark": int(wm_ids.numel()),
                "retokenized_without_watermark": count_tokens(tokenizer, plain_text),
                "retokenized_with_watermark": count_tokens(tokenizer, wm_text),
                "generation_time_without_watermark_sec": plain_elapsed,
                "generation_time_with_watermark_sec": wm_elapsed,
                "detection_without_watermark": plain_detection,
                "detection_with_watermark": wm_detection,
                "prediction_without_watermark": detector_prediction(plain_detection["raw_score"]),
                "prediction_with_watermark": detector_prediction(wm_detection["raw_score"]),
            }
        )

    result = {
        "scheme": scheme_name,
        "category": category,
        "hash_type": hash_type,
        "hash_method": hash_method,
        "samples": samples,
        "summary": summarize_samples(samples),
    }
    return result


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    set_seed(args.seed)
    model, tokenizer, device = load_causal_lm_and_tokenizer(
        args.model,
        use_gpu=device.startswith("cuda"),
        load_fp16=False,
        device=device,
    )
    records = load_c4_records(args.dataset, args.max_samples)
    selected = parse_scheme_set(args.schemes)
    variants = parse_hash_variants(args.hash_variants)

    schemes: list[dict[str, Any]] = []
    if "original" in selected:
        processor, detector = make_original_kgw(tokenizer, args, device)
        schemes.append(
            run_scheme(
                scheme_name="Original KGW",
                category="kgw",
                processor=processor,
                detector=detector,
                model=model,
                tokenizer=tokenizer,
                device=device,
                records=records,
                args=args,
            )
        )

    if "hash" in selected:
        for hash_type, hash_method in variants:
            scheme_name = f"Hash KGW {HASH_TYPES[hash_type]} {HASH_METHODS[hash_method]}"
            processor, detector = make_hash_kgw(tokenizer, args, device, hash_type, hash_method)
            schemes.append(
                run_scheme(
                    scheme_name=scheme_name,
                    category="hash_kgw",
                    processor=processor,
                    detector=detector,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    records=records,
                    args=args,
                    hash_type=hash_type,
                    hash_method=hash_method,
                )
            )

    payload = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "definition": "end-to-end generation plus detection timing for KGW/hash KGW; model is loaded once before timing.",
            "dataset": args.dataset,
            "model": args.model,
            "device": device,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "prompt_max_length": args.prompt_max_length,
            "warmup_new_tokens": args.warmup_new_tokens,
            "force_max_new_tokens": args.force_max_new_tokens,
            "schemes": sorted(selected),
            "hash_variants": variants,
            "sampling": {"do_sample": True, "top_k": args.top_k, "temperature": args.sampling_temp},
        },
        "schemes": schemes,
    }
    out_path = Path(out_dir) / "end_to_end_results.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}", flush=True)
    for scheme in schemes:
        summary = scheme["summary"]
        wet = finite_float(summary["watermarked_generation_ms_per_token"]["mean"])
        overhead = finite_float(summary["end_to_end_overhead_ms_per_token"]["mean"])
        print(f"{scheme['scheme']}: WM gen {wet:.4f} ms/token; overhead {overhead:.4f} ms/token", flush=True)


if __name__ == "__main__":
    main()
