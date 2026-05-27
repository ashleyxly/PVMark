from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
REPO_ROOT = NOTEBOOKS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(NOTEBOOKS_DIR))
import test_detect_time as synthid_bench  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from common import ensure_dir, write_json  # noqa: E402

from synthid_text import logits_processing, synthid_mixin  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("tests/baseline_comparison/original_synthid_efficiency_warm")
DEFAULT_MODEL = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))


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
    parser = argparse.ArgumentParser(description="Benchmark original SynthID WET/WDT.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backend",
        choices=["original-python", "original-rust-lcg"],
        default="original-python",
        help=(
            "Backend for Original SynthID. original-python is the upstream "
            "non-hash path; original-rust-lcg uses the same LCG logic through "
            "the Rust helper."
        ),
    )
    parser.add_argument("--token-lengths", type=int, nargs="+", default=[200])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wet-runs", type=int, default=100)
    parser.add_argument("--wdt-runs", type=int, default=100)
    parser.add_argument(
        "--skip-wdt",
        action="store_true",
        help="Only run WET timing. Useful when detector backends are not needed.",
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="warm")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--score-type",
        choices=["mean", "weighted_mean", "both"],
        default="weighted_mean",
    )
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
    logits_processing.compute_g_values_use_LCG_from_rustlib.cache_clear()
    logits_processing.compute_keys_use_hash_from_rustlib.cache_clear()
    logits_processing.invoke_sample_g_values_use_hash_from_rustlib.cache_clear()
    logits_processing.compute_ngram_keys_use_hash_from_rustlib.cache_clear()
    logits_processing.compute_g_values_use_hash_from_rustlib.cache_clear()
    logits_processing.compute_g_values_use_poseidon_fast_from_rustlib.cache_clear()
    logits_processing.compute_g_values_use_poseidon2_fast_from_rustlib.cache_clear()
    logits_processing.compute_g_values_use_mimc_fast_from_rustlib.cache_clear()


def configure_original_backend(backend: str) -> None:
    if backend == "original-python":
        logits_processing.RUST_LIB = False
        logits_processing.IS_LCG = True
    elif backend == "original-rust-lcg":
        logits_processing.RUST_LIB = True
        logits_processing.IS_LCG = True
    else:
        raise ValueError(f"Unsupported Original SynthID backend: {backend}")
    clear_hash_caches()


def synthid_config(device: torch.device) -> dict[str, Any]:
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    return config


def summarize_timing(
    durations: list[float],
    token_length: int,
    batch_size: int,
) -> dict[str, Any]:
    mean_batch_ms = statistics.fmean(durations) * 1000
    median_batch_ms = statistics.median(durations) * 1000
    summary = {
        "runs": len(durations),
        "mean_ms_per_batch": mean_batch_ms,
        "median_ms_per_batch": median_batch_ms,
        "min_ms_per_batch": min(durations) * 1000,
        "max_ms_per_batch": max(durations) * 1000,
        "p90_ms_per_batch": float(
            statistics.quantiles([d * 1000 for d in durations], n=10)[-1]
        )
        if len(durations) >= 10
        else max(durations) * 1000,
    }
    summary["mean_ms_per_sample"] = mean_batch_ms / batch_size
    summary["median_ms_per_sample"] = median_batch_ms / batch_size
    summary["p90_ms_per_sample"] = summary["p90_ms_per_batch"] / batch_size
    summary["mean_ms_per_token"] = summary["mean_ms_per_sample"] / token_length
    summary["median_ms_per_token"] = summary["median_ms_per_sample"] / token_length
    summary["samples_per_sec"] = (batch_size * 1000.0) / mean_batch_ms
    summary["duration_ms_per_batch_values"] = [float(d * 1000) for d in durations]
    summary["duration_ms_per_sample_values"] = [
        float(d * 1000 / batch_size) for d in durations
    ]
    summary["duration_ms_per_token_values"] = [
        float(d * 1000 / batch_size / token_length) for d in durations
    ]
    return summary


def run_embedding_sequence(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
) -> None:
    """Replay token-by-token SynthID embedding without timing LLM forward."""
    processor.state = None
    token_length = int(input_ids.shape[1])
    for step in range(token_length):
        # The implementation initializes state on the first generated-token
        # decision, then appends the previously generated token on later calls.
        prefix_len = max(1, step)
        score_index = max(0, step - 1)
        processor.watermarked_call(
            input_ids[:, :prefix_len],
            score_bank[:, score_index, :].clone(),
        )


def time_wet(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    token_length: int,
    device: torch.device,
) -> dict[str, Any]:
    processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    input_ids, available_tokens = synthid_bench.build_detection_batch(
        tokenizer=tokenizer,
        news_text=synthid_bench.DEFAULT_NEWS_TEXT,
        batch_size=args.batch_size,
        token_length=token_length,
        device=device,
    )
    with torch.no_grad():
        score_bank = model(input_ids).logits[:, :token_length, :].detach()

    for _ in range(args.warmup_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        run_embedding_sequence(processor, input_ids, score_bank)
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    for run_idx in range(args.wet_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        start = time.perf_counter()
        run_embedding_sequence(processor, input_ids, score_bank)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        if args.progress_every and (run_idx + 1) % args.progress_every == 0:
            print(f"WET token_length={token_length}: {run_idx + 1}/{args.wet_runs}", flush=True)

    return {
        "definition": (
            "WET is the SynthID processor overhead for token_length sequential "
            "watermark embedding decisions; LLM forward is excluded and "
            "precomputed logits are replayed."
        ),
        "token_length": token_length,
        "timed_embedding_calls": token_length,
        "available_tokens": available_tokens,
        "batch_size": args.batch_size,
        "cache_mode": args.cache_mode,
        **summarize_timing(durations, token_length, args.batch_size),
    }


def time_wdt(
    args: argparse.Namespace,
    tokenizer: Any,
    token_length: int,
    device: torch.device,
) -> dict[str, Any]:
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    token_ids, available_tokens = synthid_bench.build_detection_batch(
        tokenizer=tokenizer,
        news_text=synthid_bench.DEFAULT_NEWS_TEXT,
        batch_size=args.batch_size,
        token_length=token_length,
        device=device,
    )

    for _ in range(args.warmup_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        synthid_bench.run_detection(
            token_ids,
            logits_processor,
            tokenizer.eos_token_id,
            args.score_type,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    for run_idx in range(args.wdt_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        start = time.perf_counter()
        synthid_bench.run_detection(
            token_ids,
            logits_processor,
            tokenizer.eos_token_id,
            args.score_type,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        if args.progress_every and (run_idx + 1) % args.progress_every == 0:
            print(f"WDT token_length={token_length}: {run_idx + 1}/{args.wdt_runs}", flush=True)

    return {
        "definition": (
            "WDT is SynthID detection on a token_length-token text, including "
            "EOS mask, context repetition mask, g-value computation, and "
            "the requested detector score."
        ),
        "token_length": token_length,
        "available_tokens": available_tokens,
        "batch_size": args.batch_size,
        "cache_mode": args.cache_mode,
        "score_type": args.score_type,
        **summarize_timing(durations, token_length, args.batch_size),
    }


def main() -> None:
    args = parse_args()
    configure_original_backend(args.backend)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    output = {
        "metadata": {
            "model_name_or_path": args.model_name_or_path,
            "device": str(device),
            "backend": args.backend,
            "hash_type": None,
            "hash_type_note": "unused for Original SynthID LCG backends",
            "rust_lib": logits_processing.RUST_LIB,
            "is_lcg": logits_processing.IS_LCG,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "token_lengths": args.token_lengths,
            "batch_size": args.batch_size,
            "warmup_runs": args.warmup_runs,
            "progress_every": args.progress_every,
            "wet_runs": args.wet_runs,
            "wdt_runs": args.wdt_runs,
            "skip_wdt": args.skip_wdt,
            "cache_mode": args.cache_mode,
            "score_type": args.score_type,
            "runtime_environment": runtime_environment(),
            "note": (
                "Original SynthID timing explicitly selects a non-hash LCG "
                "backend. WET times token_length sequential watermarked_call "
                "invocations with precomputed logits; WDT times the detection "
                "core. Model loading, tokenization, and LLM forward are "
                "excluded from the timed regions."
            ),
        },
        "wet": {},
        "wdt": {},
    }

    for token_length in args.token_lengths:
        output["wet"][str(token_length)] = time_wet(args, model, tokenizer, token_length, device)
        if not args.skip_wdt:
            output["wdt"][str(token_length)] = time_wdt(args, tokenizer, token_length, device)

    out_path = output_dir / "efficiency_original_synthid_timing.json"
    write_json(out_path, output)
    print(out_path)
    print(output)


if __name__ == "__main__":
    main()
