from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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

from synthid_text import gpu_hash, logits_processing, synthid_mixin  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("tests/baseline_comparison/hash_synthid_efficiency_gpt2_eli5")
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
    parser = argparse.ArgumentParser(description="Benchmark hash-based SynthID WET/WDT.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hash-type", type=int, default=4)
    parser.add_argument("--token-lengths", type=int, nargs="+", default=[200])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wet-runs", type=int, default=100)
    parser.add_argument("--wdt-runs", type=int, default=100)
    parser.add_argument(
        "--skip-wdt",
        action="store_true",
        help="Only run WET timing. Useful when JAX detector backends are unavailable.",
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="warm")
    parser.add_argument(
        "--fused-g-values",
        action="store_true",
        help="Use the fused Rust helper that returns g-values directly for WET.",
    )
    parser.add_argument(
        "--fused-detect-g-values",
        action="store_true",
        help="Use the fused Rust helper for full-sequence detector g-values in WDT.",
    )
    parser.add_argument(
        "--fast-context-mask",
        action="store_true",
        help="Use the batched Rust/Python context repetition mask path for WDT.",
    )
    parser.add_argument(
        "--gpu-hash",
        action="store_true",
        help=(
            "Use the experimental CUDA hash backend where available. "
            "Currently implemented only for exact MiMC WET g-value computation."
        ),
    )
    parser.add_argument(
        "--gpu-fused-score-update",
        action="store_true",
        help=(
            "Also update WET top-k scores in the experimental CUDA backend. "
            "This is faster but changes floating-point reduction order slightly."
        ),
    )
    parser.add_argument(
        "--gpu-fused-history-update",
        action="store_true",
        help=(
            "Fuse online MiMC context repetition/history update into the GPU "
            "g-value kernel while keeping true sequential WET semantics."
        ),
    )
    parser.add_argument(
        "--batched-wet-replay",
        action="store_true",
        help=(
            "Replay WET as one offline multi-token batch. This preserves the "
            "precomputed-logit replay inputs and final scores, but it is not "
            "online generation latency because future logits are already known."
        ),
    )
    parser.add_argument(
        "--cuda-cpp-batched-wet",
        action="store_true",
        help=(
            "Use the optional CUDA C++ extension for batched MiMC WET replay. "
            "Requires --batched-wet-replay, --gpu-hash, and --gpu-fused-score-update."
        ),
    )
    parser.add_argument(
        "--cuda-cpp-score-update",
        action="store_true",
        help=(
            "Use the optional CUDA C++ extension only for batched repetition "
            "checking and top-k score update, while keeping MiMC hash on the "
            "Numba split-context backend."
        ),
    )
    parser.add_argument(
        "--cuda-cpp-online-wet",
        action="store_true",
        help=(
            "Use the optional CUDA C++ extension for true online MiMC WET "
            "hash/history/score update. This is not batched replay."
        ),
    )
    parser.add_argument(
        "--cpu-update-scores",
        action="store_true",
        help="Run tiny top-k score updates on CPU to avoid many small CUDA kernels.",
    )
    parser.add_argument(
        "--compile-update-scores",
        action="store_true",
        help="Use torch.compile on the tiny top-k score update recurrence.",
    )
    parser.add_argument(
        "--fast-detector-score",
        action="store_true",
        help=(
            "Compute hash-based WDT score directly from Rust/NumPy buffers, "
            "avoiding Torch CUDA and JAX round-trips. Requires weighted_mean "
            "and the optimized detector flags."
        ),
    )
    parser.add_argument(
        "--fused-detector-score",
        action="store_true",
        help=(
            "Compute weighted_mean WDT score fully in Rust, fusing detector "
            "g-values, context mask, EOS mask, and score reduction."
        ),
    )
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


def configure_hash_backend(
    hash_type: int,
    fused_g_values: bool,
    fused_detect_g_values: bool,
    fast_context_mask: bool,
    gpu_hash_backend: bool,
    gpu_fused_score_update: bool,
    gpu_fused_history_update: bool,
    cuda_cpp_online_wet: bool,
    cpu_update_scores: bool,
    compile_update_scores: bool,
) -> None:
    logits_processing.RUST_LIB = True
    logits_processing.IS_LCG = False
    logits_processing.HASH_TYPE = int(hash_type)
    logits_processing.RUST_FUSED_G_VALUES = bool(fused_g_values)
    logits_processing.RUST_FUSED_DETECT_G_VALUES = bool(fused_detect_g_values)
    logits_processing.RUST_FAST_CONTEXT_MASK = bool(fast_context_mask)
    logits_processing.GPU_HASH_BACKEND = bool(gpu_hash_backend)
    logits_processing.GPU_FUSED_SCORE_UPDATE = bool(gpu_fused_score_update)
    logits_processing.GPU_FUSED_HISTORY_UPDATE = bool(gpu_fused_history_update)
    logits_processing.CUDA_CPP_ONLINE_WET = bool(cuda_cpp_online_wet)
    logits_processing.CPU_UPDATE_SCORES = bool(cpu_update_scores)
    logits_processing.COMPILE_UPDATE_SCORES = bool(compile_update_scores)
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
        prefix_len = max(1, step)
        score_index = max(0, step - 1)
        processor.watermarked_call(
            input_ids[:, :prefix_len],
            score_bank[:, score_index, :].clone(),
        )


def _build_online_contexts(
    input_ids: torch.LongTensor,
    ngram_len: int,
) -> torch.LongTensor:
    """Build the exact contexts used by run_embedding_sequence for all steps."""
    batch_size, token_length = input_ids.shape
    context_len = ngram_len - 1
    prefix_zeros = torch.zeros(
        (batch_size, context_len),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    padded = torch.cat((prefix_zeros, input_ids[:, : max(token_length - 1, 0)]), dim=1)
    return padded.unfold(dimension=1, size=context_len, step=1)[:, :token_length, :].permute(
        1, 0, 2
    )


@torch.no_grad()
def collect_embedding_sequence_outputs(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay token-by-token SynthID embedding and keep per-step outputs."""
    processor.state = None
    updated_scores = []
    top_k_indices = []
    original_scores = []
    token_length = int(input_ids.shape[1])
    for step in range(token_length):
        prefix_len = max(1, step)
        score_index = max(0, step - 1)
        updated, indices, original = processor.watermarked_call(
            input_ids[:, :prefix_len],
            score_bank[:, score_index, :].clone(),
        )
        updated_scores.append(updated.clone())
        top_k_indices.append(indices.clone())
        original_scores.append(original.clone())
    return (
        torch.stack(updated_scores, dim=1),
        torch.stack(top_k_indices, dim=1),
        torch.stack(original_scores, dim=1),
    )


@torch.no_grad()
def batched_embedding_replay_outputs(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
    use_cuda_cpp: bool = False,
    use_cuda_cpp_score_update: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay WET with all token steps batched into one hash/update workload."""
    if not (
        logits_processing.RUST_LIB
        and logits_processing.RUST_FUSED_G_VALUES
        and logits_processing.GPU_HASH_BACKEND
        and logits_processing.GPU_FUSED_SCORE_UPDATE
        and logits_processing.HASH_TYPE == 5
        and gpu_hash.is_available()
    ):
        return collect_embedding_sequence_outputs(processor, input_ids, score_bank)

    token_length = int(input_ids.shape[1])
    batch_size = int(input_ids.shape[0])
    processor.state = None
    processor._init_state(batch_size)
    assert processor.state is not None
    processor.state.num_calls = token_length

    replay_scores = torch.cat(
        (score_bank[:, :1, :], score_bank[:, : max(token_length - 1, 0), :]),
        dim=1,
    )
    scores_processed = replay_scores / processor.temperature
    top_k_result = torch.topk(scores_processed, k=processor.top_k, dim=2)
    scores_top_k = top_k_result.values.contiguous()
    top_k_indices = top_k_result.indices.contiguous()

    contexts = _build_online_contexts(input_ids, processor.ngram_len)
    flat_contexts = contexts.reshape(token_length * batch_size, -1).contiguous()
    flat_indices = top_k_indices.permute(1, 0, 2).reshape(
        token_length * batch_size, processor.top_k
    ).contiguous()
    flat_scores = scores_top_k.permute(1, 0, 2).reshape(
        token_length * batch_size, processor.top_k
    ).contiguous()

    if use_cuda_cpp:
        from synthid_text import cuda_hash_cpp

        updated_scores = cuda_hash_cpp.compute_batched_updated_scores_use_mimc_cpp(
            flat_contexts,
            flat_indices,
            flat_scores,
            processor._rust_keys,
            token_length,
            batch_size,
            processor.context_history_size,
        )
        return (
            updated_scores.reshape(token_length, batch_size, processor.top_k).permute(
                1, 0, 2
            ),
            top_k_indices,
            scores_top_k,
        )

    g_values, context_limbs = gpu_hash.compute_g_values_use_mimc_gpu_split_context(
        flat_contexts,
        flat_indices,
        processor._rust_keys,
        dtype=None,
        return_context_tensor=True,
    )
    if use_cuda_cpp_score_update:
        from synthid_text import cuda_hash_cpp

        updated_scores = cuda_hash_cpp.update_batched_scores_cpp(
            context_limbs,
            g_values,
            flat_scores,
            token_length,
            batch_size,
            processor.context_history_size,
        )
        return (
            updated_scores.reshape(token_length, batch_size, processor.top_k).permute(
                1, 0, 2
            ),
            top_k_indices,
            scores_top_k,
        )

    repeated_flags = gpu_hash.compute_batched_repetition_flags_gpu(
        context_limbs,
        token_length,
        batch_size,
        processor.context_history_size,
    )
    updated_scores = gpu_hash.update_scores_gpu(flat_scores, g_values, repeated_flags)
    return (
        updated_scores.reshape(token_length, batch_size, processor.top_k).permute(
            1, 0, 2
        ),
        top_k_indices,
        scores_top_k,
    )


@torch.no_grad()
def run_batched_embedding_replay(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
    use_cuda_cpp: bool = False,
    use_cuda_cpp_score_update: bool = False,
) -> None:
    batched_embedding_replay_outputs(
        processor,
        input_ids,
        score_bank,
        use_cuda_cpp=use_cuda_cpp,
        use_cuda_cpp_score_update=use_cuda_cpp_score_update,
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
        if args.batched_wet_replay:
            run_batched_embedding_replay(
                processor,
                input_ids,
                score_bank,
                use_cuda_cpp=args.cuda_cpp_batched_wet,
                use_cuda_cpp_score_update=args.cuda_cpp_score_update,
            )
        else:
            run_embedding_sequence(processor, input_ids, score_bank)
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    for run_idx in range(args.wet_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        start = time.perf_counter()
        if args.batched_wet_replay:
            run_batched_embedding_replay(
                processor,
                input_ids,
                score_bank,
                use_cuda_cpp=args.cuda_cpp_batched_wet,
                use_cuda_cpp_score_update=args.cuda_cpp_score_update,
            )
        else:
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
        "batched_wet_replay": args.batched_wet_replay,
        "cuda_cpp_batched_wet": args.cuda_cpp_batched_wet,
        "cuda_cpp_score_update": args.cuda_cpp_score_update,
        "cuda_cpp_online_wet": args.cuda_cpp_online_wet,
        "gpu_fused_history_update": args.gpu_fused_history_update,
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

    if args.fused_detector_score:
        detection_fn = fused_hash_weighted_mean_detection
    elif args.fast_detector_score:
        detection_fn = fast_hash_detection
    else:
        detection_fn = synthid_bench.run_detection

    for _ in range(args.warmup_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        detection_fn(
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
        detection_fn(
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
        "fast_detector_score": args.fast_detector_score,
        "fused_detector_score": args.fused_detector_score,
        **summarize_timing(durations, token_length, args.batch_size),
    }


def weighted_mean_score_numpy(
    g_values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    depth = g_values.shape[-1]
    weights = np.linspace(10, 1, num=depth, dtype=np.float32)
    weights *= np.float32(depth) / np.sum(weights, dtype=np.float32)
    weighted_g_values = g_values.astype(np.float32, copy=False) * weights.reshape(1, 1, depth)
    num_unmasked = np.sum(mask, axis=1, dtype=np.float32)
    numerator = np.sum(
        weighted_g_values * mask[:, :, None].astype(np.float32, copy=False),
        axis=(1, 2),
        dtype=np.float32,
    )
    return numerator / (np.float32(depth) * num_unmasked)


def mean_score_numpy(
    g_values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    depth = g_values.shape[-1]
    num_unmasked = np.sum(mask, axis=1, dtype=np.float32)
    numerator = np.sum(
        g_values.astype(np.float32, copy=False)
        * mask[:, :, None].astype(np.float32, copy=False),
        axis=(1, 2),
        dtype=np.float32,
    )
    return numerator / (np.float32(depth) * num_unmasked)


def fast_hash_detection(
    token_ids: torch.LongTensor,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    eos_token_id: int,
    score_type: str,
):
    if score_type not in ("mean", "weighted_mean", "both"):
        raise ValueError(f"Unsupported score_type: {score_type}")
    if not (
        logits_processing.RUST_LIB
        and logits_processing.RUST_FUSED_DETECT_G_VALUES
        and logits_processing.RUST_FAST_CONTEXT_MASK
        and not logits_processing.IS_LCG
    ):
        return synthid_bench.run_detection(
            token_ids,
            logits_processor,
            eos_token_id,
            score_type,
        )

    token_np = logits_processing._as_numpy_int64_contiguous(token_ids)
    batch_size, seq_len = token_np.shape
    positions = np.arange(seq_len, dtype=np.int64)[None, :]
    eos_matches = token_np == int(eos_token_id)
    has_eos = np.any(eos_matches, axis=1)
    first_eos = np.argmax(eos_matches, axis=1)
    first_eos = np.where(has_eos, first_eos, seq_len)
    eos_mask = positions < first_eos[:, None]
    eos_mask = eos_mask[:, logits_processor.ngram_len - 1 :]

    context_mask = (
        logits_processing.compute_context_repetition_mask_lcg_from_rustlib_buffer(
            token_ids,
            logits_processor.ngram_len - 1,
            logits_processor.context_history_size,
        )
    )
    combined_mask = context_mask & eos_mask

    keys = logits_processor._rust_keys
    if logits_processing.HASH_TYPE == 3:
        g_values = logits_processing.compute_detect_g_values_use_poseidon_fast_from_rustlib_buffer(
            token_ids, keys, logits_processor.ngram_len
        )
    elif logits_processing.HASH_TYPE == 4:
        g_values = logits_processing.compute_detect_g_values_use_poseidon2_fast_from_rustlib_buffer(
            token_ids, keys, logits_processor.ngram_len
        )
    elif logits_processing.HASH_TYPE == 5:
        g_values = logits_processing.compute_detect_g_values_use_mimc_fast_from_rustlib_buffer(
            token_ids, keys, logits_processor.ngram_len
        )
    else:
        return synthid_bench.run_detection(
            token_ids,
            logits_processor,
            eos_token_id,
            score_type,
        )

    if score_type == "mean":
        return mean_score_numpy(g_values, combined_mask)
    if score_type == "weighted_mean":
        return weighted_mean_score_numpy(g_values, combined_mask)
    return (
        mean_score_numpy(g_values, combined_mask),
        weighted_mean_score_numpy(g_values, combined_mask),
    )


def fused_hash_weighted_mean_detection(
    token_ids: torch.LongTensor,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    eos_token_id: int,
    score_type: str,
):
    if score_type != "weighted_mean":
        return fast_hash_detection(token_ids, logits_processor, eos_token_id, score_type)
    if not (
        logits_processing.RUST_LIB
        and logits_processing.RUST_FUSED_DETECT_G_VALUES
        and logits_processing.RUST_FAST_CONTEXT_MASK
        and not logits_processing.IS_LCG
    ):
        return synthid_bench.run_detection(
            token_ids,
            logits_processor,
            eos_token_id,
            score_type,
        )

    keys = logits_processor._rust_keys
    if logits_processing.HASH_TYPE == 3:
        return logits_processing.compute_weighted_mean_score_use_poseidon_fast_from_rustlib_buffer(
            token_ids,
            keys,
            logits_processor.ngram_len,
            logits_processor.context_history_size,
            eos_token_id,
        )
    if logits_processing.HASH_TYPE == 4:
        return logits_processing.compute_weighted_mean_score_use_poseidon2_fast_from_rustlib_buffer(
            token_ids,
            keys,
            logits_processor.ngram_len,
            logits_processor.context_history_size,
            eos_token_id,
        )
    if logits_processing.HASH_TYPE == 5:
        return logits_processing.compute_weighted_mean_score_use_mimc_fast_from_rustlib_buffer(
            token_ids,
            keys,
            logits_processor.ngram_len,
            logits_processor.context_history_size,
            eos_token_id,
        )
    return fast_hash_detection(token_ids, logits_processor, eos_token_id, score_type)


def main() -> None:
    args = parse_args()
    configure_hash_backend(
        args.hash_type,
        args.fused_g_values,
        args.fused_detect_g_values,
        args.fast_context_mask,
        args.gpu_hash,
        args.gpu_fused_score_update,
        args.gpu_fused_history_update,
        args.cuda_cpp_online_wet,
        args.cpu_update_scores,
        args.compile_update_scores,
    )
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
            "backend": "hash-rust",
            "hash_type": logits_processing.HASH_TYPE,
            "fused_g_values": logits_processing.RUST_FUSED_G_VALUES,
            "fused_detect_g_values": logits_processing.RUST_FUSED_DETECT_G_VALUES,
            "fast_context_mask": logits_processing.RUST_FAST_CONTEXT_MASK,
            "gpu_hash_backend": logits_processing.GPU_HASH_BACKEND,
            "gpu_fused_score_update": logits_processing.GPU_FUSED_SCORE_UPDATE,
            "gpu_fused_history_update": logits_processing.GPU_FUSED_HISTORY_UPDATE,
            "cuda_cpp_online_wet": logits_processing.CUDA_CPP_ONLINE_WET,
            "batched_wet_replay": args.batched_wet_replay,
            "cuda_cpp_batched_wet": args.cuda_cpp_batched_wet,
            "cuda_cpp_score_update": args.cuda_cpp_score_update,
            "cpu_update_scores": logits_processing.CPU_UPDATE_SCORES,
            "compile_update_scores": logits_processing.COMPILE_UPDATE_SCORES,
            "fast_detector_score": args.fast_detector_score,
            "fused_detector_score": args.fused_detector_score,
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
                "WET times token_length sequential watermarked_call "
                "invocations with precomputed logits. WDT excludes model "
                "loading and tokenization, and uses the same static news text "
                "benchmark as notebooks/test_detect_time.py."
            ),
        },
        "wet": {},
        "wdt": {},
    }

    for token_length in args.token_lengths:
        output["wet"][str(token_length)] = time_wet(args, model, tokenizer, token_length, device)
        if not args.skip_wdt:
            output["wdt"][str(token_length)] = time_wdt(args, tokenizer, token_length, device)

    out_path = output_dir / "efficiency_hash_synthid_timing.json"
    write_json(out_path, output)
    print(out_path)
    print(output)


if __name__ == "__main__":
    main()
