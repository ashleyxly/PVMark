from __future__ import annotations

import argparse
import json
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

from synthid_text import logits_processing, synthid_mixin  # noqa: E402


DEFAULT_MODEL = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
DEFAULT_OUTPUT = Path(
    "tests/baseline_comparison/hash_synthid_wet_profile/profile.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile hash-based SynthID WET.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hash-type", type=int, default=4)
    parser.add_argument("--token-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--cache-mode", choices=["cold", "warm"], default="cold")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--fused-g-values",
        action="store_true",
        help="Profile the fused Rust g-value path.",
    )
    parser.add_argument(
        "--cpu-update-scores",
        action="store_true",
        help="Profile CPU score updates for tiny top-k tensors.",
    )
    parser.add_argument(
        "--compile-update-scores",
        action="store_true",
        help="Profile torch.compile score updates for tiny top-k tensors.",
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


def configure_backend(
    hash_type: int,
    fused_g_values: bool,
    cpu_update_scores: bool,
    compile_update_scores: bool,
) -> None:
    logits_processing.RUST_LIB = True
    logits_processing.IS_LCG = False
    logits_processing.HASH_TYPE = int(hash_type)
    logits_processing.RUST_FUSED_G_VALUES = bool(fused_g_values)
    logits_processing.CPU_UPDATE_SCORES = bool(cpu_update_scores)
    logits_processing.COMPILE_UPDATE_SCORES = bool(compile_update_scores)
    clear_hash_caches()


def synthid_config(device: torch.device) -> dict[str, Any]:
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    return config


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(callable_obj, device: torch.device):
    sync(device)
    start = time.perf_counter()
    result = callable_obj()
    sync(device)
    return result, time.perf_counter() - start


def profile_embedding_sequence(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
    device: torch.device,
) -> dict[str, float]:
    processor.state = None
    token_length = int(input_ids.shape[1])
    totals: dict[str, float] = {
        "score_preprocess_topk": 0.0,
        "state_update": 0.0,
        "compute_keys": 0.0,
        "sample_g_values": 0.0,
        "compute_fused_g_values": 0.0,
        "update_scores": 0.0,
        "context_history": 0.0,
        "total": 0.0,
    }

    for step in range(token_length):
        prefix_len = max(1, step)
        score_index = max(0, step - 1)
        current_input_ids = input_ids[:, :prefix_len]
        scores = score_bank[:, score_index, :].clone()
        sync(device)
        step_start = time.perf_counter()

        (scores_top_k, top_k_indices, batch_size), duration = timed(
            lambda: _preprocess_scores(processor, scores), device
        )
        totals["score_preprocess_topk"] += duration

        _, duration = timed(
            lambda: _update_state(processor, current_input_ids, batch_size),
            device,
        )
        totals["state_update"] += duration

        if processor.skip_first_ngram_calls and processor.state.num_calls < processor.ngram_len:
            totals["total"] += time.perf_counter() - step_start
            continue

        if logits_processing.RUST_FUSED_G_VALUES:
            (g_values, context_hash), duration = timed(
                lambda: processor._compute_g_values(
                    processor.state.context, top_k_indices
                ),
                device,
            )
            totals["compute_fused_g_values"] += duration
        else:
            (ngram_keys, context_hash), duration = timed(
                lambda: processor._compute_keys(processor.state.context, top_k_indices),
                device,
            )
            totals["compute_keys"] += duration

            g_values, duration = timed(
                lambda: processor.sample_g_values(ngram_keys), device
            )
            totals["sample_g_values"] += duration

        updated_scores, duration = timed(
            lambda: _update_scores(processor, scores_top_k, g_values), device
        )
        totals["update_scores"] += duration

        _, duration = timed(
            lambda: _update_context_history(processor, context_hash, updated_scores, scores_top_k),
            device,
        )
        totals["context_history"] += duration
        totals["total"] += time.perf_counter() - step_start

    return totals


def _preprocess_scores(
    processor: logits_processing.SynthIDLogitsProcessor,
    scores: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.LongTensor, int]:
    scores_processed = scores / processor.temperature
    top_k_result = torch.topk(scores_processed, k=processor.top_k, dim=1)
    batch_size, vocab_size = scores.shape
    if processor.apply_top_k:
        return top_k_result.values, top_k_result.indices, batch_size
    return (
        scores_processed,
        torch.stack([
            torch.arange(vocab_size, device=processor.device)
            for _ in range(batch_size)
        ]),
        batch_size,
    )


def _update_state(
    processor: logits_processing.SynthIDLogitsProcessor,
    input_ids: torch.LongTensor,
    batch_size: int,
) -> None:
    if processor.state is None:
        processor._init_state(batch_size)
    else:
        processor.state.context = torch.concat(
            (processor.state.context, input_ids[:, -1:]),
            dim=1,
        )
        processor.state.context = processor.state.context[:, 1:]
    processor.state.num_calls += 1


def _update_scores(
    processor: logits_processing.SynthIDLogitsProcessor,
    scores_top_k: torch.FloatTensor,
    g_values: torch.LongTensor,
) -> torch.FloatTensor:
    if processor._num_leaves == 2:
        if logits_processing.COMPILE_UPDATE_SCORES:
            return logits_processing._compiled_update_scores(scores_top_k, g_values)
        if logits_processing.CPU_UPDATE_SCORES:
            return logits_processing.update_scores_cpu(scores_top_k, g_values)
        return logits_processing.update_scores(scores_top_k, g_values)
    if logits_processing.COMPILE_UPDATE_SCORES:
        return logits_processing._compiled_update_scores_distortionary(
            scores_top_k, g_values, processor._num_leaves
        )
    if logits_processing.CPU_UPDATE_SCORES:
        return logits_processing.update_scores_distortionary_cpu(
            scores_top_k, g_values, processor._num_leaves
        )
    return logits_processing.update_scores_distortionary(
        scores_top_k, g_values, processor._num_leaves
    )


def _update_context_history(
    processor: logits_processing.SynthIDLogitsProcessor,
    context_hash,
    updated_scores: torch.FloatTensor,
    scores_top_k: torch.FloatTensor,
) -> torch.FloatTensor:
    if logits_processing.RUST_LIB:
        context_hash = torch.tensor(
            [int(hash_value) for hash_value in context_hash],
            device=processor.device,
            dtype=torch.float64,
        )
    context_hash = context_hash[:, None]
    is_repeated_context = (
        processor.state.context_history == context_hash
    ).any(
        dim=1,
        keepdim=True,
    )
    processor.state.context_history = torch.concat(
        (context_hash, processor.state.context_history),
        dim=1,
    )[:, :-1]
    return torch.where(
        is_repeated_context,
        input=scores_top_k,
        other=updated_scores,
    )


def summarize_runs(runs: list[dict[str, float]], token_length: int) -> dict[str, Any]:
    keys = runs[0].keys()
    summary: dict[str, Any] = {}
    for key in keys:
        values_ms = [run[key] * 1000 for run in runs]
        summary[key] = {
            "mean_ms_per_200_tokens": statistics.fmean(values_ms),
            "median_ms_per_200_tokens": statistics.median(values_ms),
            "mean_ms_per_token": statistics.fmean(values_ms) / token_length,
        }
    total_mean = summary["total"]["mean_ms_per_200_tokens"]
    for key in keys:
        summary[key]["pct_of_total"] = (
            summary[key]["mean_ms_per_200_tokens"] / total_mean * 100
            if total_mean
            else 0.0
        )
    return summary


def main() -> None:
    args = parse_args()
    configure_backend(
        args.hash_type,
        args.fused_g_values,
        args.cpu_update_scores,
        args.compile_update_scores,
    )
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    input_ids, available_tokens = synthid_bench.build_detection_batch(
        tokenizer=tokenizer,
        news_text=synthid_bench.DEFAULT_NEWS_TEXT,
        batch_size=args.batch_size,
        token_length=args.token_length,
        device=device,
    )
    with torch.no_grad():
        score_bank = model(input_ids).logits[:, : args.token_length, :].detach()

    for _ in range(args.warmup_runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        profile_embedding_sequence(processor, input_ids, score_bank, device)

    runs = []
    for _ in range(args.runs):
        if args.cache_mode == "cold":
            clear_hash_caches()
        runs.append(profile_embedding_sequence(processor, input_ids, score_bank, device))

    output = {
        "metadata": {
            "model_name_or_path": args.model_name_or_path,
            "device": str(device),
            "hash_type": logits_processing.HASH_TYPE,
            "fused_g_values": logits_processing.RUST_FUSED_G_VALUES,
            "cpu_update_scores": logits_processing.CPU_UPDATE_SCORES,
            "compile_update_scores": logits_processing.COMPILE_UPDATE_SCORES,
            "token_length": args.token_length,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "cache_mode": args.cache_mode,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "available_tokens": available_tokens,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runs": runs,
        "summary": summarize_runs(runs, args.token_length),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
