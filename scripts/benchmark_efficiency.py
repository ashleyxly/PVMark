from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from baseline_eval.common import DEFAULT_GENERATION_MODEL, DEFAULT_RESULTS_ROOT, DEFAULT_UPV_ROOT, ensure_dir, read_json, write_json
from baseline_eval.upv_network import UpvNetworkDetector


DEFAULT_OUTPUT_DIR = "test_result/efficiency_benchmark"
DEFAULT_KGW_GENERATIONS = str(Path(DEFAULT_RESULTS_ROOT) / "kgw" / "opt1.3b_c4_num100_legacy_org" / "generations.json")
DEFAULT_UPV_GENERATIONS = str(Path(DEFAULT_RESULTS_ROOT) / "unforgeable_network" / "opt1.3b_c4_num100_network_detector_repo_generator" / "generations.json")
DEFAULT_PDW_GENERATIONS = str(Path(DEFAULT_RESULTS_ROOT) / "publicly_detectable" / "opt1.3b_c4_num100_parallel7_pdw1024_timeout30" / "generations.json")
DEFAULT_HASH_RESULTS_DIR = "test_result/c4_dataset_test"
DEFAULT_UPV_GENERATOR = str(Path(DEFAULT_UPV_ROOT) / "experiments" / "main_experiments" / "generator_model" / "combine_model.pt") if DEFAULT_UPV_ROOT else ""

HASH_TYPES = {
    3: "Poseidon",
    4: "Poseidon2",
    5: "MiMC",
}

HASH_METHODS = {
    2: "TwoToOneFixed",
    4: "ThreeToOneFixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark WET/WDT for KGW, hash-based KGW, UPV, and PDW.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of silently falling back to CPU when --device starts with cuda and CUDA is unavailable.",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--wet-max-samples", type=int, default=None)
    parser.add_argument("--wdt-max-samples", type=int, default=None)
    parser.add_argument("--wet-token-count", type=int, default=200)
    parser.add_argument("--wdt-token-counts", default="50,200")
    parser.add_argument("--prompt-max-length", type=int, default=1848)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--seeding-scheme", default="simple_1")
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--original-hash-key", type=int, default=15485863)
    parser.add_argument("--hash-kgw-hash-key", type=int, default=2023)
    parser.add_argument("--hash-types", default="3,4,5", help="Comma-separated hash types to benchmark; 3=Poseidon, 4=Poseidon2, 5=MiMC.")
    parser.add_argument("--hash-methods", default="2,4", help="Comma-separated hash methods to benchmark; fixed variants are 2 and 4.")
    parser.add_argument(
        "--hash-wet-backend",
        choices=[
            "cpu-u32",
            "cpu-mask",
            "cpu-list",
            "poseidon2-gpu-greenlist",
            "poseidon2-gpu-greenlist-native",
            "poseidon2-gpu-fused",
            "poseidon2-gpu-native-fused",
            "poseidon2-gpu-mask-cache",
            "poseidon2-gpu-id-cache",
            "poseidon-gpu-id-cache",
            "mimc-gpu-id-cache",
            "rust-id-cache",
        ],
        default="cpu-u32",
        help=(
            "Embedding backend for hash-KGW fixed-threshold WET. Poseidon2 GPU modes affect "
            "hash_type=4; poseidon-gpu-id-cache affects hash_type=3; mimc-gpu-id-cache affects "
            "hash_type=5; rust-id-cache supports fixed variants for hash types 3/4/5."
        ),
    )
    parser.add_argument(
        "--hash-result-label-suffix",
        default="",
        help="Optional suffix appended to hash-KGW scheme names, useful when merging CPU/GPU backend runs.",
    )
    parser.add_argument(
        "--schemes",
        default="original,hash,upv,pdw",
        help="Comma-separated scheme groups to benchmark: original,hash,upv,pdw.",
    )
    parser.add_argument("--kgw-generations", default=DEFAULT_KGW_GENERATIONS)
    parser.add_argument("--hash-results-dir", default=DEFAULT_HASH_RESULTS_DIR)
    parser.add_argument(
        "--prefill-hash-wet-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before measuring hash-KGW WET, precompute all greenlists/masks needed by the measured "
            "token prefixes. This isolates online lookup+bias cost; prefill time is reported separately."
        ),
    )
    parser.add_argument("--upv-generations", default=DEFAULT_UPV_GENERATIONS)
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument("--upv-generator-model", default=DEFAULT_UPV_GENERATOR)
    parser.add_argument("--pdw-generations", default=DEFAULT_PDW_GENERATIONS)
    parser.add_argument("--skip-upv-processor-wet", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def resolve_device(device: str, *, require_cuda: bool = False) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        if require_cuda:
            raise SystemExit(f"CUDA is required by --device={device}, but torch.cuda.is_available() is false")
        return "cpu"
    return device


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def stats_ms(values_sec: list[float]) -> dict[str, float | None]:
    values_ms = [v * 1000.0 for v in values_sec]
    return {
        "mean": mean(values_ms),
        "median": median(values_ms),
        "min": min(values_ms) if values_ms else None,
        "p05": quantile(values_ms, 0.05),
        "p25": quantile(values_ms, 0.25),
        "p75": quantile(values_ms, 0.75),
        "p95": quantile(values_ms, 0.95),
        "max": max(values_ms) if values_ms else None,
    }


def stats_values(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values) if values else None,
        "p05": quantile(values, 0.05),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def parse_token_counts(value: str) -> list[int]:
    counts = []
    for item in value.split(","):
        item = item.strip()
        if item:
            counts.append(int(item))
    return sorted(set(counts))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def configure_hash_wet_backend(backend: str, device: str) -> dict[str, str]:
    if (
        backend.startswith("poseidon2-gpu")
        or backend.startswith("poseidon-gpu")
        or backend.startswith("mimc-gpu")
    ) and not device.startswith("cuda"):
        raise SystemExit(f"--hash-wet-backend={backend} requires a CUDA device; resolved device is {device!r}")

    env_by_backend = {
        "cpu-u32": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "cpu-mask": {
            "HASH_KGW_RUST_U32": "0",
            "HASH_KGW_RUST_MASK": "1",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "cpu-list": {
            "HASH_KGW_RUST_U32": "0",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon2-gpu-greenlist": {
            "HASH_KGW_RUST_U32": "0",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "1",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon2-gpu-greenlist-native": {
            "HASH_KGW_RUST_U32": "0",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "native",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon2-gpu-fused": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "1",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "1",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon2-gpu-native-fused": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "native",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "native",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon2-gpu-mask-cache": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "1",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "1",
        },
        "poseidon2-gpu-id-cache": {
            "HASH_KGW_RUST_U32": "0",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "native",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "poseidon-gpu-id-cache": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "mimc-gpu-id-cache": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
        "rust-id-cache": {
            "HASH_KGW_RUST_U32": "1",
            "HASH_KGW_RUST_MASK": "0",
            "HASH_KGW_POSEIDON2_GPU": "0",
            "HASH_KGW_POSEIDON2_GPU_FUSED": "0",
            "HASH_KGW_POSEIDON2_GPU_MASK_CACHE": "0",
        },
    }
    selected_env = env_by_backend[backend]
    for name, value in selected_env.items():
        os.environ[name] = value

    module = sys.modules.get("watermark_processor")
    if module is not None and hasattr(module, "_POSEIDON2_GPU_DISABLED"):
        setattr(module, "_POSEIDON2_GPU_DISABLED", False)
    return selected_env


def hash_scheme_suffix(args: argparse.Namespace) -> str:
    suffix = args.hash_result_label_suffix.strip()
    if suffix:
        return suffix
    if args.hash_wet_backend != "cpu-u32":
        return args.hash_wet_backend
    return ""


def sample_rows_from_generation(path: str, max_samples: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    return payload.get("samples", [])[:max_samples]


def samples_from_legacy_hash_file(path: Path, max_samples: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows: list[dict[str, Any]] = []
    n = min(
        len(payload.get("input_text", [])),
        len(payload.get("output_without_watermark", [])),
        len(payload.get("output_with_watermark", [])),
    )
    for idx in range(min(n, max_samples)):
        rows.append(
            {
                "id": idx,
                "input_text": payload["input_text"][idx],
                "reference_text_removed": "",
                "output_without_watermark": payload["output_without_watermark"][idx],
                "output_with_watermark": payload["output_with_watermark"][idx],
            }
        )
    return rows


def sample_rows_from_hash_results(base_dir: str, hash_type: int, hash_method: int, max_samples: int) -> list[dict[str, Any]]:
    base = Path(base_dir)
    merged = base / f"num_100_hash_type_{hash_type}_hash_method_{hash_method}.json"
    if merged.exists():
        return samples_from_legacy_hash_file(merged, max_samples)

    rows: list[dict[str, Any]] = []
    part_paths = sorted(base.glob(f"num_100_hash_type_{hash_type}_hash_method_{hash_method}_part_*.json"))
    for part_path in part_paths:
        part_rows = samples_from_legacy_hash_file(part_path, max_samples)
        rows.extend(part_rows)
        if len(rows) >= max_samples:
            break
    for idx, row in enumerate(rows[:max_samples]):
        row["id"] = idx
    return rows[:max_samples]


def token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    if not text:
        return []
    return tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)["input_ids"].squeeze(0).tolist()


def prompt_token_ids(tokenizer: Any, row: dict[str, Any], prompt_max_length: int | None) -> list[int]:
    text_parts = [
        row.get("input_text", ""),
        row.get("reference_text_removed", ""),
        row.get("output_without_watermark", ""),
        row.get("output_with_watermark", ""),
    ]
    prompt_text = next((part for part in text_parts if part), "")
    if prompt_text:
        kwargs = {
            "return_tensors": "pt",
            "add_special_tokens": True,
        }
        if prompt_max_length is not None and prompt_max_length > 0:
            kwargs.update({"truncation": True, "max_length": prompt_max_length})
        ids = tokenizer(prompt_text, **kwargs)["input_ids"].squeeze(0).tolist()
        if ids:
            return ids
    return [0]


def fixed_length_ids(
    tokenizer: Any,
    primary: str,
    fallback_parts: list[str],
    token_count: int,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    ids = token_ids(tokenizer, primary, add_special_tokens=add_special_tokens)
    if len(ids) >= token_count:
        return ids[:token_count]

    for part in fallback_parts:
        if len(ids) >= token_count:
            break
        ids.extend(token_ids(tokenizer, part, add_special_tokens=False))

    if not ids:
        ids = [0]
    while len(ids) < token_count:
        ids.extend(ids[: token_count - len(ids)])
    return ids[:token_count]


def build_benchmark_sequences(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    token_count: int,
    device: str,
    prompt_max_length: int | None = None,
) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for row in rows:
        prompt_ids = prompt_token_ids(tokenizer, row, prompt_max_length)
        continuation_ids = fixed_length_ids(
            tokenizer,
            row.get("output_with_watermark", ""),
            [row.get("reference_text_removed", ""), row.get("input_text", ""), row.get("output_without_watermark", "")],
            token_count,
            add_special_tokens=False,
        )
        full_ids = torch.tensor(prompt_ids + continuation_ids, dtype=torch.long, device=device)
        cont_tensor = torch.tensor(continuation_ids, dtype=torch.long, device=device)
        sequences.append(
            {
                "id": row.get("id"),
                "full_ids": full_ids,
                "prompt_len": len(prompt_ids),
                "continuation_ids": cont_tensor,
                "continuation_text": tokenizer.decode(continuation_ids, skip_special_tokens=True),
            }
        )
    return sequences


def clear_hash_caches() -> None:
    module = sys.modules.get("watermark_processor")
    if module is not None:
        for name in dir(module):
            obj = getattr(module, name)
            cache_clear = getattr(obj, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    poseidon2_gpu = sys.modules.get("baseline_eval.hash_kgw_poseidon2_gpu")
    if poseidon2_gpu is not None:
        clear_mask_cache = getattr(poseidon2_gpu, "clear_mask_cache", None)
        if callable(clear_mask_cache):
            clear_mask_cache()


def clear_processor_caches(processor: Any) -> None:
    cache_clear = getattr(processor, "clear_greenlist_tensor_cache", None)
    if callable(cache_clear):
        cache_clear()


def prefill_processor_wet_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}

    previous_tokens: set[int] = set()
    prefix_calls = 0
    vocab_size = int(getattr(processor, "vocab_size"))
    scores = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for seq in sequences:
            full_ids: torch.Tensor = seq["full_ids"]
            prompt_len = int(seq["prompt_len"])
            for idx in range(token_count):
                scores.zero_()
                prefix = full_ids[: prompt_len + idx]
                previous_tokens.add(int(prefix[-1].detach().item()))
                processor(prefix.unsqueeze(0), scores)
                prefix_calls += 1
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "one-time precompute of the greenlists/masks needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": prefix_calls,
        "unique_previous_tokens": len(previous_tokens),
    }


def collect_previous_tokens_for_wet(sequences: list[dict[str, Any]], token_count: int) -> list[int]:
    previous_tokens: list[int] = []
    for seq in sequences:
        full_ids: torch.Tensor = seq["full_ids"]
        prompt_len = int(seq["prompt_len"])
        for idx in range(token_count):
            previous_tokens.append(int(full_ids[prompt_len + idx - 1].detach().item()))
    return previous_tokens


def cache_processor_greenlist_ids(
    *,
    processor: Any,
    method_name: str,
    hash_type: int,
    cache_backend: str,
    previous_tokens: list[int],
    id_rows: list[torch.Tensor],
    device: str,
) -> None:
    cache = getattr(processor, "_greenlist_tensor_cache", None)
    if cache is None:
        raise ValueError("processor does not expose _greenlist_tensor_cache")
    if len(previous_tokens) != len(id_rows):
        raise ValueError("previous_tokens and id_rows must have the same length")
    device_key = str(torch.device(device))
    for prev_token, ids in zip(previous_tokens, id_rows):
        cache_key = (
            method_name,
            cache_backend,
            int(hash_type),
            int(getattr(processor, "hash_key")),
            float(getattr(processor, "gamma")),
            int(getattr(processor, "vocab_size")),
            int(prev_token),
            device_key,
        )
        cache[cache_key] = ids.to(device=device, dtype=torch.long)
        move_to_end = getattr(cache, "move_to_end", None)
        if callable(move_to_end):
            cache.move_to_end(cache_key)


def mask_rows_to_id_tensors(masks: torch.Tensor) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if masks.ndim != 2:
        raise ValueError(f"expected a 2D mask tensor, got shape {tuple(masks.shape)}")
    id_rows = [torch.nonzero(row, as_tuple=False).flatten().to(dtype=torch.long) for row in masks]
    green_counts = [int(ids.numel()) for ids in id_rows]
    return id_rows, {
        "green_count_mean": mean([float(value) for value in green_counts]),
        "green_count_min": min(green_counts) if green_counts else None,
        "green_count_max": max(green_counts) if green_counts else None,
    }


def id_rows_info(id_rows: list[torch.Tensor]) -> dict[str, Any]:
    green_counts = [int(ids.numel()) for ids in id_rows]
    return {
        "green_count_mean": mean([float(value) for value in green_counts]),
        "green_count_min": min(green_counts) if green_counts else None,
        "green_count_max": max(green_counts) if green_counts else None,
    }


def prefill_rust_id_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}
    import numpy as np
    import watermark_processor as wm

    previous_tokens = collect_previous_tokens_for_wet(sequences, token_count)
    unique_previous = sorted(set(previous_tokens))
    hash_method = wm.HashMethod(int(getattr(processor, "hash_method")))
    hash_type = int(getattr(processor, "hash_type"))
    vocab_size = int(getattr(processor, "vocab_size"))
    hash_key = int(getattr(processor, "hash_key"))
    gamma = float(getattr(processor, "gamma"))
    greenlist_size = int(vocab_size * gamma)

    synchronize(device)
    start = time.perf_counter()
    id_rows: list[torch.Tensor] = []
    if hash_method == wm.HashMethod.TwoToOneFixed:
        for prev_token in unique_previous:
            ids_bytes, _ = wm.invoke_rustlib_get_greenlist_u32_use_two_to_one_hash_and_fixed_threshold_fused_seed(
                hash_key,
                int(prev_token),
                vocab_size,
                gamma,
                wm.HASH_BIG_PRIME_HEX,
                hash_type,
            )
            ids_np = np.frombuffer(ids_bytes, dtype=np.uint32).copy()
            id_rows.append(torch.as_tensor(ids_np.astype(np.int64, copy=False), dtype=torch.long, device=device))
        method_name = "two_to_one_fixed"
    elif hash_method == wm.HashMethod.ThreeToOneFixed:
        for prev_token in unique_previous:
            ids_bytes, _ = wm.invoke_rustlib_get_greenlist_u32_use_three_to_one_hash_and_fixed_threshold(
                hash_key,
                int(prev_token),
                vocab_size,
                gamma,
                wm.HASH_BIG_PRIME_HEX,
                hash_type,
            )
            ids_np = np.frombuffer(ids_bytes, dtype=np.uint32).copy()
            id_rows.append(torch.as_tensor(ids_np.astype(np.int64, copy=False), dtype=torch.long, device=device))
        method_name = "three_to_one_fixed"
    else:
        raise ValueError(f"Rust id prefill does not support hash_method={hash_method}")

    cache_processor_greenlist_ids(
        processor=processor,
        method_name=method_name,
        hash_type=hash_type,
        cache_backend="rust_u32",
        previous_tokens=unique_previous,
        id_rows=id_rows,
        device=device,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "Rust u32 precompute of fixed-threshold green ids needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": len(previous_tokens),
        "unique_previous_tokens": len(unique_previous),
        "batched_prefixes": len(unique_previous),
        "vocab_size": vocab_size,
        "greenlist_size_configured": greenlist_size,
        **id_rows_info(id_rows),
    }


def prefill_poseidon2_gpu_mask_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}
    import watermark_processor as wm
    from baseline_eval import hash_kgw_poseidon2_gpu

    previous_tokens = collect_previous_tokens_for_wet(sequences, token_count)
    unique_previous = sorted(set(previous_tokens))
    hash_method = wm.HashMethod(int(getattr(processor, "hash_method")))
    vocab_size = int(getattr(processor, "vocab_size"))
    hash_key = int(getattr(processor, "hash_key"))
    gamma = float(getattr(processor, "gamma"))

    synchronize(device)
    start = time.perf_counter()
    if hash_method == wm.HashMethod.TwoToOneFixed:
        seed_by_prev = {
            prev_token: int(wm.invoke_rustlib_compute_hash(hash_key, prev_token, 4), 16)
            for prev_token in unique_previous
        }
        cache_info = hash_kgw_poseidon2_gpu.prefill_mask_cache_two_to_one_fixed_native(
            seed_by_prev,
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
    elif hash_method == wm.HashMethod.ThreeToOneFixed:
        cache_info = hash_kgw_poseidon2_gpu.prefill_mask_cache_three_to_one_fixed_native(
            hash_key,
            unique_previous,
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
    else:
        raise ValueError(f"batched Poseidon2 GPU prefill does not support hash_method={hash_method}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "batched native CUDA precompute of Poseidon2 green masks needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": len(previous_tokens),
        "unique_previous_tokens": len(unique_previous),
        "batched_prefixes": cache_info.get("prefixes"),
        "vocab_size": cache_info.get("vocab_size"),
    }


def prefill_poseidon2_gpu_id_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}
    import watermark_processor as wm
    from baseline_eval import hash_kgw_poseidon2_gpu

    previous_tokens = collect_previous_tokens_for_wet(sequences, token_count)
    unique_previous = sorted(set(previous_tokens))
    hash_method = wm.HashMethod(int(getattr(processor, "hash_method")))
    vocab_size = int(getattr(processor, "vocab_size"))
    hash_key = int(getattr(processor, "hash_key"))
    gamma = float(getattr(processor, "gamma"))

    synchronize(device)
    start = time.perf_counter()
    if hash_method == wm.HashMethod.TwoToOneFixed:
        seed_by_prev = {
            prev_token: int(wm.invoke_rustlib_compute_hash(hash_key, prev_token, 4), 16)
            for prev_token in unique_previous
        }
        ordered_items = sorted(seed_by_prev.items())
        masks = hash_kgw_poseidon2_gpu.get_masks_two_to_one_fixed_native(
            [seed for _prev_token, seed in ordered_items],
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="two_to_one_fixed",
            hash_type=4,
            cache_backend="poseidon2_gpu",
            previous_tokens=[prev_token for prev_token, _seed in ordered_items],
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(ordered_items)
    elif hash_method == wm.HashMethod.ThreeToOneFixed:
        masks = hash_kgw_poseidon2_gpu.get_masks_three_to_one_fixed_native(
            hash_key,
            unique_previous,
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="three_to_one_fixed",
            hash_type=4,
            cache_backend="poseidon2_gpu",
            previous_tokens=unique_previous,
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(unique_previous)
    else:
        raise ValueError(f"batched Poseidon2 GPU id prefill does not support hash_method={hash_method}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "batched native CUDA precompute of Poseidon2 green ids needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": len(previous_tokens),
        "unique_previous_tokens": len(unique_previous),
        "batched_prefixes": batched_prefixes,
        "vocab_size": vocab_size,
        **id_info,
    }


def prefill_poseidon_gpu_id_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}
    import watermark_processor as wm
    from baseline_eval import hash_kgw_poseidon2_gpu

    previous_tokens = collect_previous_tokens_for_wet(sequences, token_count)
    unique_previous = sorted(set(previous_tokens))
    hash_method = wm.HashMethod(int(getattr(processor, "hash_method")))
    hash_type = int(getattr(processor, "hash_type"))
    vocab_size = int(getattr(processor, "vocab_size"))
    hash_key = int(getattr(processor, "hash_key"))
    gamma = float(getattr(processor, "gamma"))
    if hash_type != 3:
        raise ValueError(f"Poseidon GPU id prefill supports hash_type=3 only, got hash_type={hash_type}")

    synchronize(device)
    start = time.perf_counter()
    if hash_method == wm.HashMethod.TwoToOneFixed:
        seed_by_prev = {
            prev_token: int(wm.invoke_rustlib_compute_hash(hash_key, prev_token, 3), 16)
            for prev_token in unique_previous
        }
        ordered_items = sorted(seed_by_prev.items())
        masks = hash_kgw_poseidon2_gpu.get_poseidon_fast_masks_two_to_one_fixed_native(
            [seed for _prev_token, seed in ordered_items],
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="two_to_one_fixed",
            hash_type=3,
            cache_backend="rust_u32",
            previous_tokens=[prev_token for prev_token, _seed in ordered_items],
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(ordered_items)
    elif hash_method == wm.HashMethod.ThreeToOneFixed:
        masks = hash_kgw_poseidon2_gpu.get_poseidon_fast_masks_three_to_one_fixed_native(
            hash_key,
            unique_previous,
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="three_to_one_fixed",
            hash_type=3,
            cache_backend="rust_u32",
            previous_tokens=unique_previous,
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(unique_previous)
    else:
        raise ValueError(f"batched Poseidon GPU id prefill does not support hash_method={hash_method}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "batched native CUDA precompute of Poseidon green ids needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": len(previous_tokens),
        "unique_previous_tokens": len(unique_previous),
        "batched_prefixes": batched_prefixes,
        "vocab_size": vocab_size,
        **id_info,
    }


def prefill_mimc_gpu_id_cache(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    device: str,
) -> dict[str, Any]:
    if not sequences:
        return {"elapsed_sec": 0.0, "prefix_calls": 0, "unique_previous_tokens": 0}
    import watermark_processor as wm
    from baseline_eval import hash_kgw_poseidon2_gpu

    previous_tokens = collect_previous_tokens_for_wet(sequences, token_count)
    unique_previous = sorted(set(previous_tokens))
    hash_method = wm.HashMethod(int(getattr(processor, "hash_method")))
    hash_type = int(getattr(processor, "hash_type"))
    vocab_size = int(getattr(processor, "vocab_size"))
    hash_key = int(getattr(processor, "hash_key"))
    gamma = float(getattr(processor, "gamma"))
    if hash_type != 5:
        raise ValueError(f"MiMC GPU id prefill supports hash_type=5 only, got hash_type={hash_type}")

    synchronize(device)
    start = time.perf_counter()
    if hash_method == wm.HashMethod.TwoToOneFixed:
        seed_by_prev = {
            prev_token: int(wm.invoke_rustlib_compute_hash(hash_key, prev_token, 5), 16)
            for prev_token in unique_previous
        }
        ordered_items = sorted(seed_by_prev.items())
        masks = hash_kgw_poseidon2_gpu.get_mimc_masks_two_to_one_fixed_native(
            [seed for _prev_token, seed in ordered_items],
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="two_to_one_fixed",
            hash_type=5,
            cache_backend="rust_u32",
            previous_tokens=[prev_token for prev_token, _seed in ordered_items],
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(ordered_items)
    elif hash_method == wm.HashMethod.ThreeToOneFixed:
        masks = hash_kgw_poseidon2_gpu.get_mimc_masks_three_to_one_fixed_native(
            hash_key,
            unique_previous,
            vocab_size,
            gamma,
            wm.HASH_BIG_PRIME_HEX,
            device,
        )
        id_rows, id_info = mask_rows_to_id_tensors(masks)
        cache_processor_greenlist_ids(
            processor=processor,
            method_name="three_to_one_fixed",
            hash_type=5,
            cache_backend="rust_u32",
            previous_tokens=unique_previous,
            id_rows=id_rows,
            device=device,
        )
        batched_prefixes = len(unique_previous)
    else:
        raise ValueError(f"batched MiMC GPU id prefill does not support hash_method={hash_method}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "definition": "batched native CUDA precompute of MiMC green ids needed by measured WET prefixes",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "prefix_calls": len(previous_tokens),
        "unique_previous_tokens": len(unique_previous),
        "batched_prefixes": batched_prefixes,
        "vocab_size": vocab_size,
        **id_info,
    }


def benchmark_processor_wet(
    *,
    processor: Any,
    sequences: list[dict[str, Any]],
    token_count: int,
    vocab_size: int,
    device: str,
    warmup_samples: int,
    repeat: int,
    clear_cache_fn: Callable[[], None] | None = None,
    per_repeat_setup: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if not sequences:
        return {"error": "no sequences"}

    score_pool = [torch.zeros((1, vocab_size), dtype=torch.float32, device=device) for _ in range(token_count)]

    def run_one_sequence(seq: dict[str, Any], measured: bool) -> tuple[float, int]:
        full_ids: torch.Tensor = seq["full_ids"]
        prompt_len = int(seq["prompt_len"])
        for scores in score_pool:
            scores.zero_()
        synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            for idx in range(token_count):
                prefix = full_ids[: prompt_len + idx].unsqueeze(0)
                processor(prefix, score_pool[idx])
        synchronize(device)
        elapsed = time.perf_counter() - start
        return elapsed, token_count

    warmup_n = min(warmup_samples, len(sequences))
    if warmup_n > 0:
        for seq in sequences[:warmup_n]:
            run_one_sequence(seq, measured=False)

    sample_times: list[float] = []
    total_elapsed = 0.0
    total_tokens = 0
    for _ in range(repeat):
        if clear_cache_fn:
            clear_cache_fn()
            clear_processor_caches(processor)
        if per_repeat_setup:
            per_repeat_setup()
        for seq in sequences:
            elapsed, tokens = run_one_sequence(seq, measured=True)
            sample_times.append(elapsed)
            total_elapsed += elapsed
            total_tokens += tokens

    per_token_ms = [(elapsed / token_count) * 1000.0 for elapsed in sample_times]
    return {
        "definition": (
            f"processor-only watermark embedding time over {token_count} generation steps; "
            "excludes LLM forward pass and sampling"
        ),
        "num_samples": len(sequences),
        "repeat": repeat,
        "token_count": token_count,
        "total_tokens": total_tokens,
        "total_elapsed_sec": total_elapsed,
        "aggregate_ms_per_token": (total_elapsed / total_tokens) * 1000.0 if total_tokens else None,
        "per_sample_ms_per_token": stats_values(per_token_ms),
        "wet_total_ms_from_aggregate": (total_elapsed / total_tokens) * token_count * 1000.0 if total_tokens else None,
        "wet_200_total_ms_from_aggregate": (total_elapsed / total_tokens) * token_count * 1000.0 if total_tokens else None,
    }


def benchmark_kgw_detector(
    *,
    detector: Any,
    sequences: list[dict[str, Any]],
    token_counts: list[int],
    device: str,
    warmup_samples: int,
    repeat: int,
    clear_cache_before_each: Callable[[], None] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for token_count in token_counts:
        token_id_inputs = [seq["continuation_ids"][:token_count] for seq in sequences]
        text_inputs = [detector.tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True) for ids in token_id_inputs]

        for idx in range(min(warmup_samples, len(sequences))):
            detector.detect(text=text_inputs[idx])
            detector.detect(tokenized_text=token_id_inputs[idx])
        synchronize(device)

        raw_times: list[float] = []
        ids_times: list[float] = []
        for _ in range(repeat):
            for text in text_inputs:
                if clear_cache_before_each:
                    clear_cache_before_each()
                synchronize(device)
                start = time.perf_counter()
                detector.detect(text=text)
                synchronize(device)
                raw_times.append(time.perf_counter() - start)
            for ids in token_id_inputs:
                if clear_cache_before_each:
                    clear_cache_before_each()
                synchronize(device)
                start = time.perf_counter()
                detector.detect(tokenized_text=ids)
                synchronize(device)
                ids_times.append(time.perf_counter() - start)

        results[f"wdt_{token_count}"] = {
            "definition": f"detection wall-clock time for one {token_count}-token continuation",
            "num_samples": len(sequences),
            "repeat": repeat,
            "raw_text_including_tokenizer_ms": stats_ms(raw_times),
            "token_ids_detector_only_ms": stats_ms(ids_times),
        }
    return results


def benchmark_upv_detector(
    *,
    detector: UpvNetworkDetector,
    sequences: list[dict[str, Any]],
    token_counts: list[int],
    warmup_samples: int,
    repeat: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    device = detector.device
    for token_count in token_counts:
        token_id_inputs = [seq["continuation_ids"][:token_count].detach().cpu() for seq in sequences]
        text_inputs = [detector.tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in token_id_inputs]

        for idx in range(min(warmup_samples, len(sequences))):
            detector.score_text(text_inputs[idx])
            detector.score_ids(token_id_inputs[idx])
        synchronize(device)

        raw_times: list[float] = []
        ids_times: list[float] = []
        for _ in range(repeat):
            for text in text_inputs:
                synchronize(device)
                start = time.perf_counter()
                detector.score_text(text)
                synchronize(device)
                raw_times.append(time.perf_counter() - start)
            for ids in token_id_inputs:
                synchronize(device)
                start = time.perf_counter()
                detector.score_ids(ids)
                synchronize(device)
                ids_times.append(time.perf_counter() - start)

        results[f"wdt_{token_count}"] = {
            "definition": f"UPV network detector wall-clock time for one {token_count}-token continuation",
            "num_samples": len(sequences),
            "repeat": repeat,
            "raw_text_including_tokenizer_ms": stats_ms(raw_times),
            "token_ids_detector_only_ms": stats_ms(ids_times),
        }
    return results


def summarize_generation_log_efficiency(path: str, token_counts: list[int] | tuple[int, ...] = (50, 200)) -> dict[str, Any]:
    payload = read_json(path)
    samples = payload.get("samples", [])
    wm_times = [finite_float(s.get("generation_time_with_watermark_sec")) for s in samples]
    plain_times = [finite_float(s.get("generation_time_without_watermark_sec")) for s in samples]
    wm_tokens = [finite_float(s.get("token_count_with_watermark")) for s in samples]
    plain_tokens = [finite_float(s.get("token_count_without_watermark")) for s in samples]
    det_wm_times = [finite_float(s.get("detection_time_with_watermark_sec")) for s in samples]
    det_plain_times = [finite_float(s.get("detection_time_without_watermark_sec")) for s in samples]

    rows = [
        (wm_t, plain_t, wm_n, plain_n, det_wm, det_plain)
        for wm_t, plain_t, wm_n, plain_n, det_wm, det_plain in zip(
            wm_times, plain_times, wm_tokens, plain_tokens, det_wm_times, det_plain_times
        )
        if wm_t is not None and plain_t is not None and wm_n and plain_n and det_wm is not None and det_plain is not None
    ]
    wm_per_token = [(wm_t / wm_n) * 1000.0 for wm_t, _plain_t, wm_n, _plain_n, _det_wm, _det_plain in rows]
    plain_per_token = [(_plain_t / _plain_n) * 1000.0 for _wm_t, _plain_t, _wm_n, _plain_n, _det_wm, _det_plain in rows]
    overhead_per_token = [((wm_t - plain_t) / wm_n) * 1000.0 for wm_t, plain_t, wm_n, _plain_n, _det_wm, _det_plain in rows]
    wm_det_ms = [det_wm * 1000.0 for _wm_t, _plain_t, _wm_n, _plain_n, det_wm, _det_plain in rows]
    plain_det_ms = [det_plain * 1000.0 for _wm_t, _plain_t, _wm_n, _plain_n, _det_wm, det_plain in rows]

    total_wm_time = sum(wm_t for wm_t, _plain_t, _wm_n, _plain_n, _det_wm, _det_plain in rows)
    total_plain_time = sum(plain_t for _wm_t, plain_t, _wm_n, _plain_n, _det_wm, _det_plain in rows)
    total_wm_tokens = sum(wm_n for _wm_t, _plain_t, wm_n, _plain_n, _det_wm, _det_plain in rows)
    total_plain_tokens = sum(plain_n for _wm_t, _plain_t, _wm_n, plain_n, _det_wm, _det_plain in rows)

    normalized_wdt: dict[str, Any] = {}
    for token_count in token_counts:
        normalized_wdt[f"wdt_{token_count}"] = {
            "watermarked_normalized_ms": stats_values([(det_wm / wm_n) * token_count * 1000.0 for wm_t, plain_t, wm_n, plain_n, det_wm, det_plain in rows]),
            "plain_normalized_ms": stats_values([(det_plain / plain_n) * token_count * 1000.0 for wm_t, plain_t, wm_n, plain_n, det_wm, det_plain in rows]),
        }

    return {
        "source_generations": path,
        "num_samples": len(rows),
        "avg_token_count_with_watermark": mean([wm_n for _wm_t, _plain_t, wm_n, _plain_n, _det_wm, _det_plain in rows]),
        "avg_token_count_without_watermark": mean([plain_n for _wm_t, _plain_t, _wm_n, plain_n, _det_wm, _det_plain in rows]),
        "watermarked_generation_ms_per_token": {
            "per_sample": stats_values(wm_per_token),
            "aggregate": (total_wm_time / total_wm_tokens) * 1000.0 if total_wm_tokens else None,
        },
        "plain_generation_ms_per_token": {
            "per_sample": stats_values(plain_per_token),
            "aggregate": (total_plain_time / total_plain_tokens) * 1000.0 if total_plain_tokens else None,
        },
        "end_to_end_overhead_ms_per_token": {
            "definition": "(watermarked generation time - plain generation time) / watermarked token count",
            "per_sample": stats_values(overhead_per_token),
            "aggregate": ((total_wm_time - total_plain_time) / total_wm_tokens) * 1000.0 if total_wm_tokens else None,
        },
        "full_text_detection_ms": {
            "watermarked": stats_values(wm_det_ms),
            "plain": stats_values(plain_det_ms),
        },
        "length_normalized_detection": normalized_wdt,
    }


def make_original_kgw(tokenizer: Any, args: argparse.Namespace, device: str) -> tuple[Any, Any]:
    module = importlib.import_module("watermark_processor_org_scheme")
    kwargs = {
        "vocab": list(tokenizer.get_vocab().values()),
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "hash_key": args.original_hash_key,
        "select_green_tokens": True,
    }
    processor = module.WatermarkLogitsProcessor(**kwargs)
    detector = module.WatermarkDetector(
        **kwargs,
        device=torch.device(device),
        tokenizer=tokenizer,
        z_threshold=args.z_threshold,
        normalizers=[],
        ignore_repeated_bigrams=False,
    )
    return processor, detector


def make_hash_kgw(tokenizer: Any, args: argparse.Namespace, device: str, hash_type: int, hash_method: int) -> tuple[Any, Any]:
    module = importlib.import_module("watermark_processor")
    kwargs = {
        "vocab": list(tokenizer.get_vocab().values()),
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "hash_key": args.hash_kgw_hash_key,
        "select_green_tokens": True,
        "hash_type": hash_type,
        "hash_method": hash_method,
    }
    processor = module.WatermarkLogitsProcessor(**kwargs)
    detector = module.WatermarkDetector(
        **kwargs,
        device=torch.device(device),
        tokenizer=tokenizer,
        z_threshold=args.z_threshold,
        normalizers=[],
        ignore_repeated_bigrams=False,
    )
    return processor, detector


def make_upv_processor(args: argparse.Namespace, vocab_size: int) -> tuple[Any, Callable[[], None]]:
    from baseline_eval.common import add_repo_to_path

    add_repo_to_path(args.upv_root)
    from model_key import get_model
    from watermark_model import WatermarkLogitsProcessor

    model = get_model(16, 5, None, 5)
    state_dict = torch.load(args.upv_generator_model, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    cache: dict[Any, Any] = {}
    processor = WatermarkLogitsProcessor(
        vocab=list(range(vocab_size)),
        delta=2.0,
        model=model,
        window_size=5,
        cache=cache,
        bit_number=16,
        beam_size=0,
        llm_name="opt-1.3b",
    )

    def clear_cache() -> None:
        cache.clear()

    return processor, clear_cache


def format_float(value: Any, digits: int = 4) -> str:
    value = finite_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def flatten_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wet_token_count = results.get("metadata", {}).get("wet_token_count", 200)
    prefill_hash_wet_cache = bool(results.get("metadata", {}).get("prefill_hash_wet_cache", False))
    for name, result in results.get("schemes", {}).items():
        wet = result.get("wet_processor", {})
        detection = result.get("detection", {})
        log_eff = result.get("generation_log_efficiency", {})
        pdw_eff = result.get("pdw_full_signature_efficiency", {})
        prefill = result.get("wet_cache_prefill", {})

        row = {
            "scheme": name,
            "category": result.get("category", ""),
            "hash_wet_backend": result.get("hash_wet_backend", ""),
            "wet_processor_ms_per_token": wet.get("aggregate_ms_per_token"),
            "wet_200_total_ms": wet.get("wet_total_ms_from_aggregate", wet.get("wet_200_total_ms_from_aggregate")),
            "wet_cache_prefill_ms": prefill.get("elapsed_ms"),
            "wet_cache_prefill_prefix_calls": prefill.get("prefix_calls"),
            "wet_cache_prefill_unique_previous_tokens": prefill.get("unique_previous_tokens"),
            "wet_amortized_ms_per_token": None,
            "wet_amortized_200_total_ms": None,
            "log_end_to_end_overhead_ms_per_token": log_eff.get("end_to_end_overhead_ms_per_token", {}).get("aggregate"),
            "pdw_full_generation_ms_per_token": pdw_eff.get("watermarked_generation_ms_per_token", {}).get("aggregate"),
            "pdw_full_wm_detection_ms": pdw_eff.get("full_text_detection_ms", {}).get("watermarked", {}).get("mean"),
            "pdw_full_plain_detection_ms": pdw_eff.get("full_text_detection_ms", {}).get("plain", {}).get("mean"),
        }
        wet_total = finite_float(row["wet_200_total_ms"])
        prefill_ms = finite_float(row["wet_cache_prefill_ms"])
        prefill_prefix_calls = finite_float(row["wet_cache_prefill_prefix_calls"])
        if wet_total is not None and prefill_ms is not None and prefill_prefix_calls:
            row["wet_amortized_200_total_ms"] = wet_total + (prefill_ms / prefill_prefix_calls) * wet_token_count
            row["wet_amortized_ms_per_token"] = row["wet_amortized_200_total_ms"] / wet_token_count
        if result.get("category") == "pdw":
            pdw_wet = pdw_eff.get("watermarked_generation_ms_per_token", {}).get("aggregate")
            row["reported_wet_ms_per_token"] = pdw_wet
            row["reported_wet_200_total_ms"] = pdw_wet * wet_token_count if pdw_wet is not None else None
            row["reported_efficiency_scope"] = "PDW full-signature generation; WDT is length-normalized watermarked detection"
            row["reported_avg_wm_tokens"] = pdw_eff.get("avg_token_count_with_watermark")
            row["reported_num_samples"] = pdw_eff.get("num_samples")
        else:
            row["reported_wet_ms_per_token"] = wet.get("aggregate_ms_per_token")
            row["reported_wet_200_total_ms"] = wet.get(
                "wet_total_ms_from_aggregate", wet.get("wet_200_total_ms_from_aggregate")
            )
            if result.get("category") == "hash_kgw" and prefill_hash_wet_cache:
                row["reported_efficiency_scope"] = (
                    f"{wet_token_count}-step processor WET after cache prefill using {row['hash_wet_backend']}; "
                    "prefill excluded and reported separately"
                )
            elif result.get("category") == "hash_kgw":
                row["reported_efficiency_scope"] = (
                    f"{wet_token_count}-step processor WET using {row['hash_wet_backend']}; token-id detector WDT"
                )
            else:
                row["reported_efficiency_scope"] = f"{wet_token_count}-step processor WET; token-id detector WDT"
            row["reported_avg_wm_tokens"] = None
            row["reported_num_samples"] = wet.get("num_samples") or detection.get("wdt_200", {}).get("num_samples")
        for token_count in results["metadata"]["wdt_token_counts"]:
            key = f"wdt_{token_count}"
            row[f"wdt_{token_count}_raw_ms"] = detection.get(key, {}).get("raw_text_including_tokenizer_ms", {}).get("mean")
            row[f"wdt_{token_count}_ids_ms"] = detection.get(key, {}).get("token_ids_detector_only_ms", {}).get("mean")
            row[f"pdw_wm_wdt_{token_count}_normalized_ms"] = (
                pdw_eff.get("length_normalized_detection", {})
                .get(key, {})
                .get("watermarked_normalized_ms", {})
                .get("mean")
            )
            row[f"pdw_plain_wdt_{token_count}_normalized_ms"] = (
                pdw_eff.get("length_normalized_detection", {})
                .get(key, {})
                .get("plain_normalized_ms", {})
                .get("mean")
            )
            if result.get("category") == "pdw":
                row[f"reported_wdt_{token_count}_ms"] = row[f"pdw_wm_wdt_{token_count}_normalized_ms"]
            else:
                row[f"reported_wdt_{token_count}_ms"] = row[f"wdt_{token_count}_ids_ms"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, results: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    token_counts = results["metadata"]["wdt_token_counts"]
    wet_token_count = results["metadata"].get("wet_token_count", 200)
    lines = [
        "# Efficiency Benchmark Results",
        "",
        f"Created at: {results['metadata']['created_at']}",
        "",
        "## Method",
        "",
        f"- KGW/hash-based KGW WET is processor-only embedding time over a {wet_token_count}-step run; LLM forward pass is excluded.",
        "- KGW/hash-based KGW WDT is measured on fixed-length decoded continuations; raw-text WDT includes tokenizer time and token-id WDT isolates detector logic.",
        "- UPV reports network-detector WDT and processor-only generator overhead when available; generation-log overhead is also retained.",
        "- PDW is full-signature generation, not a fixed-token-budget method. Its reported WET is full-signature generation ms/token, and its reported WDT values are length-normalized watermarked detection times.",
        "",
        "## Main Table",
        "",
    ]
    headers = [
        "Scheme",
        "Efficiency scope",
        "Reported WET ms/token",
        f"Reported WET-{wet_token_count}/equiv ms",
        "WET+prefill amortized ms/token",
        "WET cache prefill ms",
        "Prefill unique prev tokens",
    ]
    for count in token_counts:
        headers.append(f"Reported WDT-{count} ms")
    headers.extend(["Samples", "Avg WM tokens", "PDW full WM detect ms", "PDW full plain detect ms"])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        vals = [
            row["scheme"],
            row.get("reported_efficiency_scope", ""),
            format_float(row.get("reported_wet_ms_per_token")),
            format_float(row.get("reported_wet_200_total_ms")),
            format_float(row.get("wet_amortized_ms_per_token")),
            format_float(row.get("wet_cache_prefill_ms")),
            format_float(row.get("wet_cache_prefill_unique_previous_tokens"), digits=0),
        ]
        for count in token_counts:
            vals.append(format_float(row.get(f"reported_wdt_{count}_ms")))
        vals.extend(
            [
                format_float(row.get("reported_num_samples"), digits=0),
                format_float(row.get("reported_avg_wm_tokens")),
                format_float(row.get("pdw_full_wm_detection_ms")),
                format_float(row.get("pdw_full_plain_detection_ms")),
            ]
        )
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- For KGW/hash KGW/UPV, reported WET is processor-only embedding overhead over a {wet_token_count}-step run.",
            "- If hash WET cache prefill is enabled, the reported hash-KGW WET excludes that precompute; the prefill cost is shown separately.",
            f"- For PDW, reported WET is full-signature watermarked generation time divided by the actual generated token count; the WET-{wet_token_count}/equiv value is length-normalized and is not an actual {wet_token_count}-token PDW run.",
            "- `Log overhead ms/token` comes from existing generation logs and includes the difference between watermarked and plain generation wall-clock time.",
            "- For PDW, the full positive/watermarked detection time and full plain detection time are also shown because PDW plain-text rejection is much slower than positive detection.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(
    path: Path,
    markdown_path: Path,
    rows: list[dict[str, Any]],
    token_counts: list[int],
    wet_token_count: int = 200,
) -> None:
    headers = [
        "Scheme",
        "Efficiency scope",
        "Reported WET ms/token",
        f"Reported WET-{wet_token_count}/equiv ms",
        "WET+prefill amortized ms/token",
        "WET cache prefill ms",
        "Prefill unique prev tokens",
    ]
    for count in token_counts:
        headers.append(f"Reported WDT-{count} ms")
    headers.extend(["Samples", "Avg WM tokens", "PDW full WM detect ms", "PDW full plain detect ms"])

    body_rows = []
    for row in rows:
        vals = [
            row["scheme"],
            row.get("reported_efficiency_scope", ""),
            format_float(row.get("reported_wet_ms_per_token")),
            format_float(row.get("reported_wet_200_total_ms")),
            format_float(row.get("wet_amortized_ms_per_token")),
            format_float(row.get("wet_cache_prefill_ms")),
            format_float(row.get("wet_cache_prefill_unique_previous_tokens"), digits=0),
        ]
        for count in token_counts:
            vals.append(format_float(row.get(f"reported_wdt_{count}_ms")))
        vals.extend(
            [
                format_float(row.get("reported_num_samples"), digits=0),
                format_float(row.get("reported_avg_wm_tokens")),
                format_float(row.get("pdw_full_wm_detection_ms")),
                format_float(row.get("pdw_full_plain_detection_ms")),
            ]
        )
        body_rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in vals) + "</tr>")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Efficiency Benchmark Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; margin: 32px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Efficiency Benchmark Results</h1>
  <p>Companion markdown: <code>{html.escape(str(markdown_path))}</code></p>
  <p>KGW/hash KGW/UPV WET is processor-only. PDW WET uses full-signature generation logs; PDW WDT values in the main columns are length-normalized watermarked detection times.</p>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device, require_cuda=args.require_cuda)
    token_counts = parse_token_counts(args.wdt_token_counts)
    selected_hash_types = parse_int_list(args.hash_types)
    selected_hash_methods = parse_int_list(args.hash_methods)
    selected_schemes = parse_csv_set(args.schemes)
    valid_schemes = {"original", "hash", "upv", "pdw"}
    unknown_schemes = sorted(selected_schemes - valid_schemes)
    if unknown_schemes:
        raise SystemExit(f"Unknown --schemes values: {unknown_schemes}; valid values are {sorted(valid_schemes)}")
    unknown_hash_types = [value for value in selected_hash_types if value not in HASH_TYPES]
    if unknown_hash_types:
        raise SystemExit(f"Unknown --hash-types values: {unknown_hash_types}; valid values are {sorted(HASH_TYPES)}")
    unknown_hash_methods = [value for value in selected_hash_methods if value not in HASH_METHODS]
    if unknown_hash_methods:
        raise SystemExit(f"Unknown --hash-methods values: {unknown_hash_methods}; valid fixed values are {sorted(HASH_METHODS)}")
    wet_max_samples = args.wet_max_samples if args.wet_max_samples is not None else args.max_samples
    wdt_max_samples = args.wdt_max_samples if args.wdt_max_samples is not None else args.max_samples
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    vocab_size = len(list(tokenizer.get_vocab().values()))

    results: dict[str, Any] = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "device": device,
            "requested_device": args.device,
            "require_cuda": bool(args.require_cuda),
            "max_samples": args.max_samples,
            "wet_max_samples": wet_max_samples,
            "wdt_max_samples": wdt_max_samples,
            "wet_token_count": args.wet_token_count,
            "wdt_token_counts": token_counts,
            "prompt_max_length": args.prompt_max_length,
            "warmup_samples": args.warmup_samples,
            "repeat": args.repeat,
            "gamma": args.gamma,
            "delta": args.delta,
            "z_threshold": args.z_threshold,
            "hash_types": selected_hash_types,
            "hash_methods": selected_hash_methods,
            "hash_wet_backend": args.hash_wet_backend,
            "hash_backend_env": {},
            "prefill_hash_wet_cache": bool(args.prefill_hash_wet_cache),
            "schemes": sorted(selected_schemes),
        },
        "schemes": {},
    }
    if "hash" in selected_schemes:
        results["metadata"]["hash_backend_env"] = configure_hash_wet_backend(args.hash_wet_backend, device)

    if "original" in selected_schemes:
        kgw_wet_rows = sample_rows_from_generation(args.kgw_generations, wet_max_samples)
        kgw_wdt_rows = sample_rows_from_generation(args.kgw_generations, wdt_max_samples)
        kgw_wet_sequences = build_benchmark_sequences(
            tokenizer, kgw_wet_rows, args.wet_token_count, device, args.prompt_max_length
        )
        kgw_wdt_sequences = build_benchmark_sequences(
            tokenizer, kgw_wdt_rows, max(token_counts), device, args.prompt_max_length
        )
        processor, detector = make_original_kgw(tokenizer, args, device)
        print("Benchmarking Original KGW", flush=True)
        results["schemes"]["Original KGW"] = {
            "category": "kgw",
            "source": args.kgw_generations,
            "wet_processor": benchmark_processor_wet(
                processor=processor,
                sequences=kgw_wet_sequences,
                token_count=args.wet_token_count,
                vocab_size=vocab_size,
                device=device,
                warmup_samples=args.warmup_samples,
                repeat=args.repeat,
            ),
            "detection": benchmark_kgw_detector(
                detector=detector,
                sequences=kgw_wdt_sequences,
                token_counts=token_counts,
                device=device,
                warmup_samples=args.warmup_samples,
                repeat=args.repeat,
            ),
        }

    if "hash" in selected_schemes:
        label_suffix = hash_scheme_suffix(args)
        for hash_type in selected_hash_types:
            hash_name = HASH_TYPES[hash_type]
            for hash_method in selected_hash_methods:
                method_name = HASH_METHODS[hash_method]
                scheme_name = f"Hash KGW {hash_name} {method_name}"
                if label_suffix:
                    scheme_name = f"{scheme_name} [{label_suffix}]"
                print(f"Benchmarking {scheme_name}", flush=True)
                wet_rows = sample_rows_from_hash_results(args.hash_results_dir, hash_type, hash_method, wet_max_samples)
                wdt_rows = sample_rows_from_hash_results(args.hash_results_dir, hash_type, hash_method, wdt_max_samples)
                wet_sequences = build_benchmark_sequences(
                    tokenizer, wet_rows, args.wet_token_count, device, args.prompt_max_length
                )
                wdt_sequences = build_benchmark_sequences(
                    tokenizer, wdt_rows, max(token_counts), device, args.prompt_max_length
                )
                if not wet_sequences or not wdt_sequences:
                    results["schemes"][scheme_name] = {
                        "category": "hash_kgw",
                        "hash_type": hash_type,
                        "hash_method": hash_method,
                        "error": "no source samples found",
                    }
                    continue
                processor, detector = make_hash_kgw(tokenizer, args, device, hash_type, hash_method)
                hash_result = {
                    "category": "hash_kgw",
                    "hash_type": hash_type,
                    "hash_method": hash_method,
                    "hash_wet_backend": args.hash_wet_backend,
                    "hash_backend_env": dict(results["metadata"].get("hash_backend_env", {})),
                    "source": args.hash_results_dir,
                }
                hash_clear_cache_fn = clear_hash_caches
                if args.prefill_hash_wet_cache:
                    clear_hash_caches()
                    clear_processor_caches(processor)
                    if args.hash_wet_backend == "rust-id-cache":
                        hash_result["wet_cache_prefill"] = prefill_rust_id_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    elif hash_type == 4 and args.hash_wet_backend == "poseidon2-gpu-id-cache":
                        hash_result["wet_cache_prefill"] = prefill_poseidon2_gpu_id_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    elif hash_type == 3 and args.hash_wet_backend == "poseidon-gpu-id-cache":
                        hash_result["wet_cache_prefill"] = prefill_poseidon_gpu_id_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    elif hash_type == 5 and args.hash_wet_backend == "mimc-gpu-id-cache":
                        hash_result["wet_cache_prefill"] = prefill_mimc_gpu_id_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    elif hash_type == 4 and args.hash_wet_backend == "poseidon2-gpu-mask-cache":
                        hash_result["wet_cache_prefill"] = prefill_poseidon2_gpu_mask_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    else:
                        hash_result["wet_cache_prefill"] = prefill_processor_wet_cache(
                            processor=processor,
                            sequences=wet_sequences,
                            token_count=args.wet_token_count,
                            device=device,
                        )
                    hash_clear_cache_fn = None
                hash_result["wet_processor"] = benchmark_processor_wet(
                    processor=processor,
                    sequences=wet_sequences,
                    token_count=args.wet_token_count,
                    vocab_size=vocab_size,
                    device=device,
                    warmup_samples=args.warmup_samples,
                    repeat=args.repeat,
                    clear_cache_fn=hash_clear_cache_fn,
                )
                hash_result["detection"] = benchmark_kgw_detector(
                    detector=detector,
                    sequences=wdt_sequences,
                    token_counts=token_counts,
                    device=device,
                    warmup_samples=args.warmup_samples,
                    repeat=args.repeat,
                    clear_cache_before_each=clear_hash_caches,
                )
                results["schemes"][scheme_name] = hash_result

    if "upv" in selected_schemes:
        print("Benchmarking UPV network detector", flush=True)
        upv_payload = read_json(args.upv_generations)
        upv_wet_rows = upv_payload.get("samples", [])[:wet_max_samples]
        upv_wdt_rows = upv_payload.get("samples", [])[:wdt_max_samples]
        upv_wet_sequences = build_benchmark_sequences(
            tokenizer, upv_wet_rows, args.wet_token_count, device, args.prompt_max_length
        )
        upv_wdt_sequences = build_benchmark_sequences(
            tokenizer, upv_wdt_rows, max(token_counts), device, args.prompt_max_length
        )
        upv_meta = upv_payload["metadata"]["scheme_metadata"]
        upv_run_args = upv_payload["metadata"]["args"]
        upv_detector = UpvNetworkDetector(
            upv_root=args.upv_root,
            detector_model=upv_meta["network_detector_model"],
            tokenizer_path=upv_run_args["model"],
            bit_number=upv_meta["bit_number"],
            layers=upv_meta["layers"],
            fixed_length=upv_meta.get("detector_fixed_length", 200),
            threshold=upv_meta.get("network_threshold", 0.5),
            device=device,
        )
        upv_result: dict[str, Any] = {
            "category": "upv_network",
            "source": args.upv_generations,
            "generation_log_efficiency": summarize_generation_log_efficiency(args.upv_generations, token_counts),
            "detection": benchmark_upv_detector(
                detector=upv_detector,
                sequences=upv_wdt_sequences,
                token_counts=token_counts,
                warmup_samples=args.warmup_samples,
                repeat=args.repeat,
            ),
        }
        if not args.skip_upv_processor_wet:
            upv_processor, clear_upv_cache = make_upv_processor(args, vocab_size)
            upv_result["wet_processor"] = benchmark_processor_wet(
                processor=upv_processor,
                sequences=upv_wet_sequences,
                token_count=args.wet_token_count,
                vocab_size=vocab_size,
                device=device,
                warmup_samples=args.warmup_samples,
                repeat=args.repeat,
                per_repeat_setup=clear_upv_cache,
            )
        results["schemes"]["UPV network"] = upv_result

    if "pdw" in selected_schemes:
        print("Summarizing PDW full-signature efficiency", flush=True)
        results["schemes"]["PDW publicly-detectable"] = {
            "category": "pdw",
            "source": args.pdw_generations,
            "pdw_full_signature_efficiency": summarize_generation_log_efficiency(args.pdw_generations, token_counts),
            "note": "PDW output length is full-signature driven and is not a 200-token generation-budget result.",
        }

    rows = flatten_rows(results)
    json_path = out_dir / "efficiency_results.json"
    csv_path = out_dir / "efficiency_results.csv"
    md_path = out_dir / "efficiency_results.md"
    html_path = out_dir / "efficiency_results.html"
    write_json(json_path, results)
    write_csv(csv_path, rows)
    write_markdown(md_path, results, rows)
    write_html(html_path, md_path, rows, token_counts, args.wet_token_count)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {html_path}", flush=True)


if __name__ == "__main__":
    main()
