from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import hash_rustlib
import torch

from watermark_processor import HASH_BIG_PRIME_HEX, invoke_rustlib_compute_hash

from baseline_eval import hash_kgw_poseidon2_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict Poseidon2 GPU benchmark for hash-based KGW.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--vocab-size", type=int, default=50272)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--hash-key", type=int, default=2023)
    parser.add_argument("--previous-token", type=int, default=123)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--backend", choices=["numba", "torch", "native"], default="native")
    parser.add_argument("--skip-equivalence", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def stats_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def cpu_two_to_one(seed: int, vocab_size: int, gamma: float) -> list[int]:
    greenlist, _ = hash_rustlib.rayon_get_greenlist_id_and_fixed_threshold_use_multi_two_inputs_hash(
        str(seed),
        int(vocab_size),
        int(vocab_size * gamma),
        float(gamma),
        HASH_BIG_PRIME_HEX,
        4,
    )
    return list(greenlist)


def cpu_three_to_one(hash_key: int, previous_token: int, vocab_size: int, gamma: float) -> list[int]:
    greenlist, _ = hash_rustlib.rayon_get_greenlist_id_and_fixed_threshold_use_multi_three_inputs_hash(
        int(hash_key),
        int(previous_token),
        int(vocab_size),
        int(vocab_size * gamma),
        float(gamma),
        HASH_BIG_PRIME_HEX,
        4,
    )
    return list(greenlist)


def actual_two_to_one_seed(hash_key: int, previous_token: int) -> int:
    return int(invoke_rustlib_compute_hash(int(hash_key), int(previous_token), 4), 16)


def as_list(values: Any) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def bench_call(device: str, repeat: int, warmup: int, fn: Callable[[], Any]) -> tuple[list[float], Any]:
    last_result = None
    for _ in range(warmup):
        last_result = fn()
    synchronize(device)
    times: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        last_result = fn()
        synchronize(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times, last_result


def greenlist_two_to_one(backend: str, seed: int, vocab_size: int, gamma: float, device: str):
    if backend == "native":
        return hash_kgw_poseidon2_gpu.get_greenlist_ids_two_to_one_fixed_native(
            seed, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
        )
    if backend == "torch":
        return hash_kgw_poseidon2_gpu.get_greenlist_ids_two_to_one_fixed_torch(
            seed, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
        )
    return hash_kgw_poseidon2_gpu.get_greenlist_ids_two_to_one_fixed(
        seed, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
    )


def greenlist_three_to_one(backend: str, hash_key: int, previous_token: int, vocab_size: int, gamma: float, device: str):
    if backend == "native":
        return hash_kgw_poseidon2_gpu.get_greenlist_ids_three_to_one_fixed_native(
            hash_key, previous_token, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
        )
    if backend == "torch":
        return hash_kgw_poseidon2_gpu.get_greenlist_ids_three_to_one_fixed_torch(
            hash_key, previous_token, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
        )
    return hash_kgw_poseidon2_gpu.get_greenlist_ids_three_to_one_fixed(
        hash_key, previous_token, vocab_size, gamma, HASH_BIG_PRIME_HEX, device
    )


def bias_two_to_one(backend: str, seed: int, scores: torch.Tensor, gamma: float, device: str) -> None:
    scores.zero_()
    if backend == "native":
        hash_kgw_poseidon2_gpu.bias_logits_two_to_one_fixed_native(seed, scores, 1.0, gamma, HASH_BIG_PRIME_HEX, device)
    else:
        hash_kgw_poseidon2_gpu.bias_logits_two_to_one_fixed_torch(seed, scores, 1.0, gamma, HASH_BIG_PRIME_HEX, device)


def bias_three_to_one(
    backend: str,
    hash_key: int,
    previous_token: int,
    scores: torch.Tensor,
    gamma: float,
    device: str,
) -> None:
    scores.zero_()
    if backend == "native":
        hash_kgw_poseidon2_gpu.bias_logits_three_to_one_fixed_native(
            hash_key, previous_token, scores, 1.0, gamma, HASH_BIG_PRIME_HEX, device
        )
    else:
        hash_kgw_poseidon2_gpu.bias_logits_three_to_one_fixed_torch(
            hash_key, previous_token, scores, 1.0, gamma, HASH_BIG_PRIME_HEX, device
        )


def assert_equal(name: str, cpu: list[int], gpu: list[int]) -> None:
    if cpu != gpu:
        cpu_only = sorted(set(cpu) - set(gpu))[:10]
        gpu_only = sorted(set(gpu) - set(cpu))[:10]
        raise AssertionError(
            f"{name} mismatch: cpu_len={len(cpu)} gpu_len={len(gpu)} cpu_only={cpu_only} gpu_only={gpu_only}"
        )


def main() -> None:
    args = parse_args()
    hash_kgw_poseidon2_gpu.select_device(args.device)
    torch.cuda.set_device(args.device)
    if not hash_kgw_poseidon2_gpu.is_available():
        raise RuntimeError("CUDA is not available for the strict Poseidon2 GPU backend")
    if args.backend == "native" and not hash_kgw_poseidon2_gpu.native_cuda_available():
        raise RuntimeError("native CUDA kernel source is not available")

    seed = actual_two_to_one_seed(args.hash_key, args.previous_token)
    scores = torch.empty(args.vocab_size, dtype=torch.float32, device=args.device)

    cpu_t2: list[int] | None = None
    cpu_t3: list[int] | None = None
    if not args.skip_equivalence:
        cpu_t2 = cpu_two_to_one(seed, args.vocab_size, args.gamma)
        cpu_t3 = cpu_three_to_one(args.hash_key, args.previous_token, args.vocab_size, args.gamma)

    t2_green_times, t2_green = bench_call(
        args.device,
        args.repeat,
        args.warmup,
        lambda: greenlist_two_to_one(args.backend, seed, args.vocab_size, args.gamma, args.device),
    )
    t3_green_times, t3_green = bench_call(
        args.device,
        args.repeat,
        args.warmup,
        lambda: greenlist_three_to_one(
            args.backend, args.hash_key, args.previous_token, args.vocab_size, args.gamma, args.device
        ),
    )
    t2_bias_times, _ = bench_call(
        args.device,
        args.repeat,
        args.warmup,
        lambda: bias_two_to_one(args.backend, seed, scores, args.gamma, args.device),
    )
    t2_bias_green = torch.nonzero(scores, as_tuple=False).flatten().detach().cpu().tolist()
    t3_bias_times, _ = bench_call(
        args.device,
        args.repeat,
        args.warmup,
        lambda: bias_three_to_one(args.backend, args.hash_key, args.previous_token, scores, args.gamma, args.device),
    )
    t3_bias_green = torch.nonzero(scores, as_tuple=False).flatten().detach().cpu().tolist()

    t2_green_list = as_list(t2_green)
    t3_green_list = as_list(t3_green)
    if cpu_t2 is not None and cpu_t3 is not None:
        assert_equal("Poseidon2 TwoToOneFixed greenlist", cpu_t2, t2_green_list)
        assert_equal("Poseidon2 ThreeToOneFixed greenlist", cpu_t3, t3_green_list)
        assert_equal("Poseidon2 TwoToOneFixed fused bias", cpu_t2, [int(v) for v in t2_bias_green])
        assert_equal("Poseidon2 ThreeToOneFixed fused bias", cpu_t3, [int(v) for v in t3_bias_green])

    result = {
        "backend": args.backend,
        "device": args.device,
        "vocab_size": args.vocab_size,
        "gamma": args.gamma,
        "hash_key": args.hash_key,
        "previous_token": args.previous_token,
        "two_to_one_seed": seed,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "equivalence_checked": not args.skip_equivalence,
        "two_to_one_fixed": {
            "green_count": len(t2_green_list),
            "greenlist_ms": stats_ms(t2_green_times),
            "fused_bias_ms": stats_ms(t2_bias_times),
        },
        "three_to_one_fixed": {
            "green_count": len(t3_green_list),
            "greenlist_ms": stats_ms(t3_green_times),
            "fused_bias_ms": stats_ms(t3_bias_times),
        },
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
