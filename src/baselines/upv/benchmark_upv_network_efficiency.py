from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from baseline_eval.common import DEFAULT_UPV_ROOT, read_json, write_json
from baseline_eval.upv_network import UpvNetworkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark UPV network detector WET/WDT from a generation run.")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--wdt-token-count", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def mean_ms(values: list[float]) -> float:
    return statistics.mean(values) * 1000.0


def quantiles_ms(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def pick(p: float) -> float:
        index = round((len(ordered) - 1) * p)
        return ordered[index] * 1000.0

    return {
        "min": ordered[0] * 1000.0,
        "p05": pick(0.05),
        "p25": pick(0.25),
        "median": pick(0.5),
        "p75": pick(0.75),
        "p95": pick(0.95),
        "max": ordered[-1] * 1000.0,
        "mean": mean_ms(ordered),
    }


def benchmark_call(fn: Any, *, warmup: int, repeat: int, device: str) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    timings: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        synchronize(device)
        timings.append(time.perf_counter() - start)
    return timings


def first_n_token_texts(detector: UpvNetworkDetector, samples: list[dict[str, Any]], field: str, n: int) -> list[str]:
    texts: list[str] = []
    for sample in samples:
        text = sample[field]
        ids = detector.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].squeeze(0)
        texts.append(detector.tokenizer.decode(ids[:n], skip_special_tokens=True))
    return texts


def main() -> None:
    args = parse_args()
    generations = read_json(args.generations)
    meta = generations["metadata"]["scheme_metadata"]
    run_args = generations["metadata"]["args"]
    samples = generations["samples"]

    detector = UpvNetworkDetector(
        upv_root=args.upv_root,
        detector_model=meta["network_detector_model"],
        tokenizer_path=run_args["model"],
        bit_number=meta["bit_number"],
        layers=meta["layers"],
        fixed_length=meta.get("detector_fixed_length", 200),
        threshold=meta.get("network_threshold", 0.5),
        device=args.device,
    )

    wm_50_texts = first_n_token_texts(detector, samples, "output_with_watermark", args.wdt_token_count)
    plain_50_texts = first_n_token_texts(detector, samples, "output_without_watermark", args.wdt_token_count)

    def score_texts(texts: list[str]) -> None:
        for text in texts:
            detector.score_text(text)

    wm_wdt = benchmark_call(
        lambda: score_texts(wm_50_texts),
        warmup=args.warmup,
        repeat=args.repeat,
        device=detector.device,
    )
    plain_wdt = benchmark_call(
        lambda: score_texts(plain_50_texts),
        warmup=args.warmup,
        repeat=args.repeat,
        device=detector.device,
    )

    n_wm = len(wm_50_texts)
    n_plain = len(plain_50_texts)
    wet_values = [
        sample["generation_time_with_watermark_sec"] / sample["token_count_with_watermark"] * 1000.0
        for sample in samples
    ]
    plain_gen_values = [
        sample["generation_time_without_watermark_sec"] / sample["token_count_without_watermark"] * 1000.0
        for sample in samples
    ]
    overhead_values = [
        (sample["generation_time_with_watermark_sec"] - sample["generation_time_without_watermark_sec"])
        / sample["token_count_with_watermark"]
        * 1000.0
        for sample in samples
    ]

    result = {
        "metadata": {
            "source_generations": args.generations,
            "device": detector.device,
            "wdt_token_count": args.wdt_token_count,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "num_texts": {"watermarked": n_wm, "plain": n_plain},
            "detector_fixed_length": meta.get("detector_fixed_length", 200),
        },
        "wet_ms_per_token": {
            "definition": "end-to-end watermarked generation time divided by generated token count",
            "mean": statistics.mean(wet_values),
            "median": statistics.median(wet_values),
        },
        "plain_generation_ms_per_token": {
            "mean": statistics.mean(plain_gen_values),
            "median": statistics.median(plain_gen_values),
        },
        "embedding_overhead_ms_per_token": {
            "definition": "(watermarked generation time - plain generation time) divided by generated token count",
            "mean": statistics.mean(overhead_values),
            "median": statistics.median(overhead_values),
        },
        "wdt_ms_per_50_tokens": {
            "definition": "score_text on decoded 50-token continuations, including tokenizer plus detector forward pass; repeated over all samples",
            "watermarked": {
                "per_batch_of_samples": quantiles_ms(wm_wdt),
                "per_text": mean_ms(wm_wdt) / n_wm,
            },
            "plain": {
                "per_batch_of_samples": quantiles_ms(plain_wdt),
                "per_text": mean_ms(plain_wdt) / n_plain,
            },
        },
    }

    out = args.output or str(Path(args.generations).with_name("upv_network_efficiency_benchmark.json"))
    write_json(out, result)
    print(f"Wrote {out}")
    print(f"WET: {result['wet_ms_per_token']['mean']:.4f} ms/token")
    print(f"Embedding overhead: {result['embedding_overhead_ms_per_token']['mean']:.4f} ms/token")
    print(f"WDT watermarked: {result['wdt_ms_per_50_tokens']['watermarked']['per_text']:.4f} ms/50 tokens")
    print(f"WDT plain: {result['wdt_ms_per_50_tokens']['plain']['per_text']:.4f} ms/50 tokens")


if __name__ == "__main__":
    main()
