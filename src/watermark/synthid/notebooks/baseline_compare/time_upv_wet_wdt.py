from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import upv_network_detector as und  # noqa: E402


@contextmanager
def force_torch_load_cpu() -> Any:
    original_torch_load = torch.load

    def _cpu_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("map_location", "cpu")
        return original_torch_load(*args, **kwargs)

    torch.load = _cpu_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


DEFAULT_OUTPUT_DIR = Path("tests/baseline_comparison/upv_network_detector_gpt2_eli5")
DEFAULT_UPV_ROOT = Path(os.environ.get("PVMark_UPV_ROOT", "external/unforgeable_watermark"))
DEFAULT_GPT2 = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
DEFAULT_GENERATIONS = Path("tests/baseline_comparison/upv_gpt2/generations.json")
DEFAULT_ATTACKS = Path("tests/baseline_comparison/upv_gpt2/attacks.json")
DEFAULT_GENERATOR = (
    DEFAULT_UPV_ROOT / "experiments/main_experiments/generator_model/combine_model.pt"
)
DEFAULT_SUBNET = DEFAULT_UPV_ROOT / "experiments/main_experiments/generator_model/sub_net.pt"


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
    parser = argparse.ArgumentParser(description="Measure UPV WET/WDT and attack deltas.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--upv-root", default=str(DEFAULT_UPV_ROOT))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2))
    parser.add_argument("--generator-model", default=str(DEFAULT_GENERATOR))
    parser.add_argument("--subnet", default=str(DEFAULT_SUBNET))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--generations", default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--attacks", default=str(DEFAULT_ATTACKS))
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--llm-name", default="gpt2")
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--z-value", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--wet-runs", type=int, default=2000)
    parser.add_argument("--wdt-runs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wet-token-length", type=int, default=64)
    parser.add_argument("--wdt-token-length", type=int, default=50)
    parser.add_argument(
        "--wet-mode",
        choices=["single_context_x_tokens", "strict_sequential"],
        default="single_context_x_tokens",
        help=(
            "single_context_x_tokens preserves the historical UPV timing: one "
            "processor call on a long context, multiplied by wet-token-length. "
            "strict_sequential times wet-token-length online processor calls."
        ),
    )
    return parser.parse_args()


def import_upv(root: str | Path) -> tuple[Any, Any]:
    sys.path.insert(0, str(root))
    from watermark_model import Watermark, WatermarkLogitsProcessor  # type: ignore

    return Watermark, WatermarkLogitsProcessor


def load_generation_records(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def time_wet(args: argparse.Namespace) -> dict[str, Any]:
    device = und.resolve_device(args.device)
    Watermark, WatermarkLogitsProcessor = import_upv(args.upv_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=model_dtype,
    )
    model.eval()

    with force_torch_load_cpu():
        watermark = Watermark(
            bit_number=args.bit_number,
            window_size=args.window_size,
            layers=args.layers,
            delta=args.delta,
            model_dir=args.generator_model,
            beam_size=args.beam_size,
        )
    processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        delta=args.delta,
        model=watermark.model,
        window_size=args.window_size,
        cache=watermark.cache,
        bit_number=args.bit_number,
        beam_size=args.beam_size,
        llm_name=args.llm_name,
    )

    records = load_generation_records(args.generations)
    wm_records = [r for r in records if r.get("watermarked")]

    samples: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for record in wm_records:
        if args.wet_mode == "strict_sequential":
            text = record.get("full_text") or (
                str(record.get("prompt") or "") + str(record.get("completion_text") or "")
            )
            prefix_start = max(args.window_size - 1, int(record.get("prompt_token_count") or 0))
        else:
            text = record.get("completion_text", "")
            prefix_start = 0
        required_tokens = args.wet_token_length + prefix_start
        ids = tokenizer(text or " ", return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[-1] < args.window_size:
            continue
        if ids.shape[-1] < required_tokens:
            continue
        ids = ids[:, :required_tokens].to(device)
        with torch.no_grad():
            if args.wet_mode == "strict_sequential":
                scores = model(ids).logits[:, :required_tokens, :].detach()
            else:
                scores = model(ids).logits[:, -1, :].detach()
        samples.append((ids, scores, prefix_start))
        if len(samples) >= args.num_samples:
            break
    if not samples:
        raise RuntimeError("no valid WET samples")

    def run_wet_sample(ids: torch.Tensor, scores: torch.Tensor, prefix_start: int) -> None:
        if args.wet_mode == "strict_sequential":
            for step in range(args.wet_token_length):
                prefix_len = prefix_start + step
                score_index = max(0, prefix_len - 1)
                processor(ids[:, :prefix_len], scores[:, score_index, :].clone())
        else:
            processor(ids, scores.clone())

    # CUDA kernel warmup without keeping cache effects.
    with torch.no_grad():
        for ids, scores, prefix_start in samples[: min(16, len(samples))]:
            run_wet_sample(ids, scores, prefix_start)
    watermark.cache.clear()
    if device.type == "cuda":
        torch.cuda.synchronize()

    cold_durations: list[float] = []
    with torch.no_grad():
        for ids, scores, prefix_start in samples:
            start = time.perf_counter()
            run_wet_sample(ids, scores, prefix_start)
            if device.type == "cuda":
                torch.cuda.synchronize()
            cold_durations.append(time.perf_counter() - start)

    warm_durations: list[float] = []
    with torch.no_grad():
        for i in range(args.wet_runs):
            ids, scores, prefix_start = samples[i % len(samples)]
            start = time.perf_counter()
            run_wet_sample(ids, scores, prefix_start)
            if device.type == "cuda":
                torch.cuda.synchronize()
            warm_durations.append(time.perf_counter() - start)

    def summarize(durations: list[float]) -> dict[str, float]:
        denominator = args.wet_token_length if args.wet_mode == "strict_sequential" else 1
        mean_sec_per_token = statistics.mean(durations) / denominator
        median_sec_per_token = statistics.median(durations) / denominator
        return {
            "mean_sec_per_token": mean_sec_per_token,
            "median_sec_per_token": median_sec_per_token,
            "mean_sec_per_sample": statistics.mean(durations)
            if args.wet_mode == "strict_sequential"
            else mean_sec_per_token * args.wet_token_length,
            "median_sec_per_sample": statistics.median(durations)
            if args.wet_mode == "strict_sequential"
            else median_sec_per_token * args.wet_token_length,
            "p90_sec_per_token": sorted(durations)[int(0.9 * (len(durations) - 1))]
            / denominator,
            "min_sec_per_token": min(durations) / denominator,
            "max_sec_per_token": max(durations) / denominator,
        }

    if args.wet_mode == "strict_sequential":
        wet_definition = (
            f"WET is {args.wet_token_length} sequential UPV "
            "WatermarkLogitsProcessor calls, replaying online embedding "
            "decisions with precomputed logits; LLM forward is excluded."
        )
    else:
        wet_definition = (
            f"WET is one invocation of UPV WatermarkLogitsProcessor for a "
            f"{args.wet_token_length}-token context; LLM forward is excluded. "
            f"Per-sample fields multiply the one-token processor cost by "
            f"{args.wet_token_length} for the paper's fixed-length comparison."
        )

    return {
        "definition": wet_definition,
        "wet_mode": args.wet_mode,
        "timed_embedding_calls": args.wet_token_length
        if args.wet_mode == "strict_sequential"
        else 1,
        "context_offset_tokens": {
            "mean": statistics.mean([s[2] for s in samples]),
            "min": min(s[2] for s in samples),
            "max": max(s[2] for s in samples),
        },
        "num_prefix_samples": len(samples),
        "cold_first_pass": {
            "runs": len(cold_durations),
            **summarize(cold_durations),
        },
        "warm_cached": {
            "runs": len(warm_durations),
            **summarize(warm_durations),
        },
    }


def time_wdt(args: argparse.Namespace) -> dict[str, Any]:
    device = und.resolve_device(args.device)
    tokenizer = und.load_tokenizer(args.model_name_or_path)
    detector = und.load_detector(args, device)
    records = load_generation_records(args.generations)
    texts = [r.get("completion_text", "") for r in records[: args.num_samples]]
    texts_50 = []
    for text in texts:
        ids = tokenizer(text or "", return_tensors=None, add_special_tokens=True)["input_ids"][
            : args.wdt_token_length
        ]
        texts_50.append(tokenizer.decode(ids, skip_special_tokens=True))

    # Warmup.
    for _ in range(3):
        und.predict_texts(detector, tokenizer, texts_50[: min(256, len(texts_50))], args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    for _ in range(args.wdt_runs):
        start = time.perf_counter()
        und.predict_texts(detector, tokenizer, texts_50, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter() - start) / len(texts_50))

    return {
        "definition": (
            f"WDT is network-based detector wall-clock time for a "
            f"{args.wdt_token_length}-token text, including tokenization, bit "
            f"feature construction, and network forward."
        ),
        "num_texts_per_run": len(texts_50),
        "runs": args.wdt_runs,
        "mean_sec_per_text": statistics.mean(durations),
        "median_sec_per_text": statistics.median(durations),
        "p90_sec_per_text": sorted(durations)[int(0.9 * (len(durations) - 1))],
        "min_sec_per_text": min(durations),
        "max_sec_per_text": max(durations),
    }


def token_edit_stats(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    with open(args.attacks, "r", encoding="utf-8") as f:
        records = json.load(f)["records"]

    def levenshtein_ratio(a: list[int], b: list[int]) -> float:
        if not a and not b:
            return 0.0
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            curr = [i]
            for j, y in enumerate(b, 1):
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[-1] + 1,
                        prev[j - 1] + (0 if x == y else 1),
                    )
                )
            prev = curr
        return prev[-1] / max(len(a), len(b), 1)

    by_attack: dict[str, dict[str, list[float]]] = {}
    for record in records:
        original_ids = tokenizer(record.get("original_text", ""), add_special_tokens=False)[
            "input_ids"
        ]
        for attack_name, text in record.get("attacks", {}).items():
            attacked_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            entry = by_attack.setdefault(
                attack_name,
                {
                    "orig_len": [],
                    "attacked_len": [],
                    "len_ratio": [],
                    "edit_ratio": [],
                    "unchanged": [],
                },
            )
            entry["orig_len"].append(float(len(original_ids)))
            entry["attacked_len"].append(float(len(attacked_ids)))
            entry["len_ratio"].append(float(len(attacked_ids) / max(len(original_ids), 1)))
            entry["edit_ratio"].append(float(levenshtein_ratio(original_ids, attacked_ids)))
            entry["unchanged"].append(1.0 if original_ids == attacked_ids else 0.0)

    def summarize(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        attack: {name: summarize(values) for name, values in stats.items()}
        for attack, stats in sorted(by_attack.items())
    }


def main() -> None:
    args = parse_args()
    output = {
        "metadata": {
            "model_name_or_path": args.model_name_or_path,
            "generator_model": args.generator_model,
            "detector_checkpoint": str(und.checkpoint_path(args)),
            "device": str(und.resolve_device(args.device)),
            "wet_token_length": args.wet_token_length,
            "wdt_token_length": args.wdt_token_length,
            "wet_mode": args.wet_mode,
            "runtime_environment": runtime_environment(),
        },
        (
            "wet_1_token"
            if args.wet_token_length == 64
            else f"wet_{args.wet_token_length}_tokens"
        ): time_wet(args),
        (
            "wdt_50_tokens"
            if args.wdt_token_length == 50
            else f"wdt_{args.wdt_token_length}_tokens"
        ): time_wdt(args),
        "attack_token_edit_stats": token_edit_stats(args),
    }
    suffix = ""
    if args.wet_token_length != 64 or args.wdt_token_length != 50:
        suffix = f"_wet{args.wet_token_length}_wdt{args.wdt_token_length}"
    if args.wet_mode != "single_context_x_tokens":
        suffix += f"_{args.wet_mode}"
    out_path = Path(args.output_dir) / f"wet_wdt_network_z{und.format_z(args.z_value)}{suffix}.json"
    und.write_json(out_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
