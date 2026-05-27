from __future__ import annotations

import argparse
import html
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from baseline_eval import hash_kgw_poseidon2_gpu
from baseline_eval.benchmark_efficiency import (
    DEFAULT_HASH_RESULTS_DIR,
    HASH_METHODS,
    HASH_TYPES,
    benchmark_processor_wet,
    build_benchmark_sequences,
    parse_token_counts,
    resolve_device,
    sample_rows_from_hash_results,
    stats_ms,
    synchronize,
)
from baseline_eval.common import DEFAULT_GENERATION_MODEL, DEFAULT_RESULTS_ROOT, ensure_dir, write_json


DEFAULT_OUTPUT_ROOT = str(Path(DEFAULT_RESULTS_ROOT) / "full_table_wet_wdt_benchmark")
DEFAULT_POSEIDON_ROOT = str(Path(DEFAULT_RESULTS_ROOT) / "poseidon_full_table_cache")
DEFAULT_HASH_ROOT = str(Path(DEFAULT_RESULTS_ROOT) / "hash_kgw_full_table_cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark online WET/WDT using precomputed hash-KGW full tables.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poseidon-root", default=DEFAULT_POSEIDON_ROOT)
    parser.add_argument("--hash-root", default=DEFAULT_HASH_ROOT)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--targets", default="3:2,3:4,4:2,5:2,5:4")
    parser.add_argument("--hash-key", type=int, default=2023)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--hash-results-dir", default=DEFAULT_HASH_RESULTS_DIR)
    parser.add_argument("--wet-max-samples", type=int, default=100)
    parser.add_argument("--wdt-max-samples", type=int, default=100)
    parser.add_argument("--wet-token-count", type=int, default=200)
    parser.add_argument("--wdt-token-counts", default="50,200")
    parser.add_argument("--prompt-max-length", type=int, default=1848)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1)
    return parser.parse_args()


def parse_targets(value: str) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, sep, right = item.partition(":")
        if not sep:
            raise SystemExit(f"Invalid target {item!r}; expected hash_type:hash_method")
        hash_type = int(left)
        hash_method = int(right)
        if hash_type not in HASH_TYPES:
            raise SystemExit(f"Unsupported hash_type={hash_type}; valid values are {sorted(HASH_TYPES)}")
        if hash_method not in HASH_METHODS:
            raise SystemExit(f"Unsupported hash_method={hash_method}; valid values are {sorted(HASH_METHODS)}")
        target = (hash_type, hash_method)
        if target not in targets:
            targets.append(target)
    return targets


def gamma_tag(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p").replace("-", "m")


def table_stem(hash_type: int, hash_method: int, vocab_size: int, gamma: float, hash_key: int) -> str:
    if hash_type == 3:
        return (
            f"poseidon_hash_type3_method{hash_method}_{HASH_METHODS[hash_method]}"
            f"_vocab{vocab_size}_gamma{gamma_tag(gamma)}_key{hash_key}_uint8"
        )
    return (
        f"hashkgw_{HASH_TYPES[hash_type]}_hash_type{hash_type}"
        f"_method{hash_method}_{HASH_METHODS[hash_method]}"
        f"_vocab{vocab_size}_gamma{gamma_tag(gamma)}_key{hash_key}_uint8"
    )


def table_root(hash_type: int, args: argparse.Namespace) -> Path:
    return Path(args.poseidon_root) if hash_type == 3 else Path(args.hash_root)


def table_paths(
    args: argparse.Namespace,
    hash_type: int,
    hash_method: int,
    vocab_size: int,
) -> dict[str, Path]:
    root = table_root(hash_type, args)
    stem = table_stem(hash_type, hash_method, vocab_size, args.gamma, args.hash_key)
    return {
        "table": root / f"{stem}.npy",
        "metadata": root / f"{stem}_metadata.json",
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_size_gib(path: Path) -> float | None:
    if not path.exists():
        return None
    return path.stat().st_size / float(1024**3)


class FullTableLogitsProcessor:
    def __init__(self, table_gpu: torch.Tensor, delta: float, device: str) -> None:
        if table_gpu.dtype != torch.uint8 or not table_gpu.is_cuda or table_gpu.ndim != 2:
            raise ValueError("table_gpu must be CUDA uint8 with shape [vocab_size, vocab_size]")
        self.table_gpu = table_gpu
        self.delta = float(delta)
        self.device = device
        self.vocab_size = int(table_gpu.shape[1])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[0] != scores.shape[0]:
            raise ValueError("input_ids and scores batch sizes must match")
        for batch_index in range(input_ids.shape[0]):
            previous_token = int(input_ids[batch_index, -1].detach().item())
            mask = self.table_gpu[previous_token]
            hash_kgw_poseidon2_gpu.bias_logits_with_mask(mask, scores[batch_index], self.delta, self.device)
        return scores


def load_table_to_gpu(table_path: Path, device: str) -> tuple[torch.Tensor, dict[str, Any]]:
    synchronize(device)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table_np = np.load(table_path, mmap_mode="r")
        cpu_tensor = torch.from_numpy(table_np)
        table_gpu = cpu_tensor.to(device=device, dtype=torch.uint8, copy=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    return table_gpu, {
        "definition": "mmap full uint8 table and copy it into GPU memory",
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "shape": list(table_gpu.shape),
        "dtype": str(table_gpu.dtype),
        "device": str(table_gpu.device),
    }


def benchmark_wet(
    *,
    table_path: Path,
    sequences: list[dict[str, Any]],
    vocab_size: int,
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    table_gpu, load_info = load_table_to_gpu(table_path, device)
    try:
        processor = FullTableLogitsProcessor(table_gpu, args.delta, device)
        wet = benchmark_processor_wet(
            processor=processor,
            sequences=sequences,
            token_count=args.wet_token_count,
            vocab_size=vocab_size,
            device=device,
            warmup_samples=args.warmup_samples,
            repeat=args.repeat,
        )
    finally:
        del table_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"load_table_to_gpu": load_info, "wet_processor": wet}


class FullTableDetector:
    def __init__(self, table_np: np.ndarray, gamma: float) -> None:
        self.table_np = table_np
        self.gamma = float(gamma)

    def score_ids(self, token_ids: torch.Tensor, token_count: int) -> dict[str, Any]:
        ids = token_ids[:token_count].detach().cpu().numpy().astype(np.int64, copy=False)
        if ids.shape[0] < 2:
            return {
                "num_tokens_scored": 0,
                "num_green_tokens": 0,
                "green_fraction": 0.0,
                "z_score": 0.0,
            }
        prev = ids[:-1]
        curr = ids[1:]
        green_mask = self.table_np[prev, curr].astype(bool, copy=False)
        num_scored = int(green_mask.shape[0])
        green_count = int(green_mask.sum())
        denom = math.sqrt(num_scored * self.gamma * (1.0 - self.gamma)) if num_scored > 0 else 0.0
        z_score = ((green_count - self.gamma * num_scored) / denom) if denom > 0.0 else 0.0
        return {
            "num_tokens_scored": num_scored,
            "num_green_tokens": green_count,
            "green_fraction": green_count / num_scored if num_scored else 0.0,
            "z_score": z_score,
        }


def benchmark_wdt(
    *,
    table_path: Path,
    sequences: list[dict[str, Any]],
    token_counts: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    load_start = time.perf_counter()
    table_np = np.load(table_path, mmap_mode="r")
    load_elapsed = time.perf_counter() - load_start
    detector = FullTableDetector(table_np, args.gamma)

    results: dict[str, Any] = {
        "load_table_mmap": {
            "definition": "mmap full uint8 table for CPU token-id WDT lookup",
            "elapsed_sec": load_elapsed,
            "elapsed_ms": load_elapsed * 1000.0,
            "shape": list(table_np.shape),
            "dtype": str(table_np.dtype),
        }
    }
    for token_count in token_counts:
        token_id_inputs = [seq["continuation_ids"][:token_count].detach().cpu() for seq in sequences]
        for idx in range(min(args.warmup_samples, len(token_id_inputs))):
            detector.score_ids(token_id_inputs[idx], token_count)

        times: list[float] = []
        for _ in range(args.repeat):
            for ids in token_id_inputs:
                start = time.perf_counter()
                detector.score_ids(ids, token_count)
                times.append(time.perf_counter() - start)
        results[f"wdt_{token_count}"] = {
            "definition": (
                f"token-id detector-only WDT for one {token_count}-token continuation using "
                "precomputed full-table membership lookup; excludes tokenizer"
            ),
            "num_samples": len(token_id_inputs),
            "repeat": args.repeat,
            "token_ids_detector_only_ms": stats_ms(times),
        }
    return results


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        f_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f_value):
        return str(value)
    return f"{f_value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_reports(output_root: Path, results: dict[str, Any]) -> dict[str, str]:
    rows: list[list[str]] = []
    for key in sorted(results["targets"]):
        item = results["targets"][key]
        gen = item.get("generation_metadata", {})
        wet = item.get("wet", {}).get("wet_processor", {})
        wdt = item.get("wdt", {})
        row = [
            key,
            gen.get("hash_name", ""),
            gen.get("hash_method_name", ""),
            fmt_float(gen.get("generation_sec"), 3),
            fmt_float(gen.get("file_size_gib"), 3),
            fmt_float(item.get("wet", {}).get("load_table_to_gpu", {}).get("elapsed_sec"), 3),
            fmt_float(wet.get("aggregate_ms_per_token"), 4),
            fmt_float(wet.get("wet_total_ms_from_aggregate"), 3),
        ]
        for token_count in results["metadata"]["wdt_token_counts"]:
            row.append(
                fmt_float(
                    wdt.get(f"wdt_{token_count}", {})
                    .get("token_ids_detector_only_ms", {})
                    .get("mean"),
                    4,
                )
            )
        rows.append(row)

    headers = [
        "Target",
        "Hash",
        "Method",
        "Full-table gen sec",
        "Table GiB",
        "GPU load sec",
        "WET ms/token",
        "WET total ms",
    ]
    headers.extend(f"WDT-{token_count} ids ms" for token_count in results["metadata"]["wdt_token_counts"])

    json_path = output_root / "full_table_wet_wdt_results.json"
    md_path = output_root / "full_table_wet_wdt_results.md"
    html_path = output_root / "full_table_wet_wdt_results.html"
    write_json(json_path, results)

    md_text = "\n".join(
        [
            "# Full-Table WET/WDT Results",
            "",
            f"Created: {results['metadata']['created_at']}",
            "",
            "## Scope",
            "",
            (
                "WET loads one full `uint8` membership table to GPU and measures processor-only "
                "`processor(prefix, scores)` time. WDT mmaps the same table on CPU and measures "
                "token-id detector-only membership lookup, excluding tokenizer."
            ),
            "",
            "## Summary",
            "",
            markdown_table(headers, rows),
            "",
            "## Artifacts",
            "",
            f"- Result JSON: `{json_path}`",
            f"- Output root: `{output_root}`",
        ]
    )
    md_path.write_text(md_text, encoding="utf-8")

    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Full-Table WET/WDT Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
    th {{ background: #f9fafb; }}
    td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>Full-Table WET/WDT Results</h1>
  <p>Created: <code>{html.escape(results['metadata']['created_at'])}</code></p>
  <p>WET uses GPU full-table lookup plus mask bias. WDT uses CPU mmap full-table membership lookup on token ids.</p>
  <table>
    <thead><tr>{''.join(f'<th>{html.escape(header)}</th>' for header in headers)}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
  <p>Result JSON: <code>{html.escape(str(json_path))}</code></p>
  <p>Companion Markdown: <code>{html.escape(str(md_path))}</code></p>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)
    targets = parse_targets(args.targets)
    token_counts = parse_token_counts(args.wdt_token_counts)
    device = resolve_device(args.device, require_cuda=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    vocab_size = len(list(tokenizer.get_vocab().values()))

    results: dict[str, Any] = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "definition": "Unified WET/WDT benchmark over precomputed full hash-KGW tables.",
            "model": args.model,
            "device": device,
            "targets": [f"{hash_type}:{hash_method}" for hash_type, hash_method in targets],
            "hash_key": int(args.hash_key),
            "gamma": float(args.gamma),
            "delta": float(args.delta),
            "vocab_size": int(vocab_size),
            "wet_max_samples": int(args.wet_max_samples),
            "wdt_max_samples": int(args.wdt_max_samples),
            "wet_token_count": int(args.wet_token_count),
            "wdt_token_counts": token_counts,
            "warmup_samples": int(args.warmup_samples),
            "repeat": int(args.repeat),
            "poseidon_root": args.poseidon_root,
            "hash_root": args.hash_root,
        },
        "targets": {},
    }

    for hash_type, hash_method in targets:
        key = f"{hash_type}:{hash_method}"
        print(f"Benchmarking full-table target={key} {HASH_TYPES[hash_type]} {HASH_METHODS[hash_method]}", flush=True)
        paths = table_paths(args, hash_type, hash_method, vocab_size)
        if not paths["table"].exists():
            raise SystemExit(f"Missing table for target={key}: {paths['table']}")
        wet_rows = sample_rows_from_hash_results(args.hash_results_dir, hash_type, hash_method, args.wet_max_samples)
        wdt_rows = sample_rows_from_hash_results(args.hash_results_dir, hash_type, hash_method, args.wdt_max_samples)
        wet_sequences = build_benchmark_sequences(tokenizer, wet_rows, args.wet_token_count, device, args.prompt_max_length)
        wdt_sequences = build_benchmark_sequences(tokenizer, wdt_rows, max(token_counts), device, args.prompt_max_length)
        if not wet_sequences:
            raise SystemExit(f"No WET samples for target={key}")
        if not wdt_sequences:
            raise SystemExit(f"No WDT samples for target={key}")

        metadata = load_json(paths["metadata"])
        metadata.setdefault("file_size_gib", file_size_gib(paths["table"]))
        wet = benchmark_wet(
            table_path=paths["table"],
            sequences=wet_sequences,
            vocab_size=vocab_size,
            args=args,
            device=device,
        )
        wdt = benchmark_wdt(
            table_path=paths["table"],
            sequences=wdt_sequences,
            token_counts=token_counts,
            args=args,
        )
        results["targets"][key] = {
            "table_path": str(paths["table"]),
            "metadata_path": str(paths["metadata"]),
            "generation_metadata": metadata,
            "wet": wet,
            "wdt": wdt,
        }

    report_paths = write_reports(output_root, results)
    print(json.dumps({"reports": report_paths}, indent=2), flush=True)


if __name__ == "__main__":
    main()
