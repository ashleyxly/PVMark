from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "baseline_comparison" / "hash_synthid_gpu_hash_2026-05-25"
DOC_MD = REPO_ROOT / "docs" / "hash_synthid_gpu_hash_results_2026-05-25.md"
DOC_HTML = REPO_ROOT / "docs" / "hash_synthid_gpu_hash_results_2026-05-25.html"
SUMMARY_CSV = OUT_DIR / "summary.csv"


RUNS = [
    {
        "scheme": "Original SynthID non-hash LCG",
        "mode": "benchmark",
        "artifact": REPO_ROOT
        / "tests"
        / "baseline_comparison"
        / "original_synthid_seq_20260525"
        / "efficiency_original_synthid_timing.json",
        "equivalence": "Reference benchmark",
        "notes": "Original SynthID path, no hash-based BN254 computation.",
    },
    {
        "scheme": "Hash-based SynthID MiMC T5 Rust reference",
        "mode": "rust_cpu",
        "artifact": REPO_ROOT
        / "tests"
        / "baseline_comparison"
        / "hash_synthid_gpu_mimc_rust_reference_seq_20260525"
        / "efficiency_hash_synthid_timing.json",
        "equivalence": "Exact current Rust semantics",
        "notes": "CPU/Rayon hash backend; current run showed high WET variance.",
    },
    {
        "scheme": "Hash-based SynthID MiMC T5 GPU hash",
        "mode": "gpu_hash_exact",
        "artifact": REPO_ROOT
        / "tests"
        / "baseline_comparison"
        / "hash_synthid_gpu_mimc_exact_wet_wdt_seq_20260525"
        / "efficiency_hash_synthid_timing.json",
        "equivalence": "Exact g-values and output scores",
        "notes": "BN254 MiMC hash and context-history check on GPU; PyTorch score update retained.",
    },
    {
        "scheme": "Hash-based SynthID MiMC T5 GPU hash + fused score",
        "mode": "gpu_hash_fused_score",
        "artifact": REPO_ROOT
        / "tests"
        / "baseline_comparison"
        / "hash_synthid_gpu_mimc_fused_score_wet_wdt_seq_20260525"
        / "efficiency_hash_synthid_timing.json",
        "equivalence": "Exact g-values; float32 score-order diff <= 1e-3",
        "notes": "Also updates top-k scores in CUDA; fastest MiMC GPU WET path.",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timing_row(item: dict[str, Any]) -> dict[str, Any]:
    data = load_json(item["artifact"])
    meta = data["metadata"]
    wet = data["wet"]["200"]
    wdt = data["wdt"]["200"]
    return {
        "scheme": item["scheme"],
        "mode": item["mode"],
        "hash_type": meta.get("hash_type"),
        "gpu_hash_backend": meta.get("gpu_hash_backend", False),
        "gpu_fused_score_update": meta.get("gpu_fused_score_update", False),
        "wet_runs": wet["runs"],
        "wdt_runs": wdt["runs"],
        "warmup_runs": meta["warmup_runs"],
        "cache_mode": meta["cache_mode"],
        "token_length": wet["token_length"],
        "batch_size": wet["batch_size"],
        "top_k": meta["top_k"],
        "score_type": meta["score_type"],
        "wet_mean_ms_per_200": wet["mean_ms_per_batch"],
        "wet_median_ms_per_200": wet["median_ms_per_batch"],
        "wet_p90_ms_per_200": wet["p90_ms_per_batch"],
        "wet_mean_ms_per_token": wet["mean_ms_per_token"],
        "wdt_mean_ms_per_200": wdt["mean_ms_per_batch"],
        "wdt_median_ms_per_200": wdt["median_ms_per_batch"],
        "wdt_p90_ms_per_200": wdt["p90_ms_per_batch"],
        "wdt_mean_ms_per_token": wdt["mean_ms_per_token"],
        "equivalence": item["equivalence"],
        "notes": item["notes"],
        "artifact": str(item["artifact"].relative_to(REPO_ROOT)),
    }


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_markdown(rows: list[dict[str, Any]]) -> str:
    original = rows[0]
    main_rows = []
    for row in rows:
        main_rows.append(
            [
                row["scheme"],
                fmt_ms(row["wet_mean_ms_per_200"]),
                fmt_ms(row["wet_median_ms_per_200"]),
                fmt_ms(row["wdt_mean_ms_per_200"]),
                fmt_ms(row["wdt_median_ms_per_200"]),
                fmt_ratio(original["wet_mean_ms_per_200"] / row["wet_mean_ms_per_200"]),
                fmt_ratio(original["wdt_mean_ms_per_200"] / row["wdt_mean_ms_per_200"]),
            ]
        )

    detail_rows = [
        [
            row["scheme"],
            str(row["gpu_hash_backend"]),
            str(row["gpu_fused_score_update"]),
            row["equivalence"],
            row["artifact"],
        ]
        for row in rows
    ]

    return f"""# Hash-based SynthID GPU Hash Results, 2026-05-25

## Summary

This report records the experimental GPU implementation for hash-based SynthID MiMC T5. The current GPU path uses Numba CUDA and exact BN254 multi-limb arithmetic for MiMC hash computation. The existing optimized Rust/Python version is preserved in `tests/baseline_comparison/hash_synthid_gpu_hash_backup_20260525_120942/`.

## What Was Implemented

- Added `src/synthid_text/gpu_hash.py`.
- Implemented BN254 field arithmetic on GPU with eight little-endian 32-bit limbs.
- Implemented exact MiMC-7-91 BN254 hashing on GPU using the same round constants as `artifact/third_party/hash_function/arkworks-mimc`.
- Added optional `GPU_HASH_BACKEND` / `--gpu-hash` path for MiMC WET g-value computation.
- Added GPU context-history repetition checking, so the WET path avoids copying context hashes back to CPU each token.
- Added optional `GPU_FUSED_SCORE_UPDATE` / `--gpu-fused-score-update` to update top-k scores on GPU. This is faster but changes float32 reduction order slightly.
- WDT remains on the previously optimized Rust fused detector-score path; it is already much faster than the original benchmark.

## Equivalence

- `--gpu-hash` default path: GPU MiMC g-values, context hashes, and `watermarked_call` output match the Rust reference exactly in the tested cases.
- `--gpu-hash --gpu-fused-score-update`: MiMC g-values remain exact, but top-k score update is computed in a CUDA kernel with a different float32 operation order. Observed max score difference: about `3.09e-4`, within `1e-3` tolerance.
- The attempted single-kernel WET fusion was kept as prototype code but not used by default because it serialized too much candidate/key work and was slower.

## Sequential Benchmark Setup

- Date: 2026-05-25
- GPU: `CUDA_VISIBLE_DEVICES=2`, visible as `cuda:0`
- Model: `$PVMark_GPT2_MODEL`
- Token length: 200
- Batch size: 1
- Top-k: 40
- Score type: `weighted_mean`
- Warmup runs: 10
- WET runs: 30
- WDT runs: 300
- Cache mode: warm

## Results

{markdown_table(["Scheme", "WET mean / 200", "WET median / 200", "WDT mean / 200", "WDT median / 200", "WET speed vs original", "WDT speed vs original"], main_rows)}

## Run Details

{markdown_table(["Scheme", "GPU hash", "GPU fused score", "Equivalence", "Artifact"], detail_rows)}

## Interpretation

The strict GPU hash path lowers MiMC WET compared with the current sequential Rust reference run, but it is still slower than the original SynthID benchmark. The faster fused-score GPU path reaches about 607 ms per 200-token WET, which is closer to the original 400 ms benchmark but still not equal. WDT is already substantially faster than the original benchmark because detection uses the Rust fused detector-score path.

The remaining WET gap is mostly launch overhead and under-occupancy: watermark embedding is token-by-token, batch size is 1, and each token only has 40 candidates and 30 keys. A production CUDA/C++ extension or persistent kernel could reduce launch overhead further; Triton/Numba JIT is useful for the prototype but not ideal for tiny per-token kernels.

## Best Current Presentation Option

For strict equivalence, use the earlier fair-environment Poseidon2 T4 result: WET 380.07 ms and WDT 4.34 ms, faster than original SynthID in that run. For the GPU-hash experiment specifically, present MiMC GPU as a feasibility prototype: exact hash-on-GPU is implemented and validated, but the current Numba path does not yet beat original SynthID WET.
"""


def build_html(markdown_text: str, rows: list[dict[str, Any]]) -> str:
    original = rows[0]
    main_rows = [
        [
            row["scheme"],
            fmt_ms(row["wet_mean_ms_per_200"]),
            fmt_ms(row["wet_median_ms_per_200"]),
            fmt_ms(row["wdt_mean_ms_per_200"]),
            fmt_ms(row["wdt_median_ms_per_200"]),
            fmt_ratio(original["wet_mean_ms_per_200"] / row["wet_mean_ms_per_200"]),
            fmt_ratio(original["wdt_mean_ms_per_200"] / row["wdt_mean_ms_per_200"]),
        ]
        for row in rows
    ]
    detail_rows = [
        [
            row["scheme"],
            str(row["gpu_hash_backend"]),
            str(row["gpu_fused_score_update"]),
            row["equivalence"],
            row["artifact"],
        ]
        for row in rows
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hash-based SynthID GPU Hash Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Hash-based SynthID GPU Hash Results, 2026-05-25</h1>
  <h2>Summary</h2>
  <p>This report records the experimental GPU implementation for hash-based SynthID MiMC T5. The current GPU path uses Numba CUDA and exact BN254 multi-limb arithmetic for MiMC hash computation.</p>
  <h2>What Was Implemented</h2>
  <ul>
    <li>Added <code>src/synthid_text/gpu_hash.py</code>.</li>
    <li>Implemented BN254 arithmetic with eight 32-bit limbs and exact MiMC-7-91 BN254 hashing on GPU.</li>
    <li>Added <code>--gpu-hash</code> and optional <code>--gpu-fused-score-update</code>.</li>
    <li>Moved MiMC WET context-history repetition checking to GPU.</li>
    <li>Kept WDT on the optimized Rust fused detector-score path.</li>
  </ul>
  <h2>Results</h2>
  {html_table(["Scheme", "WET mean / 200", "WET median / 200", "WDT mean / 200", "WDT median / 200", "WET speed vs original", "WDT speed vs original"], main_rows)}
  <h2>Run Details</h2>
  {html_table(["Scheme", "GPU hash", "GPU fused score", "Equivalence", "Artifact"], detail_rows)}
  <h2>Interpretation</h2>
  <p>The strict GPU hash path is exact in tested cases, but still slower than original SynthID WET. The optional fused-score path is faster but introduces small float32 operation-order differences. For the strongest strict-equivalence paper result, the earlier Poseidon2 T4 fair-environment result remains the best candidate.</p>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [timing_row(item) for item in RUNS]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = build_markdown(rows)
    DOC_MD.write_text(md, encoding="utf-8")
    DOC_HTML.write_text(build_html(md, rows), encoding="utf-8")
    print(SUMMARY_CSV)
    print(DOC_MD)
    print(DOC_HTML)


if __name__ == "__main__":
    main()
