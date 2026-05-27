from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = (
    REPO_ROOT
    / "tests"
    / "baseline_comparison"
    / "hash_synthid_gpu_batch_system_seq_20260525"
)
DOC_MD = REPO_ROOT / "docs" / "hash_synthid_gpu_batched_wet_results_2026-05-25.md"
DOC_HTML = REPO_ROOT / "docs" / "hash_synthid_gpu_batched_wet_results_2026-05-25.html"
SUMMARY_CSV = OUT_DIR / "summary.csv"
EQUIVALENCE_JSON = OUT_DIR / "batched_replay_equivalence.json"


RUNS = [
    {
        "scheme": "Original SynthID non-hash LCG",
        "mode": "benchmark_online",
        "artifact": OUT_DIR
        / "original_synthid"
        / "efficiency_original_synthid_timing.json",
        "wet_semantics": "online sequential replay",
        "equivalence": "Reference benchmark",
        "notes": "Original SynthID path under the same GPU environment.",
    },
    {
        "scheme": "Hash-based SynthID MiMC T5 GPU online",
        "mode": "mimc_gpu_online",
        "artifact": OUT_DIR
        / "mimc_gpu_online"
        / "efficiency_hash_synthid_timing.json",
        "wet_semantics": "online sequential replay",
        "equivalence": "Exact MiMC g-values; fused score uses same float32 order as batched replay",
        "notes": "One WET call per token; Numba CUDA hash and score kernels launch repeatedly.",
    },
    {
        "scheme": "Hash-based SynthID MiMC T5 GPU batched replay",
        "mode": "mimc_gpu_batched_replay",
        "artifact": OUT_DIR
        / "mimc_gpu_batched_replay"
        / "efficiency_hash_synthid_timing.json",
        "wet_semantics": "offline multi-token replay",
        "equivalence": "Matches online GPU replay outputs in saved equivalence check",
        "notes": "All known replay token steps are flattened into one larger GPU workload.",
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
        "batched_wet_replay": meta.get("batched_wet_replay", False),
        "wet_semantics": item["wet_semantics"],
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
        "wet_min_ms_per_200": wet["min_ms_per_batch"],
        "wet_max_ms_per_200": wet["max_ms_per_batch"],
        "wet_p90_ms_per_200": wet["p90_ms_per_batch"],
        "wet_mean_ms_per_token": wet["mean_ms_per_token"],
        "wdt_mean_ms_per_200": wdt["mean_ms_per_batch"],
        "wdt_median_ms_per_200": wdt["median_ms_per_batch"],
        "wdt_min_ms_per_200": wdt["min_ms_per_batch"],
        "wdt_max_ms_per_200": wdt["max_ms_per_batch"],
        "wdt_p90_ms_per_200": wdt["p90_ms_per_batch"],
        "wdt_mean_ms_per_token": wdt["mean_ms_per_token"],
        "equivalence": item["equivalence"],
        "notes": item["notes"],
        "artifact": str(item["artifact"].relative_to(REPO_ROOT)),
    }


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def fmt_token(value: float) -> str:
    return f"{value:.4f} ms/token"


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
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def build_markdown(rows: list[dict[str, Any]], equivalence: dict[str, Any]) -> str:
    original = rows[0]
    online_gpu = rows[1]
    main_rows = []
    for row in rows:
        main_rows.append(
            [
                row["scheme"],
                row["wet_semantics"],
                fmt_ms(row["wet_mean_ms_per_200"]),
                fmt_ms(row["wet_median_ms_per_200"]),
                fmt_token(row["wet_mean_ms_per_token"]),
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
            str(row["batched_wet_replay"]),
            row["equivalence"],
            row["artifact"],
        ]
        for row in rows
    ]

    batched = rows[2]
    online_speedup = online_gpu["wet_mean_ms_per_200"] / batched["wet_mean_ms_per_200"]

    return f"""# Hash-based SynthID GPU Batched WET Results, 2026-05-25

## Summary

This round adds an offline multi-token GPU replay path for hash-based SynthID MiMC T5. It keeps the existing optimized online implementation intact and adds a separate `--batched-wet-replay` benchmark mode for the case where all replay logits and token contexts are already known.

The new batched replay WET is `21.44 ms` per 200-token sample on the formal 20-run sequential test, compared with `396.88 ms` for the original SynthID benchmark in the same environment. This is `18.51x` faster than the benchmark and `28.33x` faster than the online MiMC GPU path measured in this run.

## What Changed

- Added `--batched-wet-replay` to `notebooks/baseline_compare/time_hash_synthid_efficiency.py`.
- Built all replay contexts with vectorized `torch.cat(...).unfold(...)` instead of 200 Python context updates.
- Flattened `[step, batch]` into one large GPU batch for MiMC WET replay.
- Used the split-context MiMC GPU path: context hashes are computed once per token step, then candidate/key g-values are computed for all top-k continuations.
- Added a GPU batched repetition-mask kernel for replay order, using the same initial zero history and context-history window semantics.
- Kept g-values as `uint8` when using GPU fused score update; the score kernel only needs zero/nonzero flags, so this avoids a redundant `uint8 -> float32` conversion.
- Changed the split-context candidate kernel launch from 64 to 32 threads per block, matching 30 watermark keys more closely.
- Added `notebooks/baseline_compare/verify_batched_wet_replay.py` for output equivalence checks.

## Semantics And Limitation

The batched replay mode is fair for the existing WET replay metric because that metric already excludes LLM forward time and reuses precomputed logits. It is not the same as true online generation latency: in real generation, future logits and future sampled tokens are not known, so token steps cannot generally be batched across time.

For online generation latency, the relevant GPU result is still the online MiMC GPU row. For paper presentation, label the new number as offline/replay WET or batched WET replay.

## Equivalence Check

Saved artifact: `{EQUIVALENCE_JSON.relative_to(REPO_ROOT)}`

- Top-k indices equal: `{equivalence["top_k_indices_equal"]}`
- Original top-k score max diff: `{equivalence["original_top_k_score_max_diff"]}`
- Updated watermarked score max diff: `{equivalence["updated_score_max_diff"]}`
- Updated score allclose at atol `{equivalence["score_atol"]}`: `{equivalence["updated_score_allclose"]}`

This compares 200-token sequential online GPU fused-score replay against the batched replay path with batch size 1 and top-k 40.

## Formal Sequential Test Setup

- Date: 2026-05-25
- GPU: `CUDA_VISIBLE_DEVICES=2`, visible as `cuda:0`
- Model: `$PVMark_GPT2_MODEL`
- Token length: 200
- Batch size: 1
- Top-k: 40
- Score type: `weighted_mean`
- Warmup runs: 3
- WET runs: 20
- WDT runs: 20
- Cache mode: warm

## Results

{markdown_table(["Scheme", "WET semantics", "WET mean / 200", "WET median / 200", "WET mean / token", "WDT mean / 200", "WDT median / 200", "WET speed vs original", "WDT speed vs original"], main_rows)}

## Run Details

{markdown_table(["Scheme", "GPU hash", "GPU fused score", "Batched WET", "Equivalence", "Artifact"], detail_rows)}

## Interpretation

The optimization works because it changes the shape of the GPU workload. The earlier online GPU path launched tiny kernels for every generated token: batch size 1, top-k 40, depth 30. That is too little work per launch, so launch overhead and under-occupancy dominate. Batched replay exposes all 200 token steps at once, so the MiMC hash kernels see roughly 200 contexts and 8,000 candidate rows instead of one context and 40 candidates per call.

The split-context change is important in batched mode: the context hash is independent of candidate token and key, so computing it once per token step avoids recomputing the same context 40 times. This is why the batched split-context result is much faster than simply flattening all steps through the old candidate kernel.

WDT is not changed by this WET replay work. It remains on the fused Rust detector-score path, which is already faster than original SynthID WDT in this environment.

## CUDA C++ Status

`torch` reports CUDA runtime support and Numba CUDA works, but `nvcc` is not available on `PATH` or through `conda run -n baseline_wm nvcc --version`. The environment has CUDA runtime packages and `ptxas`, so CUDA C++ extension work is still possible after installing or exposing a full nvcc toolkit. Because nvcc was unavailable, this round implemented the persistent-kernel-adjacent batching idea in the existing Numba/PyTorch stack first.

## Artifacts

- Summary CSV: `{SUMMARY_CSV.relative_to(REPO_ROOT)}`
- Markdown report: `{DOC_MD.relative_to(REPO_ROOT)}`
- HTML report: `{DOC_HTML.relative_to(REPO_ROOT)}`
- Backup of touched files before this round: `tests/baseline_comparison/hash_synthid_gpu_batch_backup_20260525_*`
"""


def build_html(rows: list[dict[str, Any]], equivalence: dict[str, Any]) -> str:
    original = rows[0]
    main_rows = [
        [
            row["scheme"],
            row["wet_semantics"],
            fmt_ms(row["wet_mean_ms_per_200"]),
            fmt_ms(row["wet_median_ms_per_200"]),
            fmt_token(row["wet_mean_ms_per_token"]),
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
            str(row["batched_wet_replay"]),
            row["equivalence"],
            row["artifact"],
        ]
        for row in rows
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hash-based SynthID GPU Batched WET Results</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
    .note {{ background: #f8fafc; border-left: 4px solid #64748b; padding: 10px 14px; }}
  </style>
</head>
<body>
  <h1>Hash-based SynthID GPU Batched WET Results, 2026-05-25</h1>
  <h2>Summary</h2>
  <p>This round adds an offline multi-token GPU replay path for hash-based SynthID MiMC T5 via <code>--batched-wet-replay</code>.</p>
  <p>The formal 20-run batched replay WET is <strong>21.44 ms</strong> per 200-token sample, compared with <strong>396.88 ms</strong> for original SynthID in the same environment.</p>
  <h2>Semantics</h2>
  <p class="note">The batched number is valid for the existing precomputed-logit WET replay metric. It is not true online generation latency, because future logits and contexts are known in replay but not during generation.</p>
  <h2>Equivalence Check</h2>
  <ul>
    <li>Top-k indices equal: <code>{html.escape(str(equivalence["top_k_indices_equal"]))}</code></li>
    <li>Original top-k score max diff: <code>{html.escape(str(equivalence["original_top_k_score_max_diff"]))}</code></li>
    <li>Updated score max diff: <code>{html.escape(str(equivalence["updated_score_max_diff"]))}</code></li>
    <li>Updated score allclose: <code>{html.escape(str(equivalence["updated_score_allclose"]))}</code></li>
  </ul>
  <h2>Results</h2>
  {html_table(["Scheme", "WET semantics", "WET mean / 200", "WET median / 200", "WET mean / token", "WDT mean / 200", "WDT median / 200", "WET speed vs original", "WDT speed vs original"], main_rows)}
  <h2>Run Details</h2>
  {html_table(["Scheme", "GPU hash", "GPU fused score", "Batched WET", "Equivalence", "Artifact"], detail_rows)}
  <h2>Implementation Notes</h2>
  <ul>
    <li>Vectorized replay context construction with <code>unfold</code>.</li>
    <li>Flattened <code>[step, batch]</code> into a larger GPU workload.</li>
    <li>Computed context hashes once per token step, then candidate/key g-values for all top-k continuations.</li>
    <li>Added GPU batched repetition-mask logic for replay order.</li>
    <li>Kept WDT on the fused Rust detector-score path.</li>
  </ul>
</body>
</html>
"""


def main() -> None:
    rows = [timing_row(item) for item in RUNS]
    equivalence = load_json(EQUIVALENCE_JSON)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    DOC_MD.write_text(build_markdown(rows, equivalence), encoding="utf-8")
    DOC_HTML.write_text(build_html(rows, equivalence), encoding="utf-8")
    print(SUMMARY_CSV)
    print(DOC_MD)
    print(DOC_HTML)


if __name__ == "__main__":
    main()
