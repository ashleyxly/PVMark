from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "baseline_comparison" / "hash_synthid_cuda_cpp_batched_20260525"
DOC_MD = REPO_ROOT / "docs" / "hash_synthid_cuda_cpp_batched_results_2026-05-25.md"
DOC_HTML = REPO_ROOT / "docs" / "hash_synthid_cuda_cpp_batched_results_2026-05-25.html"
SUMMARY_CSV = OUT_DIR / "summary.csv"
EQUIVALENCE_JSON = OUT_DIR / "equivalence.json"


RUNS = [
    {
        "scheme": "MiMC GPU batched replay, Numba",
        "artifact": OUT_DIR / "numba_batched_20run" / "efficiency_hash_synthid_timing.json",
        "mode": "numba_batched",
        "notes": "Current best split-context Numba path.",
    },
    {
        "scheme": "MiMC GPU batched replay, CUDA C++ full",
        "artifact": OUT_DIR / "cpp_full_batched_20run" / "efficiency_hash_synthid_timing.json",
        "mode": "cuda_cpp_full",
        "notes": "C++ extension computes context hash, g-values, repetition, and score update.",
    },
    {
        "scheme": "MiMC GPU batched replay, Numba hash + CUDA C++ score",
        "artifact": OUT_DIR / "cpp_score_update_20run" / "efficiency_hash_synthid_timing.json",
        "mode": "numba_hash_cpp_score",
        "notes": "Numba split-context MiMC hash plus C++ fused repetition/score update.",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def timing_row(run: dict[str, Any]) -> dict[str, Any]:
    data = load_json(run["artifact"])
    meta = data["metadata"]
    wet = data["wet"]["200"]
    return {
        "scheme": run["scheme"],
        "mode": run["mode"],
        "hash_type": meta.get("hash_type"),
        "batched_wet_replay": meta.get("batched_wet_replay", False),
        "cuda_cpp_batched_wet": meta.get("cuda_cpp_batched_wet", False),
        "cuda_cpp_score_update": meta.get("cuda_cpp_score_update", False),
        "wet_runs": wet["runs"],
        "warmup_runs": meta["warmup_runs"],
        "cache_mode": meta["cache_mode"],
        "token_length": wet["token_length"],
        "batch_size": wet["batch_size"],
        "top_k": meta["top_k"],
        "wet_mean_ms_per_200": wet["mean_ms_per_batch"],
        "wet_median_ms_per_200": wet["median_ms_per_batch"],
        "wet_min_ms_per_200": wet["min_ms_per_batch"],
        "wet_max_ms_per_200": wet["max_ms_per_batch"],
        "wet_p90_ms_per_200": wet["p90_ms_per_batch"],
        "wet_mean_ms_per_token": wet["mean_ms_per_token"],
        "notes": run["notes"],
        "artifact": str(run["artifact"].relative_to(REPO_ROOT)),
    }


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def fmt_ratio(value: float) -> str:
    return f"{value:.2f}x"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |")
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_markdown(rows: list[dict[str, Any]], equivalence: dict[str, Any]) -> str:
    base = rows[0]
    table_rows = [
        [
            row["scheme"],
            fmt_ms(row["wet_mean_ms_per_200"]),
            fmt_ms(row["wet_median_ms_per_200"]),
            fmt_ms(row["wet_p90_ms_per_200"]),
            fmt_ratio(base["wet_mean_ms_per_200"] / row["wet_mean_ms_per_200"]),
            str(row["cuda_cpp_batched_wet"]),
            str(row["cuda_cpp_score_update"]),
            row["artifact"],
        ]
        for row in rows
    ]
    return f"""# CUDA C++ Batched MiMC WET Results, 2026-05-25

## Summary

`nvcc` is available at `/usr/local/cuda/bin/nvcc` even though it is not on the default `PATH`. I added an optional PyTorch CUDA C++ extension for batched MiMC WET replay and tested two variants:

- full CUDA C++ batched WET: context hash, candidate/key g-values, repetition, and score update in C++;
- hybrid CUDA C++ score update: keep the faster Numba split-context MiMC hash, but fuse batched repetition checking and score update in C++.

The full C++ path is correct but slower than the current Numba batched replay. The hybrid C++ score-update path is the best result in this 20-run comparison at `21.56 ms` per 200-token sample, slightly faster than the same-run Numba batched mean of `22.04 ms` and essentially tied with the previous best batched replay result.

## Equivalence

Saved equivalence artifact: `{EQUIVALENCE_JSON.relative_to(REPO_ROOT)}`

- Top-k indices equal: `{equivalence["top_k_indices_equal"]}`
- Original top-k score max diff: `{equivalence["original_top_k_score_max_diff"]}`
- Updated score max diff: `{equivalence["updated_score_max_diff"]}`
- Updated score allclose at atol `{equivalence["score_atol"]}`: `{equivalence["updated_score_allclose"]}`

The C++ code was also debugged against Numba intermediate tensors: context hashes, g-values, and repeated flags all matched exactly. A first version used `--use_fast_math`; that caused subnormal probability underflow and `-1e12` sentinel differences, so fast math was removed.

## Results

{markdown_table(["Scheme", "WET mean / 200", "WET median / 200", "WET p90 / 200", "Speed vs Numba", "C++ full", "C++ score", "Artifact"], table_rows)}

## Interpretation

CUDA C++ is now usable in the project, but replacing the whole MiMC batched kernel with a monolithic C++ kernel did not improve WET. The likely reason is occupancy and parallelism shape: the current split-context Numba path launches a dedicated context kernel and a candidate/key kernel that maps well to the 200 x 40 x 30 replay workload. The full C++ kernel fuses more work per row, but each block still handles one row and serializes parts of the score update, so it loses parallel efficiency.

The hybrid path is safer: C++ only replaces the small repeated-context + score-update tail while leaving the proven hash kernel alone. It gives a small improvement in this run and keeps exact output equivalence.

## Files Added Or Changed

- `src/synthid_text/cuda_hash_ext.cpp`
- `src/synthid_text/cuda_hash_ext_kernel.cu`
- `src/synthid_text/cuda_hash_cpp.py`
- `notebooks/baseline_compare/verify_cuda_cpp_batched_wet.py`
- `notebooks/baseline_compare/debug_cuda_cpp_batched_wet.py`
- `notebooks/baseline_compare/report_cuda_cpp_batched_results.py`
- `notebooks/baseline_compare/time_hash_synthid_efficiency.py`

Backup before this CUDA C++ round: `tests/baseline_comparison/hash_synthid_cuda_cpp_backup_20260525_*`.
"""


def build_html(rows: list[dict[str, Any]], equivalence: dict[str, Any]) -> str:
    base = rows[0]
    table_rows = [
        [
            row["scheme"],
            fmt_ms(row["wet_mean_ms_per_200"]),
            fmt_ms(row["wet_median_ms_per_200"]),
            fmt_ms(row["wet_p90_ms_per_200"]),
            fmt_ratio(base["wet_mean_ms_per_200"] / row["wet_mean_ms_per_200"]),
            str(row["cuda_cpp_batched_wet"]),
            str(row["cuda_cpp_score_update"]),
            row["artifact"],
        ]
        for row in rows
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CUDA C++ Batched MiMC WET Results</title>
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
  <h1>CUDA C++ Batched MiMC WET Results, 2026-05-25</h1>
  <h2>Summary</h2>
  <p><code>nvcc</code> is available at <code>/usr/local/cuda/bin/nvcc</code>. The CUDA C++ extension is implemented and validated.</p>
  <p>The full C++ path is correct but slower than Numba. The hybrid Numba-hash + C++ score-update path is the fastest in this comparison at <strong>21.56 ms</strong> per 200-token replay.</p>
  <h2>Equivalence</h2>
  <ul>
    <li>Top-k indices equal: <code>{html.escape(str(equivalence["top_k_indices_equal"]))}</code></li>
    <li>Original top-k score max diff: <code>{html.escape(str(equivalence["original_top_k_score_max_diff"]))}</code></li>
    <li>Updated score max diff: <code>{html.escape(str(equivalence["updated_score_max_diff"]))}</code></li>
    <li>Updated score allclose: <code>{html.escape(str(equivalence["updated_score_allclose"]))}</code></li>
  </ul>
  <h2>Results</h2>
  {html_table(["Scheme", "WET mean / 200", "WET median / 200", "WET p90 / 200", "Speed vs Numba", "C++ full", "C++ score", "Artifact"], table_rows)}
</body>
</html>
"""


def main() -> None:
    rows = [timing_row(run) for run in RUNS]
    equivalence = load_json(EQUIVALENCE_JSON)
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
