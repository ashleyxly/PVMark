from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "baseline_comparison" / "hash_synthid_variants_true_online_20260525"
SUMMARY_JSON = OUT_DIR / "true_online_variant_wet_summary_2026-05-25.json"
DOC_MD = REPO_ROOT / "docs" / "hash_synthid_variants_true_online_wet_2026-05-25.md"
DOC_HTML = REPO_ROOT / "docs" / "hash_synthid_variants_true_online_wet_2026-05-25.html"


def load_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text(encoding="utf-8"))


def fmt_ms(value: float | None) -> str:
  if value is None:
    return "-"
  return f"{value:.2f} ms"


def fmt_speed(value: float | None) -> str:
  if value is None:
    return "-"
  if value >= 1:
    return f"{value:.2f}x"
  return f"{value:.2f}x"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
  lines = [
      "| " + " | ".join(headers) + " |",
      "| " + " | ".join(["---"] * len(headers)) + " |",
  ]
  for row in rows:
    lines.append("| " + " | ".join(row) + " |")
  return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
  header = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
  body = "\n".join(
      "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>"
      for row in rows
  )
  return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def row_cells(row: dict[str, Any]) -> list[str]:
  return [
      row["scheme"],
      str(row.get("hash_type") or "original"),
      fmt_ms(row["mean_ms_per_200"]),
      fmt_ms(row["median_ms_per_200"]),
      fmt_ms(row["p90_ms_per_200"]),
      fmt_speed(row["mean_speed_vs_original"]),
      fmt_speed(row["median_speed_vs_original"]),
      str(row["compile_update_scores"]),
      str(row["gpu_hash_backend"]),
  ]


def build_markdown(summary: dict[str, Any]) -> str:
  rows = summary["rows"]
  table_rows = [row_cells(row) for row in rows]
  headers = [
      "Scheme",
      "Hash type",
      "Mean / 200",
      "Median / 200",
      "P90 / 200",
      "Mean speed vs original",
      "Median speed vs original",
      "compile",
      "GPU hash",
  ]
  best_median = min(rows[1:4], key=lambda row: row["median_ms_per_200"])
  best_mean = min(rows[1:4], key=lambda row: row["mean_ms_per_200"])
  return f"""# Hash-based SynthID true-online WET variants, 2026-05-25

This report only covers true-online WET: 200 sequential `watermarked_call()` decisions with precomputed GPT-2 logits. It does not use batched replay.

## Final Results

{md_table(headers, table_rows)}

Key readout:

- Best hash-based median: `{best_median["scheme"]}` at `{fmt_ms(best_median["median_ms_per_200"])}` for 200 online steps.
- Best hash-based mean: `{best_mean["scheme"]}` at `{fmt_ms(best_mean["mean_ms_per_200"])}` for 200 online steps.
- Original SynthID benchmark: `{fmt_ms(rows[0]["mean_ms_per_200"])}` mean, `{fmt_ms(rows[0]["median_ms_per_200"])}` median.
- Poseidon2 T4 is in the same WET order as Original SynthID; its median is slightly faster in this run, while its mean is affected by tail jitter.
- MiMC GPU true-online is implemented and measured, but is slower than MiMC Rust for `batch_size=1, top_k=40` because each generated token launches tiny under-occupied kernels.

## Optimizations Applied

- Rust fused g-values for all three variants: each online step computes context hash, candidate hash, key hash, and g-values in one Rust call returning a flat buffer.
- CPU-side online context for Rust hash paths: for Poseidon/Poseidon2/MiMC Rust paths, the `ngram_len - 1` context is maintained as a small NumPy array, avoiding unnecessary CUDA context tensor slicing and host round-trips before Rust hashing.
- Context history key normalization: Rust context hashes are stored as canonical strings instead of parsing 254-bit decimal strings into Python integers on every step.
- MiMC GPU true-online path: MiMC can run hash, context-history update, and score update through CUDA/Numba kernels with `--gpu-hash --gpu-fused-score-update --gpu-fused-history-update`.
- `torch.compile` score update is used for the reported optimized Rust paths to reduce repeated small Torch operations around the top-k score recurrence.

## Current Code Logic

In `SynthIDLogitsProcessor.watermarked_call()`:

1. The logits are temperature-scaled and reduced to top-k candidates.
2. For non-GPU Rust hash paths, the online context is updated in `state.context_cpu`; for MiMC GPU, CUDA tensors are retained.
3. `_compute_g_values()` dispatches by `HASH_TYPE`: `3` Poseidon, `4` Poseidon2, `5` MiMC.
4. The Rust fused helper returns `g_values` plus the context hash for repetition checking.
5. The optimized score update runs with `torch.compile` unless the MiMC GPU path is enabled.
6. Repeated contexts skip watermarking exactly as before.

## Artifacts

- Summary JSON: `{SUMMARY_JSON.relative_to(REPO_ROOT)}`
- Summary CSV: `{(OUT_DIR / "true_online_variant_wet_summary_2026-05-25.csv").relative_to(REPO_ROOT)}`
- Poseidon final JSON: `tests/baseline_comparison/hash_synthid_variants_true_online_20260525/poseidon_compile_final/efficiency_hash_synthid_timing.json`
- Poseidon2 final JSON: `tests/baseline_comparison/hash_synthid_variants_true_online_20260525/poseidon2_compile_final/efficiency_hash_synthid_timing.json`
- MiMC Rust final JSON: `tests/baseline_comparison/hash_synthid_variants_true_online_20260525/mimc_rust_compile_final/efficiency_hash_synthid_timing.json`
- MiMC GPU final JSON: `tests/baseline_comparison/hash_synthid_variants_true_online_20260525/mimc_gpu_online_formal/efficiency_hash_synthid_timing.json`
- Backup before this round: `tests/baseline_comparison/hash_synthid_variants_true_online_backup_20260525_151352/`
"""


def build_html(summary: dict[str, Any]) -> str:
  rows = summary["rows"]
  table_rows = [row_cells(row) for row in rows]
  headers = [
      "Scheme",
      "Hash type",
      "Mean / 200",
      "Median / 200",
      "P90 / 200",
      "Mean speed vs original",
      "Median speed vs original",
      "compile",
      "GPU hash",
  ]
  best_median = min(rows[1:4], key=lambda row: row["median_ms_per_200"])
  best_mean = min(rows[1:4], key=lambda row: row["mean_ms_per_200"])
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hash-based SynthID true-online WET variants</title>
  <style>
    body {{ font-family: system-ui, sans-serif; line-height: 1.5; margin: 40px; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Hash-based SynthID true-online WET variants, 2026-05-25</h1>
  <p>This report only covers true-online WET: 200 sequential <code>watermarked_call()</code> decisions with precomputed GPT-2 logits. It does not use batched replay.</p>
  <h2>Final Results</h2>
  {html_table(headers, table_rows)}
  <h2>Key Readout</h2>
  <ul>
    <li>Best hash-based median: <code>{html.escape(best_median["scheme"])}</code> at <code>{fmt_ms(best_median["median_ms_per_200"])}</code>.</li>
    <li>Best hash-based mean: <code>{html.escape(best_mean["scheme"])}</code> at <code>{fmt_ms(best_mean["mean_ms_per_200"])}</code>.</li>
    <li>Original SynthID benchmark: <code>{fmt_ms(rows[0]["mean_ms_per_200"])}</code> mean, <code>{fmt_ms(rows[0]["median_ms_per_200"])}</code> median.</li>
    <li>Poseidon2 T4 is in the same WET order as Original SynthID; its median is slightly faster in this run.</li>
    <li>MiMC GPU true-online is implemented but slower than MiMC Rust at <code>batch_size=1, top_k=40</code> due to tiny under-occupied kernels.</li>
  </ul>
  <h2>Optimizations Applied</h2>
  <ul>
    <li>Rust fused g-values for Poseidon, Poseidon2, and MiMC.</li>
    <li>CPU-side online context for Rust hash paths.</li>
    <li>String context-history keys instead of per-step 254-bit decimal parsing.</li>
    <li>MiMC GPU true-online hash/history/score path.</li>
    <li><code>torch.compile</code> score-update path for optimized Rust measurements.</li>
  </ul>
  <h2>Current Code Logic</h2>
  <ol>
    <li>Temperature-scale logits and select top-k candidates.</li>
    <li>Maintain CPU context for Rust hash paths and CUDA context for MiMC GPU.</li>
    <li>Dispatch fused g-value computation by hash type.</li>
    <li>Update top-k scores through compiled Torch or MiMC GPU score update.</li>
    <li>Apply repeated-context skip logic.</li>
  </ol>
  <h2>Artifacts</h2>
  <ul>
    <li>Summary JSON: <code>{html.escape(str(SUMMARY_JSON.relative_to(REPO_ROOT)))}</code></li>
    <li>Summary CSV: <code>{html.escape(str((OUT_DIR / "true_online_variant_wet_summary_2026-05-25.csv").relative_to(REPO_ROOT)))}</code></li>
    <li>Backup: <code>tests/baseline_comparison/hash_synthid_variants_true_online_backup_20260525_151352/</code></li>
  </ul>
</body>
</html>
"""


def main() -> None:
  summary = load_json(SUMMARY_JSON)
  DOC_MD.write_text(build_markdown(summary), encoding="utf-8")
  DOC_HTML.write_text(build_html(summary), encoding="utf-8")
  print(DOC_MD)
  print(DOC_HTML)


if __name__ == "__main__":
  main()
