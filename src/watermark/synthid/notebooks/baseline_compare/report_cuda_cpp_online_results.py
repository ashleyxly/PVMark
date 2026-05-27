from __future__ import annotations

import html
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "tests" / "baseline_comparison" / "hash_synthid_cuda_cpp_online_20260525"
DOC_MD = REPO_ROOT / "docs" / "hash_synthid_true_online_gpu_results_2026-05-25.md"
DOC_HTML = REPO_ROOT / "docs" / "hash_synthid_true_online_gpu_results_2026-05-25.html"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_ms(value: float) -> str:
    return f"{value:.2f} ms"


def fmt_x(value: float) -> str:
    return f"{value:.2f}x"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def main() -> None:
    formal = read_json(OUT_DIR / "history_fused_online_formal" / "efficiency_hash_synthid_timing.json")
    cpp_solo = read_json(OUT_DIR / "cuda_cpp_online_solo" / "efficiency_hash_synthid_timing.json")
    eq_cpp = read_json(OUT_DIR / "equivalence.json")
    eq_history = read_json(OUT_DIR / "equivalence_history_fused.json")

    formal_wet = formal["wet"]["200"]
    formal_wdt = formal["wdt"]["200"]
    cpp_wet = cpp_solo["wet"]["200"]

    original_wet_ms = 396.88276210799813
    original_wdt_ms = 31.545541621744633
    prior_online_wet_ms = 607.40

    rows = [
        [
            "Original SynthID non-hash LCG",
            "online sequential replay",
            fmt_ms(original_wet_ms),
            fmt_ms(original_wdt_ms),
            "1.00x",
            "1.00x",
            "Reference from same GPU environment",
        ],
        [
            "Hash-based SynthID MiMC GPU, previous online",
            "online sequential replay",
            "607.40 ms",
            "8-9 ms class",
            fmt_x(original_wet_ms / prior_online_wet_ms),
            "~3.6x",
            "Existing Numba GPU hash + fused score path",
        ],
        [
            "Hash-based SynthID MiMC GPU, history-fused online",
            "online sequential replay",
            fmt_ms(formal_wet["mean_ms_per_batch"]),
            fmt_ms(formal_wdt["mean_ms_per_batch"]),
            fmt_x(original_wet_ms / formal_wet["mean_ms_per_batch"]),
            fmt_x(original_wdt_ms / formal_wdt["mean_ms_per_batch"]),
            "Final retained true-online path",
        ],
        [
            "Hash-based SynthID MiMC CUDA C++ single-kernel online",
            "online sequential replay",
            fmt_ms(cpp_wet["mean_ms_per_batch"]),
            "not timed",
            fmt_x(original_wet_ms / cpp_wet["mean_ms_per_batch"]),
            "-",
            "Equivalent but slower; not retained",
        ],
    ]

    detail_rows = [
        [
            "Numba history-fused online",
            str(eq_history["top_k_indices_equal"]),
            str(eq_history["original_top_k_score_max_diff"]),
            str(eq_history["updated_score_max_diff"]),
            str(eq_history["updated_score_allclose"]),
            str(OUT_DIR / "equivalence_history_fused.json"),
        ],
        [
            "CUDA C++ single-kernel online",
            str(eq_cpp["top_k_indices_equal"]),
            str(eq_cpp["original_top_k_score_max_diff"]),
            str(eq_cpp["updated_score_max_diff"]),
            str(eq_cpp["updated_score_allclose"]),
            str(OUT_DIR / "equivalence.json"),
        ],
    ]

    changed_files = [
        "src/synthid_text/gpu_hash.py",
        "src/synthid_text/logits_processing.py",
        "src/synthid_text/cuda_hash_cpp.py",
        "src/synthid_text/cuda_hash_ext.cpp",
        "src/synthid_text/cuda_hash_ext_kernel.cu",
        "notebooks/baseline_compare/time_hash_synthid_efficiency.py",
        "notebooks/baseline_compare/verify_cuda_cpp_online_wet.py",
        "notebooks/baseline_compare/verify_gpu_history_fused_online_wet.py",
    ]

    md = f"""# Hash-based SynthID true-online GPU WET optimization

Date: 2026-05-25

This report covers only the true online WET path: 200 sequential `watermarked_call()` decisions with precomputed logits. It does not use batched WET replay and does not use future token contexts.

## Result summary

{markdown_table(["Scheme", "WET semantics", "WET mean / 200", "WDT mean / 200", "WET speed vs original", "WDT speed vs original", "Note"], rows)}

The retained implementation is the Numba GPU history-fused online path. Its formal 20-run result is `{fmt_ms(formal_wet["mean_ms_per_batch"])}` WET and `{fmt_ms(formal_wdt["mean_ms_per_batch"])}` WDT for 200 tokens. It is slightly faster than the previous online MiMC GPU result (`607.40 ms`) but still slower than Original SynthID (`396.88 ms`) because true online MiMC still launches a small candidate-parallel hash workload at every generated token.

## Equivalence checks

{markdown_table(["Path", "Top-k equal", "Original score max diff", "Updated score max diff", "Allclose", "Artifact"], detail_rows)}

## What was implemented

- Added `GPU_FUSED_HISTORY_UPDATE` and `--gpu-fused-history-update`.
- Added `gpu_hash.compute_g_values_and_repetition_use_mimc_gpu(...)` to the true-online WET path in `watermarked_call()`.
- The retained path fuses MiMC context hashing, context repetition lookup, and circular history write into the candidate-parallel g-value kernel.
- It keeps the existing separate GPU score-update kernel, because that kernel is cheap and this preserves candidate-level MiMC parallelism.
- Added a CUDA C++ true-online fused kernel and `--cuda-cpp-online-wet` for evaluation.
- The CUDA C++ single-kernel path computes context hash, repetition/history update, MiMC g-values, and score update in one launch.
- The CUDA C++ single-kernel path is equivalent but slower for `batch_size=1, top_k=40` because it maps one online step to one block per batch item, which underutilizes the GPU. The Numba candidate-parallel path maps one online step to roughly `batch_size * top_k` candidate blocks, which exposes more MiMC parallelism.

## Current code logic

For `HASH_TYPE == 5`, `RUST_FUSED_G_VALUES=True`, `GPU_HASH_BACKEND=True`, `GPU_FUSED_SCORE_UPDATE=True`:

1. `watermarked_call()` computes top-k scores and indices.
2. If `CUDA_CPP_ONLINE_WET=True`, it calls `cuda_hash_cpp.compute_online_updated_scores_use_mimc_cpp(...)`. This path is available for analysis but is not the fastest retained path.
3. If `GPU_FUSED_HISTORY_UPDATE=True`, it calls `gpu_hash.compute_g_values_and_repetition_use_mimc_gpu(...)`.
4. That Numba kernel computes the MiMC context hash from the current online context, checks `context_history_gpu_limbs`, writes the current context hash into the circular history slot, and emits `g_values` plus `repeated_flags`.
5. `gpu_hash.update_scores_gpu(...)` updates top-k scores from `g_values` and returns original scores unchanged for repeated contexts.
6. The processor increments `context_history_gpu_index` after each true online step.
7. WDT remains on the fused Rust detector-score path (`--fused-detect-g-values --fast-context-mask --fused-detector-score`).

## Artifacts

- Formal timing JSON: `{OUT_DIR / "history_fused_online_formal" / "efficiency_hash_synthid_timing.json"}`
- CUDA C++ online timing JSON: `{OUT_DIR / "cuda_cpp_online_solo" / "efficiency_hash_synthid_timing.json"}`
- Numba history-fused equivalence JSON: `{OUT_DIR / "equivalence_history_fused.json"}`
- CUDA C++ online equivalence JSON: `{OUT_DIR / "equivalence.json"}`
- Backup before this round: `tests/baseline_comparison/hash_synthid_online_fused_backup_20260525_143831/`

## Changed files

{chr(10).join(f"- `{path}`" for path in changed_files)}
"""

    DOC_MD.write_text(md, encoding="utf-8")

    css = """
body { font-family: Arial, sans-serif; margin: 32px; color: #17202a; line-height: 1.45; }
h1, h2 { color: #102a43; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; }
th, td { border: 1px solid #c8d0d9; padding: 8px 10px; text-align: left; vertical-align: top; }
th { background: #eef2f7; }
code { background: #f3f5f7; padding: 1px 4px; border-radius: 4px; }
ul { margin-top: 8px; }
.note { background: #f7fbff; border-left: 4px solid #2f80ed; padding: 10px 12px; }
"""
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hash-based SynthID true-online GPU WET optimization</title>
  <style>{css}</style>
</head>
<body>
  <h1>Hash-based SynthID true-online GPU WET optimization</h1>
  <p>Date: 2026-05-25</p>
  <p class="note">This report covers true online WET only: 200 sequential <code>watermarked_call()</code> decisions with precomputed logits. It does not use batched WET replay.</p>
  <h2>Result summary</h2>
  {html_table(["Scheme", "WET semantics", "WET mean / 200", "WDT mean / 200", "WET speed vs original", "WDT speed vs original", "Note"], rows)}
  <p>The retained implementation is the Numba GPU history-fused online path. Its formal 20-run result is <strong>{fmt_ms(formal_wet["mean_ms_per_batch"])}</strong> WET and <strong>{fmt_ms(formal_wdt["mean_ms_per_batch"])}</strong> WDT for 200 tokens.</p>
  <h2>Equivalence checks</h2>
  {html_table(["Path", "Top-k equal", "Original score max diff", "Updated score max diff", "Allclose", "Artifact"], detail_rows)}
  <h2>What was implemented</h2>
  <ul>
    <li>Added <code>GPU_FUSED_HISTORY_UPDATE</code> and <code>--gpu-fused-history-update</code>.</li>
    <li>Retained path fuses MiMC context hashing, repetition lookup, and circular history write into the candidate-parallel g-value kernel.</li>
    <li>Kept the separate GPU score-update kernel to preserve candidate-level MiMC parallelism.</li>
    <li>Added CUDA C++ <code>--cuda-cpp-online-wet</code> for a true-online single-kernel experiment.</li>
    <li>The CUDA C++ single-kernel path is equivalent but slower for <code>batch_size=1, top_k=40</code> due to under-occupancy.</li>
  </ul>
  <h2>Current code logic</h2>
  <ol>
    <li><code>watermarked_call()</code> computes top-k scores and indices.</li>
    <li>If <code>CUDA_CPP_ONLINE_WET=True</code>, it calls the CUDA C++ true-online fused op. This path is retained for analysis, not selected as fastest.</li>
    <li>If <code>GPU_FUSED_HISTORY_UPDATE=True</code>, it calls <code>gpu_hash.compute_g_values_and_repetition_use_mimc_gpu(...)</code>.</li>
    <li>The Numba kernel computes the MiMC context hash, checks and updates GPU context history, and emits <code>g_values</code> plus repetition flags.</li>
    <li><code>gpu_hash.update_scores_gpu(...)</code> applies the SynthID score recurrence and preserves original scores for repeated contexts.</li>
    <li>WDT uses the fused Rust detector-score path.</li>
  </ol>
  <h2>Artifacts</h2>
  <ul>
    <li>Formal timing JSON: <code>{html.escape(str(OUT_DIR / "history_fused_online_formal" / "efficiency_hash_synthid_timing.json"))}</code></li>
    <li>CUDA C++ online timing JSON: <code>{html.escape(str(OUT_DIR / "cuda_cpp_online_solo" / "efficiency_hash_synthid_timing.json"))}</code></li>
    <li>Numba history-fused equivalence JSON: <code>{html.escape(str(OUT_DIR / "equivalence_history_fused.json"))}</code></li>
    <li>CUDA C++ online equivalence JSON: <code>{html.escape(str(OUT_DIR / "equivalence.json"))}</code></li>
    <li>Backup before this round: <code>tests/baseline_comparison/hash_synthid_online_fused_backup_20260525_143831/</code></li>
  </ul>
  <h2>Changed files</h2>
  <ul>{''.join(f'<li><code>{html.escape(path)}</code></li>' for path in changed_files)}</ul>
</body>
</html>
"""
    DOC_HTML.write_text(html_doc, encoding="utf-8")
    print(DOC_MD)
    print(DOC_HTML)


if __name__ == "__main__":
    main()
