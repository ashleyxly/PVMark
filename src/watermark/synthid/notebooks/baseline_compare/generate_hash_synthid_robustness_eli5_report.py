#!/usr/bin/env python3
"""Generate the ELI5 SynthID robustness HTML report from saved artifacts."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "hash_synthid_robustness_eli5_2026-05-25.html"
THRESHOLD = 0.514
EXPECTED_ATTACK_RECORDS = 1000

MODELS = [
    ("GPT2", "GPT-2 on ELI5"),
    ("GEMMA_2B", "GEMMA-2B-IT on ELI5"),
]

HASH_TYPES = [
    (3, "Poseidon", "Type 3"),
    (4, "Poseidon2", "Type 4"),
    (5, "MiMC", "Type 5"),
]

ATTACKS = [
    ("word_deletion", "Word deletion", "attack1_text", "WD 0.3"),
    ("synonym_substitution", "WordNet synonym", "attack2_text", "Synonym 0.5"),
    (
        "context_aware_synonym_substitution",
        "Context-aware synonym",
        "attack3_text",
        "Context-aware 0.5",
    ),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_scores(rows: list[Any]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if isinstance(row, list):
            values.extend(
                float(value)
                for value in row
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            )
        elif isinstance(row, (int, float)) and math.isfinite(float(row)):
            values.append(float(row))
    return values


def pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def ppl(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def esc(value: Any) -> str:
    return html.escape(str(value))


def td(value: Any, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{esc(value)}</td>"


def th(value: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<th{attr}>{esc(value)}</th>"


def table(headers: list[str], rows: list[list[Any]], numeric_from: int | None = None) -> str:
    head = "".join(
        th(header, "num" if numeric_from is not None and idx >= numeric_from else "")
        for idx, header in enumerate(headers)
    )
    body_lines = []
    for row in rows:
        body = "".join(
            td(cell, "num" if numeric_from is not None and idx >= numeric_from else "")
            for idx, cell in enumerate(row)
        )
        body_lines.append(f"<tr>{body}</tr>")
    return (
        "<table>\n<thead><tr>"
        + head
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_lines)
        + "\n</tbody>\n</table>"
    )


def source_path(kind: str, hash_type: int, model: str, model_suffix: str) -> Path:
    if kind == "score":
        return (
            ROOT
            / f"tests/WM_UWM/Score/Score_Type_{hash_type}_ModelName.{model}_{model_suffix}_results.json"
        )
    if kind == "detect":
        return (
            ROOT
            / f"tests/WM_UWM/Attack/Detect/Detect_Attack_Type_{hash_type}_ModelName.{model}_{model_suffix}_results.json"
        )
    if kind == "ppl":
        return (
            ROOT
            / f"tests/WM_UWM/Attack/PPL/PPL_Attack_Type_{hash_type}_ModelName.{model}_{model_suffix}_results.json"
        )
    raise ValueError(kind)


def add_scheme_records(
    *,
    records: list[dict[str, Any]],
    clean_rows: list[dict[str, Any]],
    sources: list[Path],
    model_key: str,
    model_label: str,
    family: str,
    scheme: str,
    hash_type: int,
    model_suffix: str,
) -> None:
    score_file = source_path("score", hash_type, model_key, model_suffix)
    detect_file = source_path("detect", hash_type, model_key, model_suffix)
    ppl_file = source_path("ppl", hash_type, model_key, model_suffix)
    sources.extend([score_file, detect_file, ppl_file])

    score_data = read_json(score_file)
    detect_data = read_json(detect_file)
    ppl_data = read_json(ppl_file)

    clean_total = int(score_data["total_float_count"])
    clean_detected = int(score_data["count_over_4"])
    clean_rows.append(
        {
            "family": family,
            "model": model_label,
            "scheme": scheme,
            "detected": clean_detected,
            "total": clean_total,
            "success_rate": clean_detected / clean_total if clean_total else None,
            "score": float(score_data["average_z_score"]),
            "ppl": float(ppl_data["original_text"]["mean_perplexity"]),
            "ppl_count": len(ppl_data["original_text"]["perplexities"]),
        }
    )

    for attack_key, attack_label, ppl_key, attack_short in ATTACKS:
        scores = flatten_scores(detect_data[f"{attack_key}_weighted_mean_scores"])
        detected = sum(value >= THRESHOLD for value in scores)
        total = len(scores)
        records.append(
            {
                "family": family,
                "model": model_label,
                "model_key": model_key,
                "scheme": scheme,
                "attack": attack_label,
                "attack_short": attack_short,
                "detected": detected,
                "total": total,
                "success_rate": detected / total if total else None,
                "score": mean(scores) if scores else None,
                "ppl": float(ppl_data[ppl_key]["mean_perplexity"]),
                "ppl_count": len(ppl_data[ppl_key]["perplexities"]),
                "missing_detection": max(EXPECTED_ATTACK_RECORDS - total, 0),
            }
        )


def build_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    records: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    sources: list[Path] = []

    for model_key, model_label in MODELS:
        for hash_type, hash_name, hash_label in HASH_TYPES:
            add_scheme_records(
                records=records,
                clean_rows=clean_rows,
                sources=sources,
                model_key=model_key,
                model_label=model_label,
                family="Hash-based SynthID",
                scheme=f"{hash_label} / {hash_name}",
                hash_type=hash_type,
                model_suffix="WM",
            )

        add_scheme_records(
            records=records,
            clean_rows=clean_rows,
            sources=sources,
            model_key=model_key,
            model_label=model_label,
            family="Original Impl Benchmark",
            scheme="Original SynthID Impl / Type 3 Org_WM",
            hash_type=3,
            model_suffix="Org_WM",
        )

    return records, clean_rows, sources


def build_wide_rows(records: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    headers = ["Model / dataset", "Family", "Scheme"]
    for _, _, _, attack_short in ATTACKS:
        headers.extend(
            [f"{attack_short} success", f"{attack_short} score", f"{attack_short} PPL"]
        )

    rows: list[list[Any]] = []
    seen: list[tuple[str, str, str]] = []
    for record in records:
        key = (record["model"], record["family"], record["scheme"])
        if key not in seen:
            seen.append(key)

    for model_label, family, scheme in seen:
        subset = [
            r
            for r in records
            if r["model"] == model_label and r["family"] == family and r["scheme"] == scheme
        ]
        row: list[Any] = [model_label, family, scheme]
        for _, attack_label, _, _ in ATTACKS:
            rec = next(r for r in subset if r["attack"] == attack_label)
            row.extend(
                [
                    f"{rec['detected']}/{rec['total']} ({pct(rec['success_rate'])})",
                    num(rec["score"]),
                    ppl(rec["ppl"]),
                ]
            )
        rows.append(row)
    return headers, rows


def build_hash_macro_rows(records: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for _, model_label in MODELS:
        for _, attack_label, _, attack_short in ATTACKS:
            subset = [
                r
                for r in records
                if r["model"] == model_label
                and r["attack"] == attack_label
                and r["family"] == "Hash-based SynthID"
            ]
            rows.append(
                [
                    model_label,
                    attack_short,
                    pct(mean(r["success_rate"] for r in subset if r["success_rate"] is not None)),
                    num(mean(r["score"] for r in subset if r["score"] is not None)),
                    ppl(mean(r["ppl"] for r in subset if r["ppl"] is not None)),
                ]
            )
    return rows


def render_report(records: list[dict[str, Any]], clean_rows: list[dict[str, Any]], sources: list[Path]) -> str:
    wide_headers, wide_rows = build_wide_rows(records)
    macro_rows = build_hash_macro_rows(records)

    benchmark_rows = [
        [
            r["model"],
            r["scheme"],
            r["attack"],
            f"{r['detected']}/{r['total']} ({pct(r['success_rate'])})",
            num(r["score"]),
            ppl(r["ppl"]),
        ]
        for r in records
        if r["family"] == "Original Impl Benchmark"
    ]

    clean_table_rows = [
        [
            row["model"],
            row["family"],
            row["scheme"],
            f"{row['detected']}/{row['total']} ({pct(row['success_rate'])})",
            num(row["score"]),
            ppl(row["ppl"]),
            row["ppl_count"],
        ]
        for row in clean_rows
    ]

    detail_rows = [
        [
            r["model"],
            r["family"],
            r["scheme"],
            r["attack"],
            f"{r['detected']}/{r['total']}",
            pct(r["success_rate"]),
            num(r["score"]),
            ppl(r["ppl"]),
            r["ppl_count"],
            r["missing_detection"],
        ]
        for r in records
    ]

    source_items = "\n".join(
        f"<li><code>{esc(str(path.relative_to(ROOT)))}</code></li>"
        for path in sorted(set(sources))
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SynthID Robustness on ELI5</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d252d;
      --muted: #5e6b78;
      --line: #d9dee5;
      --accent: #0f766e;
      --accent-2: #8a5a00;
      --head: #e9f3f2;
      --code: #eef0f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    header {{
      background: #12322f;
      color: #fff;
      padding: 32px 24px 26px;
      border-bottom: 5px solid var(--accent);
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 30px 0 12px; font-size: 20px; }}
    p {{ margin: 8px 0; }}
    .subtitle {{ color: #cfe1df; max-width: 1040px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 4px 10px;
      border: 1px solid rgba(255, 255, 255, .24);
      background: rgba(255,255,255,.08);
      color: #fff;
      border-radius: 4px;
      white-space: nowrap;
    }}
    section {{ margin: 0 0 28px; }}
    .note {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-left: 4px solid var(--accent-2);
      padding: 14px 16px;
      margin: 18px 0;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; margin: 18px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); padding: 14px 16px; border-radius: 6px; }}
    .metric b {{ display: block; font-size: 13px; color: var(--muted); font-weight: 600; }}
    .metric span {{ display: block; margin-top: 4px; font-size: 20px; font-weight: 700; }}
    .table-wrap {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1080px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ background: var(--head); text-align: left; font-size: 13px; color: #173b38; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: #fbfcfd; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    code {{ background: var(--code); padding: 1px 4px; border-radius: 3px; }}
    ul {{ margin: 8px 0 0 20px; padding: 0; }}
    li {{ margin: 4px 0; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 780px) {{
      main {{ padding: 16px; }}
      header {{ padding: 24px 16px; }}
      h1 {{ font-size: 24px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .pill {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SynthID Robustness on ELI5</h1>
    <p class="subtitle">Saved robustness artifacts for GPT-2 on ELI5 and GEMMA-2B-IT on ELI5. The report covers hash-based SynthID variants and the original implementation benchmark, without rerunning experiments.</p>
    <div class="meta">
      <span class="pill">Generated: 2026-05-25</span>
      <span class="pill">Detection threshold: weighted mean score &gt;= {THRESHOLD}</span>
      <span class="pill">Schemes: hash-based variants + original implementation benchmark</span>
    </div>
  </header>
  <main>
    <section>
      <h2>Metric Definitions</h2>
      <div class="note">
        <p><b>Success rate</b> is the post-attack watermark detection rate: attacked watermarked text is still detected as watermarked.</p>
        <p><b>Watermark score</b> is the mean saved SynthID <code>weighted_mean_score</code> over attacked watermarked texts. Historical JSON files sometimes call clean scores <code>average_z_score</code>, but these values are weighted mean g-value scores.</p>
        <p><b>PPL</b> is the saved mean perplexity for the attacked text artifact. The PPL JSON files do not store evaluator metadata, so this report only reproduces the saved values.</p>
      </div>
      <div class="note">
        <p><b>Benchmark coverage:</b> the original implementation benchmark uses the complete saved <code>Type_3_ModelName.*_Org_WM</code> score, detect, and PPL artifacts. Local Type 4/5 <code>Org_WM</code> attack texts exist, but matching saved detect/PPL/score artifacts were not present, so they are not reported here.</p>
      </div>
      <div class="grid">
        <div class="metric"><b>Rows summarized</b><span>{len(records)}</span></div>
        <div class="metric"><b>Models / datasets</b><span>2</span></div>
        <div class="metric"><b>Attack methods</b><span>3</span></div>
      </div>
    </section>

    <section>
      <h2>Robustness Summary</h2>
      <p class="small">Each row reports success / watermark score / PPL for texts modified by the three attacks.</p>
      <div class="table-wrap">
        {table(wide_headers, wide_rows, numeric_from=3)}
      </div>
    </section>

    <section>
      <h2>Original Impl Benchmark</h2>
      <p class="small">Same metrics as the hash-based rows, shown separately for quick benchmark comparison.</p>
      <div class="table-wrap">
        {table(["Model / dataset", "Benchmark artifact", "Attack", "Success", "Watermark score", "PPL"], benchmark_rows, numeric_from=3)}
      </div>
    </section>

    <section>
      <h2>Hash-Only Macro Average By Model</h2>
      <p class="small">Unweighted average across the three hash variants for each model and attack. The original implementation benchmark is excluded from these averages.</p>
      <div class="table-wrap">
        {table(["Model / dataset", "Attack", "Success rate", "Watermark score", "PPL"], macro_rows, numeric_from=2)}
      </div>
    </section>

    <section>
      <h2>Clean Detection Context</h2>
      <p class="small">Clean rows are included only as context for the attack results.</p>
      <div class="table-wrap">
        {table(["Model / dataset", "Family", "Scheme", "Clean success", "Clean watermark score", "Clean PPL", "PPL count"], clean_table_rows, numeric_from=3)}
      </div>
    </section>

    <section>
      <h2>Detailed Rows</h2>
      <div class="table-wrap">
        {table(["Model / dataset", "Family", "Scheme", "Attack", "Detected / valid", "Success rate", "Watermark score", "PPL", "PPL count", "Missing detection rows"], detail_rows, numeric_from=4)}
      </div>
    </section>

    <section>
      <h2>Source Artifacts</h2>
      <p class="small">All values were computed from these saved files under the repository root.</p>
      <ul>
        {source_items}
      </ul>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    records, clean_rows, sources = build_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render_report(records, clean_rows, sources), encoding="utf-8")
    print(OUT)
    print(f"records={len(records)} sources={len(set(sources))}")


if __name__ == "__main__":
    main()
