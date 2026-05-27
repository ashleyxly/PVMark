#!/usr/bin/env python3
"""Generate a Markdown/HTML robustness report from saved experiment artifacts."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REPORT_DATE = "2026-05-24"
MD_OUT = DOCS / f"robustness_results_{REPORT_DATE}.md"
HTML_OUT = DOCS / f"robustness_results_{REPORT_DATE}.html"

SYNTHID_THRESHOLD = 0.514

ATTACKS = [
    ("word_deletion", "WD / word deletion 0.3", "attack1_text"),
    ("synonym_substitution", "WordNet synonym 0.5", "attack2_text"),
    ("context_aware_synonym_substitution", "Context-aware synonym 0.5", "attack3_text"),
]


def read_json(path: str | Path) -> Any:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def valid_float(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def flatten_score_rows(rows: list[Any]) -> list[float]:
    values: list[float] = []
    for row in rows:
        if isinstance(row, list):
            values.extend(float(value) for value in row if valid_float(value))
        elif valid_float(row):
            values.append(float(row))
    return values


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_ppl(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fmt_count(pos: int | None, total: int | None) -> str:
    if pos is None or total is None:
        return "N/A"
    return f"{pos}/{total}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{body}</tr>")
    return (
        "<table>\n<thead><tr>"
        + head
        + "</tr></thead>\n<tbody>\n"
        + "\n".join(body_rows)
        + "\n</tbody>\n</table>"
    )


def summarize_synthid(
    label: str,
    score_path: str,
    detect_path: str,
    ppl_path: str,
    note: str,
    expected_attack_records: int = 1000,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    score = read_json(score_path)
    detect = read_json(detect_path)
    ppl = read_json(ppl_path)

    clean_total = int(score["total_float_count"])
    clean_pos = int(score["count_over_4"])
    clean = {
        "scheme": label,
        "clean_success_rate": clean_pos / clean_total if clean_total else None,
        "clean_count": fmt_count(clean_pos, clean_total),
        "clean_score": float(score["average_z_score"]),
        "clean_ppl": float(ppl["original_text"]["mean_perplexity"]),
        "score_definition": "SynthID weighted mean g-value; historical files call it z-score",
        "note": note,
    }

    rows: list[dict[str, Any]] = []
    for attack_key, attack_label, ppl_key in ATTACKS:
        scores = flatten_score_rows(detect[f"{attack_key}_weighted_mean_scores"])
        positive = sum(value >= SYNTHID_THRESHOLD for value in scores)
        total = len(scores)
        post_success = positive / total if total else None
        rows.append(
            {
                "scheme": label,
                "attack": attack_label,
                "attack_key": attack_key,
                "post_success": post_success,
                "attack_success": 1 - post_success if post_success is not None else None,
                "detected": positive,
                "total": total,
                "mean_score": mean(scores) if scores else None,
                "ppl": float(ppl[ppl_key]["mean_perplexity"]),
                "ppl_count": len(ppl[ppl_key]["perplexities"]),
                "detection_missing_or_error": max(expected_attack_records - total, 0),
            }
        )

    sources = [score_path, detect_path, ppl_path]
    return clean, rows, sources


def summarize_upv() -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    path = "tests/baseline_comparison/upv_network_detector_gpt2_eli5/robustness_ppl_score_summary_network_z1.json"
    data = read_json(path)
    normal = data["normal"]
    wm_det = normal["wm_detection"]
    clean = {
        "scheme": "UPV network detector",
        "clean_success_rate": wm_det["positive_rate"],
        "clean_count": fmt_count(wm_det["positive_count"], wm_det["valid_count"]),
        "clean_score": normal["wm_score"]["mean"],
        "clean_ppl": normal["wm_ppl"]["mean"],
        "score_definition": "UPV network classifier probability; threshold > 0.5",
        "note": "z_value=1 network detector artifact",
    }

    rows: list[dict[str, Any]] = []
    for attack_key, attack_label, _ in ATTACKS:
        attack = data["robustness"][attack_key]
        det = attack["wm_detection"]
        score = attack["wm_score"]
        ppl = attack["wm_ppl"]
        post_success = det["positive_rate"]
        rows.append(
            {
                "scheme": "UPV network detector",
                "attack": attack_label,
                "attack_key": attack_key,
                "post_success": post_success,
                "attack_success": 1 - post_success,
                "detected": det["positive_count"],
                "total": det["valid_count"],
                "mean_score": score["mean"],
                "ppl": ppl["mean"],
                "ppl_count": ppl["count"],
                "detection_missing_or_error": det["error_count"],
            }
        )
    return clean, rows, [path]


def summarize_pdw() -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    summary_path = "tests/baseline_comparison/pdw_gpt2/summary.json"
    ppl_path = "tests/baseline_comparison/pdw_gpt2/ppl.json"
    robustness_path = "tests/baseline_comparison/pdw_gpt2/robustness_summary.json"
    context_path = "tests/baseline_comparison/pdw_context_window_gpt2/robustness_summary.json"

    summary = read_json(summary_path)["summary"]
    ppl_records = read_json(ppl_path)["records"]
    robustness = read_json(robustness_path)["summary"]
    context = read_json(context_path)["summary"]["context_aware_synonym_substitution"]

    clean_ppls = [
        float(record["ppl"])
        for record in ppl_records
        if record.get("watermarked") is True
        and record.get("attack") is None
        and record.get("error") is None
        and valid_float(record.get("ppl"))
    ]
    wm_det = summary["effectiveness"]["wm_detection"]
    clean = {
        "scheme": "PDW publicly detectable",
        "clean_success_rate": wm_det["positive_rate"],
        "clean_count": fmt_count(wm_det["positive_count"], wm_det["valid_count"]),
        "clean_score": None,
        "clean_ppl": mean(clean_ppls) if clean_ppls else None,
        "score_definition": "PDW public verification is boolean; no continuous score saved",
        "note": f"clean WM PPL count={len(clean_ppls)}",
    }

    rows: list[dict[str, Any]] = []
    for attack_key, attack_label, _ in ATTACKS:
        attack = context if attack_key == "context_aware_synonym_substitution" else robustness[attack_key]
        det = attack["wm_detection"]
        ppl = attack["wm_ppl"]
        post_success = det["positive_rate"]
        rows.append(
            {
                "scheme": "PDW publicly detectable",
                "attack": attack_label,
                "attack_key": attack_key,
                "post_success": post_success,
                "attack_success": 1 - post_success,
                "detected": det["positive_count"],
                "total": det["valid_count"],
                "mean_score": None,
                "ppl": ppl["mean"],
                "ppl_count": ppl["count"],
                "detection_missing_or_error": det["error_count"],
            }
        )
    return clean, rows, [summary_path, ppl_path, robustness_path, context_path]


def build_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    clean_rows: list[dict[str, Any]] = []
    attack_rows: list[dict[str, Any]] = []
    sources: list[str] = []

    synthid_specs = [
        (
            "Original SynthID-Text (non-hash, Type 3 GPT-2 Org_WM)",
            "tests/WM_UWM/Score/Score_Type_3_ModelName.GPT2_Org_WM_results.json",
            "tests/WM_UWM/Attack/Detect/Detect_Attack_Type_3_ModelName.GPT2_Org_WM_results.json",
            "tests/WM_UWM/Attack/PPL/PPL_Attack_Type_3_ModelName.GPT2_Org_WM_results.json",
            "Only Type 3 Org_WM attack-detection artifact is present locally.",
        ),
        (
            "Hash SynthID Poseidon (Type 3, requested two-to-one fixed variant)",
            "tests/WM_UWM/Score/Score_Type_3_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/Detect/Detect_Attack_Type_3_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/PPL/PPL_Attack_Type_3_ModelName.GPT2_WM_results.json",
            "Saved hash_type=3 / Poseidon GPT-2 WM artifact; hash_method metadata is not stored.",
        ),
        (
            "Hash SynthID Poseidon2 (Type 4, requested two-to-one fixed variant)",
            "tests/WM_UWM/Score/Score_Type_4_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/Detect/Detect_Attack_Type_4_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/PPL/PPL_Attack_Type_4_ModelName.GPT2_WM_results.json",
            "Saved hash_type=4 / Poseidon2 GPT-2 WM artifact; hash_method metadata is not stored.",
        ),
        (
            "Hash SynthID MiMC (Type 5, requested two-to-one fixed variant)",
            "tests/WM_UWM/Score/Score_Type_5_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/Detect/Detect_Attack_Type_5_ModelName.GPT2_WM_results.json",
            "tests/WM_UWM/Attack/PPL/PPL_Attack_Type_5_ModelName.GPT2_WM_results.json",
            "Saved hash_type=5 / MiMC GPT-2 WM artifact; hash_method metadata is not stored.",
        ),
    ]

    for spec in synthid_specs:
        clean, attacks, paths = summarize_synthid(*spec)
        clean_rows.append(clean)
        attack_rows.extend(attacks)
        sources.extend(paths)

    for summarizer in (summarize_upv, summarize_pdw):
        clean, attacks, paths = summarizer()
        clean_rows.append(clean)
        attack_rows.extend(attacks)
        sources.extend(paths)

    return clean_rows, attack_rows, sorted(set(sources))


def macro_post_success(rows: list[dict[str, Any]], scheme: str) -> float | None:
    values = [
        row["post_success"]
        for row in rows
        if row["scheme"] == scheme and row["post_success"] is not None
    ]
    return mean(values) if values else None


def build_markdown(clean_rows: list[dict[str, Any]], attack_rows: list[dict[str, Any]], sources: list[str]) -> str:
    clean_table = md_table(
        [
            "Scheme",
            "Clean WM Success",
            "Detected/Valid",
            "Mean watermark score / Z-score field",
            "Clean WM PPL",
            "Score definition / note",
        ],
        [
            [
                row["scheme"],
                fmt_pct(row["clean_success_rate"]),
                row["clean_count"],
                fmt_num(row["clean_score"]),
                fmt_ppl(row["clean_ppl"]),
                f"{row['score_definition']}; {row['note']}",
            ]
            for row in clean_rows
        ],
    )

    wide_rows: list[list[str]] = []
    for clean in clean_rows:
        scheme = clean["scheme"]
        by_attack = {row["attack_key"]: row for row in attack_rows if row["scheme"] == scheme}
        wide_rows.append(
            [
                scheme,
                fmt_pct(by_attack["word_deletion"]["post_success"]),
                fmt_num(by_attack["word_deletion"]["mean_score"]),
                fmt_ppl(by_attack["word_deletion"]["ppl"]),
                fmt_pct(by_attack["synonym_substitution"]["post_success"]),
                fmt_num(by_attack["synonym_substitution"]["mean_score"]),
                fmt_ppl(by_attack["synonym_substitution"]["ppl"]),
                fmt_pct(by_attack["context_aware_synonym_substitution"]["post_success"]),
                fmt_num(by_attack["context_aware_synonym_substitution"]["mean_score"]),
                fmt_ppl(by_attack["context_aware_synonym_substitution"]["ppl"]),
                fmt_pct(macro_post_success(attack_rows, scheme)),
            ]
        )
    wide_table = md_table(
        [
            "Scheme",
            "WD Success",
            "WD Z/score",
            "WD PPL",
            "WordNet Success",
            "WordNet Z/score",
            "WordNet PPL",
            "Context Success",
            "Context Z/score",
            "Context PPL",
            "Macro Post-attack Success",
        ],
        wide_rows,
    )

    detail_table = md_table(
        [
            "Scheme",
            "Attack",
            "Post-attack Success",
            "Attack Success",
            "Detected/Valid",
            "Mean watermark score / Z-score field",
            "Attacked text PPL",
            "PPL count",
            "Detection missing/error",
        ],
        [
            [
                row["scheme"],
                row["attack"],
                fmt_pct(row["post_success"]),
                fmt_pct(row["attack_success"]),
                fmt_count(row["detected"], row["total"]),
                fmt_num(row["mean_score"]),
                fmt_ppl(row["ppl"]),
                str(row["ppl_count"]),
                str(row["detection_missing_or_error"]),
            ]
            for row in attack_rows
        ],
    )

    source_lines = "\n".join(f"- `{source}`" for source in sources)
    return f"""# Robustness Results Summary

Date: {REPORT_DATE}

This report consolidates the saved GPT-2 + ELI5 robustness artifacts under the current repository path.

## Metric Definitions

- `Post-attack Success`: attacked watermarked text is still detected as watermarked. This is the robustness rate.
- `Attack Success`: `1 - Post-attack Success`, i.e. the attack successfully removes or breaks detection.
- `Mean watermark score / Z-score field`: method-specific detector score. For SynthID this is the weighted mean g-value score; historical JSON files call it `average_z_score`, but it is not a standard statistical z-score. For UPV it is the network classifier probability. PDW has only boolean public verification, so the score is `N/A`.
- `Attacked text PPL`: perplexity of attacked watermarked completion text. UPV/PDW PPL artifacts use OPT-2.7B. SynthID historical PPL artifacts were produced by the attack PPL pipeline but do not embed the evaluator path in JSON.
- Detection thresholds: SynthID weighted mean `>= 0.514`; UPV score `> 0.5`; PDW public verification is `True`.

## Attack Setup

- `WD / word deletion 0.3`: random word deletion with deletion ratio `0.3`.
- `WordNet synonym 0.5`: random synonym substitution with WordNet and substitution ratio `0.5`.
- `Context-aware synonym 0.5`: context-aware synonym substitution with BERT and substitution ratio `0.5`. For PDW, the context-aware row uses the long-text-safe retest in `tests/baseline_comparison/pdw_context_window_gpt2/`.

## Clean Detection Context

{clean_table}

## Robustness At A Glance

{wide_table}

## Detailed Robustness Rows

{detail_table}

## Notes

- Original SynthID-Text refers to the saved non-hash `Org_WM` Type 3 GPT-2 artifact. Type 4/5 `Org_WM` attack texts exist, but matching saved attack-detection artifacts were not found locally.
- Hash-based SynthID rows use the saved GPT-2 Type 3/4/5 artifacts: Poseidon, Poseidon2, and MiMC. The result JSON files record `hash_type` but do not store a separate `hash_method`; this report groups them under the requested two-to-one fixed heading, but the saved JSON alone cannot distinguish fixed vs sort hash-method metadata.
- PDW context-aware results use the retest because the old MarkLLM context-aware attack artifact had invalid long-text BERT masking for PDW watermarked texts.
- Scores are not directly comparable across methods; compare success rates and PPL within the same attack setting.

## Source Artifacts

{source_lines}
"""


def build_html(markdown: str, clean_rows: list[dict[str, Any]], attack_rows: list[dict[str, Any]], sources: list[str]) -> str:
    clean_html = html_table(
        [
            "Scheme",
            "Clean WM Success",
            "Detected/Valid",
            "Mean watermark score / Z-score field",
            "Clean WM PPL",
            "Score definition / note",
        ],
        [
            [
                row["scheme"],
                fmt_pct(row["clean_success_rate"]),
                row["clean_count"],
                fmt_num(row["clean_score"]),
                fmt_ppl(row["clean_ppl"]),
                f"{row['score_definition']}; {row['note']}",
            ]
            for row in clean_rows
        ],
    )

    wide_rows: list[list[str]] = []
    for clean in clean_rows:
        scheme = clean["scheme"]
        by_attack = {row["attack_key"]: row for row in attack_rows if row["scheme"] == scheme}
        wide_rows.append(
            [
                scheme,
                fmt_pct(by_attack["word_deletion"]["post_success"]),
                fmt_num(by_attack["word_deletion"]["mean_score"]),
                fmt_ppl(by_attack["word_deletion"]["ppl"]),
                fmt_pct(by_attack["synonym_substitution"]["post_success"]),
                fmt_num(by_attack["synonym_substitution"]["mean_score"]),
                fmt_ppl(by_attack["synonym_substitution"]["ppl"]),
                fmt_pct(by_attack["context_aware_synonym_substitution"]["post_success"]),
                fmt_num(by_attack["context_aware_synonym_substitution"]["mean_score"]),
                fmt_ppl(by_attack["context_aware_synonym_substitution"]["ppl"]),
                fmt_pct(macro_post_success(attack_rows, scheme)),
            ]
        )
    wide_html = html_table(
        [
            "Scheme",
            "WD Success",
            "WD Z/score",
            "WD PPL",
            "WordNet Success",
            "WordNet Z/score",
            "WordNet PPL",
            "Context Success",
            "Context Z/score",
            "Context PPL",
            "Macro Post-attack Success",
        ],
        wide_rows,
    )

    detail_html = html_table(
        [
            "Scheme",
            "Attack",
            "Post-attack Success",
            "Attack Success",
            "Detected/Valid",
            "Mean watermark score / Z-score field",
            "Attacked text PPL",
            "PPL count",
            "Detection missing/error",
        ],
        [
            [
                row["scheme"],
                row["attack"],
                fmt_pct(row["post_success"]),
                fmt_pct(row["attack_success"]),
                fmt_count(row["detected"], row["total"]),
                fmt_num(row["mean_score"]),
                fmt_ppl(row["ppl"]),
                str(row["ppl_count"]),
                str(row["detection_missing_or_error"]),
            ]
            for row in attack_rows
        ],
    )

    source_items = "\n".join(f"<li><code>{html.escape(source)}</code></li>" for source in sources)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Robustness Results Summary</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      color: #1f2933;
      margin: 32px;
      max-width: 1480px;
    }}
    h1, h2 {{ color: #102a43; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 16px 0 28px;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d9e2ec;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      background: #f0f4f8;
      text-align: left;
      position: sticky;
      top: 0;
    }}
    code {{
      background: #f0f4f8;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #829ab1;
      padding: 12px 16px;
      margin: 16px 0 24px;
    }}
  </style>
</head>
<body>
  <h1>Robustness Results Summary</h1>
  <p><strong>Date:</strong> {REPORT_DATE}</p>
  <p>This report consolidates the saved GPT-2 + ELI5 robustness artifacts under the current repository path.</p>

  <h2>Metric Definitions</h2>
  <ul>
    <li><code>Post-attack Success</code>: attacked watermarked text is still detected as watermarked. This is the robustness rate.</li>
    <li><code>Attack Success</code>: <code>1 - Post-attack Success</code>, i.e. the attack successfully removes or breaks detection.</li>
    <li><code>Mean watermark score / Z-score field</code>: method-specific detector score. SynthID uses weighted mean g-value; historical JSON calls it <code>average_z_score</code>, but it is not a standard statistical z-score. UPV uses classifier probability. PDW has only boolean public verification.</li>
    <li><code>Attacked text PPL</code>: perplexity of attacked watermarked completion text.</li>
    <li>Detection thresholds: SynthID weighted mean <code>&gt;= 0.514</code>; UPV score <code>&gt; 0.5</code>; PDW public verification is <code>True</code>.</li>
  </ul>

  <h2>Attack Setup</h2>
  <ul>
    <li><code>WD / word deletion 0.3</code>: random word deletion with deletion ratio <code>0.3</code>.</li>
    <li><code>WordNet synonym 0.5</code>: random synonym substitution with WordNet and substitution ratio <code>0.5</code>.</li>
    <li><code>Context-aware synonym 0.5</code>: context-aware synonym substitution with BERT and substitution ratio <code>0.5</code>. For PDW, this row uses the long-text-safe retest.</li>
  </ul>

  <h2>Clean Detection Context</h2>
  {clean_html}

  <h2>Robustness At A Glance</h2>
  {wide_html}

  <h2>Detailed Robustness Rows</h2>
  {detail_html}

  <h2>Notes</h2>
  <div class="note">
    <p>Original SynthID-Text refers to the saved non-hash <code>Org_WM</code> Type 3 GPT-2 artifact. Type 4/5 <code>Org_WM</code> attack texts exist, but matching saved attack-detection artifacts were not found locally.</p>
    <p>Hash-based SynthID rows use the saved GPT-2 Type 3/4/5 artifacts: Poseidon, Poseidon2, and MiMC. The result JSON files record <code>hash_type</code> but do not store a separate <code>hash_method</code>; this report groups them under the requested two-to-one fixed heading, but the saved JSON alone cannot distinguish fixed vs sort hash-method metadata.</p>
    <p>PDW context-aware results use the retest because the old MarkLLM context-aware attack artifact had invalid long-text BERT masking for PDW watermarked texts.</p>
    <p>Scores are not directly comparable across methods; compare success rates and PPL within the same attack setting.</p>
  </div>

  <h2>Source Artifacts</h2>
  <ul>
    {source_items}
  </ul>
</body>
</html>
"""


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    clean_rows, attack_rows, sources = build_data()
    markdown = build_markdown(clean_rows, attack_rows, sources)
    html_text = build_html(markdown, clean_rows, attack_rows, sources)
    MD_OUT.write_text(markdown, encoding="utf-8")
    HTML_OUT.write_text(html_text, encoding="utf-8")
    print(f"wrote {MD_OUT}")
    print(f"wrote {HTML_OUT}")


if __name__ == "__main__":
    main()
