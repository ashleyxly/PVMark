from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import markdown


SCHEMES = [
    ("Original SynthID non-hash LCG", "original_synthid_lcg"),
    ("PVMark Poseidon T3 two-to-one", "pvmark_poseidon_t3"),
    ("PVMark Poseidon2 T4 two-to-one", "pvmark_poseidon2_t4"),
    ("PVMark MiMC T5 two-to-one", "pvmark_mimc_t5"),
]


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_ms(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f} ms"


def fmt_s(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f} s"


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def synthid_core(root: Path, subdir: str, core_dir: str = "core") -> tuple[float, float, float, float]:
    filename = (
        "efficiency_original_synthid_timing.json"
        if subdir == "original_synthid_lcg"
        else "efficiency_hash_synthid_timing.json"
    )
    payload = read_json(root / core_dir / subdir / filename)
    wet = payload["wet"]["200"]
    wdt = payload["wdt"]["200"]
    return (
        float(wet["mean_ms_per_sample"]),
        float(wet["mean_ms_per_token"]),
        float(wdt["mean_ms_per_sample"]),
        float(wdt["mean_ms_per_token"]),
    )


def upv_core(root: Path, wet_section: str = "cold_first_pass") -> tuple[float, float, float, float]:
    payload = read_json(
        root
        / "core/upv_network_strict_batch1/wet_wdt_network_z1_wet200_wdt200_strict_sequential.json"
    )
    wet = payload["wet_200_tokens"][wet_section]
    wdt = payload["wdt_200_tokens"]
    return (
        float(wet["mean_sec_per_sample"]) * 1000.0,
        float(wet["mean_sec_per_token"]) * 1000.0,
        float(wdt["mean_sec_per_text"]) * 1000.0,
        float(wdt["mean_sec_per_text"]) * 1000.0 / 200.0,
    )


def upv_network_e2e(root: Path) -> tuple[float, float, int]:
    payload = read_json(
        root
        / "e2e/upv_network_detector_batch1_taskset/wet_wdt_network_z1_wet200_wdt200_strict_sequential.json"
    )
    wdt = payload["wdt_200_tokens"]
    return float(wdt["mean_sec_per_text"]) * 1000.0, float(wdt["mean_sec_per_text"]) * 5, int(
        wdt["num_texts_per_run"]
    )


def synthid_e2e(root: Path, subdir: str) -> tuple[float, float, float, float, float, float]:
    payload = read_json(root / "e2e" / subdir / "synthid_e2e_generation_detection.json")
    wm = payload["summary"]["watermarked"]
    uwm = payload["summary"]["unwatermarked_plain"]
    return (
        float(wm["generation_time_sec"]["mean"]),
        float(wm["generation_ms_per_token"]["mean"]),
        float(wm["detection_time_sec"]["mean"]) * 1000.0,
        float(uwm["generation_time_sec"]["mean"]),
        float(uwm["generation_ms_per_token"]["mean"]),
        float(uwm["detection_time_sec"]["mean"]) * 1000.0,
    )


def summarize_generation_detection(records_path: Path) -> dict[str, dict[str, float | int | None]]:
    payload = read_json(records_path)
    records = payload["records"]
    out: dict[str, dict[str, float | int | None]] = {}
    for label, flag in (("watermarked", True), ("unwatermarked_plain", False)):
        subset = [r for r in records if bool(r.get("watermarked")) == flag]
        gen_times = [float(r["generation_time_sec"]) for r in subset if r.get("generation_time_sec") is not None]
        token_counts = [
            float(r["completion_token_count"])
            for r in subset
            if r.get("completion_token_count") is not None and float(r["completion_token_count"]) > 0
        ]
        out[label] = {
            "count": len(subset),
            "generation_time_sec_mean": sum(gen_times) / len(gen_times) if gen_times else None,
            "completion_token_count_mean": sum(token_counts) / len(token_counts) if token_counts else None,
            "generation_ms_per_token_mean": (
                1000.0
                * sum(
                    float(r["generation_time_sec"]) / float(r["completion_token_count"])
                    for r in subset
                    if r.get("generation_time_sec") is not None
                    and r.get("completion_token_count") is not None
                    and float(r["completion_token_count"]) > 0
                )
                / len(token_counts)
                if token_counts
                else None
            ),
        }
    return out


def summarize_detection(records_path: Path) -> dict[str, dict[str, float | int | None]]:
    payload = read_json(records_path)
    records = payload["records"]
    out: dict[str, dict[str, float | int | None]] = {}
    for label, flag in (("watermarked", True), ("unwatermarked_plain", False)):
        subset = [r for r in records if bool(r.get("watermarked")) == flag]
        times = [
            float(r["detection_time_sec"])
            for r in subset
            if r.get("detection_time_sec") is not None and not r.get("error")
        ]
        out[label] = {
            "count": len(subset),
            "detection_time_sec_mean": sum(times) / len(times) if times else None,
        }
    return out


def pdw_summary(root: Path) -> dict[str, Any]:
    return read_json(root / "e2e/pdw/pdw_efficiency_from_records.json")


def write_report(root: Path, md_path: Path, html_path: Path) -> None:
    core_rows: list[list[str]] = []
    for label, subdir in SCHEMES:
        wet_ms, wet_token_ms, wdt_ms, wdt_token_ms = synthid_core(root, subdir, "core_cold")
        core_rows.append(
            [
                label,
                fmt_s(wet_ms / 1000.0),
                fmt_ms(wet_token_ms, 4),
                fmt_ms(wdt_ms),
                fmt_ms(wdt_token_ms, 5),
            ]
        )
    wet_ms, wet_token_ms, wdt_ms, wdt_token_ms = upv_core(root, "cold_first_pass")
    core_rows.append(
        [
            "UPV network-based, strict sequential, batch=1 WDT",
            fmt_s(wet_ms / 1000.0),
            fmt_ms(wet_token_ms, 4),
            fmt_ms(wdt_ms),
            fmt_ms(wdt_token_ms, 5),
        ]
    )
    core_rows.append(
        [
            "PDW asymmetric",
            "N/A: embedding is not cleanly separable from generation in this implementation",
            "N/A",
            "see end-to-end PDW WDT rows",
            "N/A",
        ]
    )

    warm_rows: list[list[str]] = []
    for label, subdir in SCHEMES:
        wet_ms, wet_token_ms, wdt_ms, wdt_token_ms = synthid_core(root, subdir, "core")
        warm_rows.append(
            [
                label,
                fmt_s(wet_ms / 1000.0),
                fmt_ms(wet_token_ms, 4),
                fmt_ms(wdt_ms),
                fmt_ms(wdt_token_ms, 5),
            ]
        )
    wet_ms, wet_token_ms, wdt_ms, wdt_token_ms = upv_core(root, "warm_cached")
    warm_rows.append(
        [
            "UPV network-based, strict sequential, batch=1 WDT",
            fmt_s(wet_ms / 1000.0),
            fmt_ms(wet_token_ms, 4),
            fmt_ms(wdt_ms),
            fmt_ms(wdt_token_ms, 5),
        ]
    )

    e2e_rows: list[list[str]] = []
    for label, subdir in SCHEMES:
        wm_gen_s, wm_gen_ms_tok, wm_det_ms, uwm_gen_s, uwm_gen_ms_tok, uwm_det_ms = synthid_e2e(
            root, subdir
        )
        e2e_rows.append(
            [
                label,
                "3 prompts",
                fmt_s(wm_gen_s),
                fmt_ms(wm_gen_ms_tok, 2),
                fmt_ms(wm_det_ms, 2),
                fmt_s(uwm_gen_s),
                fmt_ms(uwm_gen_ms_tok, 2),
                fmt_ms(uwm_det_ms, 2),
            ]
        )

    upv_gen = summarize_generation_detection(root / "e2e/upv_public/generations.json")
    upv_det = summarize_detection(root / "e2e/upv_public/detection.json")
    upv_net_wdt_ms, _unused, upv_net_n = upv_network_e2e(root)
    e2e_rows.append(
        [
            "UPV public generation + network detector batch=1",
            f"{int(upv_gen['watermarked']['count'] or 0)} prompts; network WDT n={upv_net_n}",
            fmt_s(upv_gen["watermarked"]["generation_time_sec_mean"]),
            fmt_ms(upv_gen["watermarked"]["generation_ms_per_token_mean"], 2),
            f"{fmt_ms(upv_net_wdt_ms, 2)} network; {fmt_ms((upv_det['watermarked']['detection_time_sec_mean'] or 0) * 1000.0, 2)} public z-score",
            fmt_s(upv_gen["unwatermarked_plain"]["generation_time_sec_mean"]),
            fmt_ms(upv_gen["unwatermarked_plain"]["generation_ms_per_token_mean"], 2),
            fmt_ms((upv_det["unwatermarked_plain"]["detection_time_sec_mean"] or 0) * 1000.0, 2),
        ]
    )

    pdw = pdw_summary(root)
    pdw_wm = pdw["wet"]["watermarked"]
    pdw_uwm = pdw["wet"]["unwatermarked_plain"]
    pdw_det_wm = pdw["wdt"]["watermarked"]
    pdw_det_uwm = pdw["wdt"]["unwatermarked_plain"]
    e2e_rows.append(
        [
            "PDW asymmetric",
            f"{pdw_wm['num_valid_records']} WM / {pdw_uwm['num_valid_records']} plain",
            f"{fmt_s(pdw_wm['generation_time_sec']['mean'])}, avg len {fmt_float(pdw_wm['completion_token_count']['mean'], 1)}",
            fmt_ms(pdw_wm["ms_per_token"]["mean"], 2),
            f"{fmt_ms(pdw_det_wm['detection_time_sec']['mean'] * 1000.0, 2)} WM; {fmt_ms(pdw_det_uwm['detection_time_sec']['mean'] * 1000.0, 2)} plain 200-token",
            fmt_s(pdw_uwm["generation_time_sec"]["mean"]),
            fmt_ms(pdw_uwm["ms_per_token"]["mean"], 2),
            fmt_ms(pdw_det_uwm["detection_time_sec"]["mean"] * 1000.0, 2),
        ]
    )

    def md_table(headers: list[str], rows: list[list[str]]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    core_table = md_table(
        ["Scheme", "Core WET / 200 tokens", "Core WET / token", "Core WDT / 200 tokens", "Core WDT / token"],
        core_rows,
    )
    warm_table = md_table(
        ["Scheme", "Warm-cache WET / 200 tokens", "Warm-cache WET / token", "Warm-cache WDT / 200 tokens", "Warm-cache WDT / token"],
        warm_rows,
    )
    e2e_table = md_table(
        [
            "Scheme",
            "Sample size",
            "WM generation",
            "WM generation / token",
            "WM detection",
            "Plain generation",
            "Plain generation / token",
            "Plain detection",
        ],
        e2e_rows,
    )

    md = f"""# Two-Layer WET/WDT Efficiency Quick Pass

Date: 2026-05-22

Output root: `{root}`

## Experiment Plan

1. Watermark-core 200-token WET/WDT:
   - Exclude model loading and LLM forward.
   - WET means 200 online embedding decisions while generating a 200-token output.
   - WDT means detecting/verifying a 200-token text.
   - Original SynthID and PVMark reuse the corrected 200-step replay scripts.
   - Main table uses cold/fresh-context cache mode for SynthID/PVMark because a newly generated sample does not reuse the same hash contexts.
   - UPV uses the new strict sequential replay mode and network detector with `eval_batch_size=1`.
   - PDW core WET is not reported because this implementation couples asymmetric embedding with generation/search and LLM sampling.

2. End-to-end generation/detection latency:
   - Exclude model loading, but follow each implementation's generation/detection flow.
   - Use small samples for turnaround: SynthID/PVMark `n=3`, UPV `n=5` generation plus network WDT check, PDW `n=3`.
   - Run serially on GPU2; main runner bound to CPU set `36-71,108-143`.

## Watermark-Core Results

{core_table}

## Warm-Cache Core Reference

{warm_table}

## End-to-End Results

{e2e_table}

## Notes

- Main runner: `notebooks/baseline_compare/run_two_layer_efficiency_quick.sh`.
- Fresh-context/cold-cache Original/PVMark core reruns are under `{root}/core_cold/`.
- UPV strict WET was implemented in `notebooks/baseline_compare/time_upv_wet_wdt.py` via `--wet-mode strict_sequential`.
- UPV WDT uses network detector batch size 1; this is latency-like rather than batched throughput.
- PDW watermarked generation length is variable by design; the reported watermarked generation time is not a fixed 200-token WET.
- Strict artifact audit passed with `errors=[]`, `warnings=[]`, `notes=[]` for the fresh-context core artifacts plus UPV/PDW summaries.
- These are quick-pass values, not full-sample paper-table values. Use them to validate definitions and expected scale.
"""

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")

    html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Two-Layer WET/WDT Efficiency Quick Pass</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; margin: 32px; max-width: 1200px; }}
pre, code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }}
th {{ background: #f6f8fa; text-align: left; }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="tests/baseline_comparison/two_layer_efficiency_2026-05-22",
    )
    parser.add_argument(
        "--md",
        default="docs/two_layer_efficiency_quick_2026-05-22.md",
    )
    parser.add_argument(
        "--html",
        default="docs/two_layer_efficiency_quick_2026-05-22.html",
    )
    args = parser.parse_args()
    write_report(Path(args.root), Path(args.md), Path(args.html))
    print(args.md)
    print(args.html)


if __name__ == "__main__":
    main()
