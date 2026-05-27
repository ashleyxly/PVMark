from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCHEMES = [
    (
        "Original SynthID non-hash LCG",
        "original_lcg",
        "efficiency_original_synthid_timing.json",
    ),
    (
        "Hash-based SynthID Poseidon2 T4",
        "poseidon2_t4",
        "efficiency_hash_synthid_timing.json",
    ),
    (
        "Hash-based SynthID Poseidon T3",
        "poseidon_t3",
        "efficiency_hash_synthid_timing.json",
    ),
    (
        "Hash-based SynthID MiMC T5",
        "mimc_t5",
        "efficiency_hash_synthid_timing.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated fair-environment WET/WDT benchmarks."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--token-length", type=str, default="200")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p90": None,
            "p95": None,
            "std": None,
        }
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def section_values(section: dict[str, Any], key: str) -> list[float]:
    values = section.get("duration_ms_per_sample_values")
    if isinstance(values, list) and values:
        return [float(v) for v in values]
    if key == "mean":
        value = section.get("mean_ms_per_sample")
    elif key == "median":
        value = section.get("median_ms_per_sample")
    elif key == "min":
        value = section.get("min_ms_per_batch")
    elif key == "max":
        value = section.get("max_ms_per_batch")
    else:
        value = None
    return [float(value)] if value is not None else []


def extract_metric(payload: dict[str, Any], metric: str, token_length: str) -> dict[str, Any]:
    section = payload.get(metric, {}).get(token_length, {})
    raw_values = section_values(section, "raw")
    if not raw_values:
        raw_values = [
            float(section[k])
            for k in ("min_ms_per_batch", "median_ms_per_sample", "mean_ms_per_sample", "max_ms_per_batch")
            if section.get(k) is not None
        ]
    return {
        "raw_values_ms": raw_values,
        "run_summary": {
            "runs": section.get("runs"),
            "mean_ms": section.get("mean_ms_per_sample"),
            "median_ms": section.get("median_ms_per_sample"),
            "min_ms": section.get("min_ms_per_batch"),
            "max_ms": section.get("max_ms_per_batch"),
            "p90_ms": section.get("p90_ms_per_sample"),
        },
    }


def fmt_ms(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f} ms"


def fmt_float(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def collect(root: Path, token_length: str) -> dict[str, Any]:
    repeat_dirs = sorted(p for p in root.glob("repeat_*") if p.is_dir())
    output: dict[str, Any] = {
        "metadata": {
            "root": str(root),
            "token_length": int(token_length),
            "num_repeat_dirs": len(repeat_dirs),
            "definition": (
                "Repeated process-level WET/WDT benchmark. WET excludes LLM forward "
                "and replays precomputed logits through sequential watermarked_call "
                "invocations. WDT excludes model loading/tokenization and includes "
                "EOS mask, context repetition mask, g-value computation, and "
                "weighted-mean score."
            ),
            "note": (
                "Use aggregate median/mean for paper tables. Aggregate min/max are "
                "best/worst observed timed samples and should be labeled as best-case "
                "or worst-case, not as representative latency."
            ),
        },
        "schemes": {},
    }
    for label, subdir, filename in SCHEMES:
        scheme_runs: list[dict[str, Any]] = []
        wet_values: list[float] = []
        wdt_values: list[float] = []
        for repeat_dir in repeat_dirs:
            path = repeat_dir / subdir / filename
            if not path.exists():
                continue
            payload = read_json(path)
            wet = extract_metric(payload, "wet", token_length)
            wdt = extract_metric(payload, "wdt", token_length)
            wet_values.extend(wet["raw_values_ms"])
            wdt_values.extend(wdt["raw_values_ms"])
            scheme_runs.append(
                {
                    "repeat": repeat_dir.name,
                    "artifact": str(path),
                    "wet": wet["run_summary"],
                    "wdt": wdt["run_summary"],
                }
            )
        wet_run_means = [
            float(r["wet"]["mean_ms"]) for r in scheme_runs if r["wet"].get("mean_ms") is not None
        ]
        wdt_run_means = [
            float(r["wdt"]["mean_ms"]) for r in scheme_runs if r["wdt"].get("mean_ms") is not None
        ]
        output["schemes"][label] = {
            "runs": scheme_runs,
            "wet_ms": describe(wet_values),
            "wdt_ms": describe(wdt_values),
            "wet_run_mean_ms": describe(wet_run_means),
            "wdt_run_mean_ms": describe(wdt_run_means),
        }

    original = output["schemes"].get("Original SynthID non-hash LCG", {})
    original_wet_mean = (original.get("wet_ms") or {}).get("mean")
    original_wdt_mean = (original.get("wdt_ms") or {}).get("mean")
    original_wet_median = (original.get("wet_ms") or {}).get("median")
    original_wdt_median = (original.get("wdt_ms") or {}).get("median")
    for scheme in output["schemes"].values():
        wet = scheme["wet_ms"]
        wdt = scheme["wdt_ms"]
        wet["speed_vs_original_mean"] = (
            float(original_wet_mean) / float(wet["mean"])
            if original_wet_mean and wet.get("mean")
            else None
        )
        wet["speed_vs_original_median"] = (
            float(original_wet_median) / float(wet["median"])
            if original_wet_median and wet.get("median")
            else None
        )
        wdt["speed_vs_original_mean"] = (
            float(original_wdt_mean) / float(wdt["mean"])
            if original_wdt_mean and wdt.get("mean")
            else None
        )
        wdt["speed_vs_original_median"] = (
            float(original_wdt_median) / float(wdt["median"])
            if original_wdt_median and wdt.get("median")
            else None
        )
    return output


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scheme",
        "metric",
        "n_samples",
        "mean_ms",
        "median_ms",
        "min_ms",
        "max_ms",
        "p10_ms",
        "p90_ms",
        "p95_ms",
        "std_ms",
        "speed_vs_original_mean",
        "speed_vs_original_median",
        "run_mean_mean_ms",
        "run_mean_std_ms",
        "best_run_mean_ms",
        "worst_run_mean_ms",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for scheme, data in payload["schemes"].items():
            for metric, key, run_key in (
                ("WET", "wet_ms", "wet_run_mean_ms"),
                ("WDT", "wdt_ms", "wdt_run_mean_ms"),
            ):
                stats = data[key]
                run_stats = data[run_key]
                writer.writerow(
                    {
                        "scheme": scheme,
                        "metric": metric,
                        "n_samples": stats["n"],
                        "mean_ms": stats["mean"],
                        "median_ms": stats["median"],
                        "min_ms": stats["min"],
                        "max_ms": stats["max"],
                        "p10_ms": stats["p10"],
                        "p90_ms": stats["p90"],
                        "p95_ms": stats["p95"],
                        "std_ms": stats["std"],
                        "speed_vs_original_mean": stats.get("speed_vs_original_mean"),
                        "speed_vs_original_median": stats.get("speed_vs_original_median"),
                        "run_mean_mean_ms": run_stats["mean"],
                        "run_mean_std_ms": run_stats["std"],
                        "best_run_mean_ms": run_stats["min"],
                        "worst_run_mean_ms": run_stats["max"],
                    }
                )


def table_rows(payload: dict[str, Any], metric_key: str) -> str:
    rows = []
    for scheme, data in payload["schemes"].items():
        stats = data[metric_key]
        run_stats = data["wet_run_mean_ms" if metric_key == "wet_ms" else "wdt_run_mean_ms"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(scheme)}</td>"
            f"<td class=\"num\">{int(stats['n'] or 0)}</td>"
            f"<td class=\"num\">{fmt_ms(stats['mean'])}</td>"
            f"<td class=\"num\">{fmt_ms(stats['median'])}</td>"
            f"<td class=\"num\">{fmt_ms(stats['min'])}</td>"
            f"<td class=\"num\">{fmt_ms(stats['max'])}</td>"
            f"<td class=\"num\">{fmt_ms(stats['p90'])}</td>"
            f"<td class=\"num\">{fmt_ms(stats['std'])}</td>"
            f"<td class=\"num\">{fmt_float(stats.get('speed_vs_original_mean'), 2)}x</td>"
            f"<td class=\"num\">{fmt_ms(run_stats['min'])}</td>"
            f"<td class=\"num\">{fmt_ms(run_stats['max'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = html.escape(payload["metadata"]["root"])
    style = """
body{font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f7f8fb;color:#17202a}
main{max-width:1180px;margin:0 auto;padding:32px 20px 56px}
h1{font-size:30px;margin:0 0 8px}h2{font-size:21px;margin:30px 0 12px}
.note{background:#fff8e8;border:1px solid #ead29b;border-radius:8px;padding:12px 14px;margin:14px 0;color:#4e3600}
.panel{background:white;border:1px solid #d9dee7;border-radius:8px;padding:16px;overflow-x:auto;margin:14px 0}
table{width:100%;border-collapse:collapse;min-width:980px;font-size:14px}th,td{border:1px solid #d9dee7;padding:8px 9px;vertical-align:top}
th{background:#edf2f7;text-align:left}.num{text-align:right;white-space:nowrap}code{background:#eef2f6;border:1px solid #dce3ec;border-radius:4px;padding:1px 4px}.muted{color:#596579}
"""
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hash-based SynthID repeated WET/WDT benchmark</title>
  <style>{style}</style>
</head>
<body>
<main>
  <h1>Hash-based SynthID repeated WET/WDT benchmark</h1>
  <p class="muted">Root: <code>{root}</code></p>
  <div class="note">
    这份报告用于抵消 CPU/GPU 资源波动。主表建议报告 mean/median/p90/std；min/max 是观测到的最快/最慢 timed sample，只应标注为 best/worst observed，不建议把 fastest 当作代表性结果。
  </div>

  <h2>WET, 200 Tokens</h2>
  <div class="panel">
    <table>
      <thead><tr><th>Scheme</th><th class="num">timed samples</th><th class="num">mean</th><th class="num">median</th><th class="num">fastest</th><th class="num">slowest</th><th class="num">p90</th><th class="num">std</th><th class="num">mean speed vs original</th><th class="num">best run mean</th><th class="num">worst run mean</th></tr></thead>
      <tbody>{table_rows(payload, "wet_ms")}</tbody>
    </table>
  </div>

  <h2>WDT, 200 Tokens</h2>
  <div class="panel">
    <table>
      <thead><tr><th>Scheme</th><th class="num">timed samples</th><th class="num">mean</th><th class="num">median</th><th class="num">fastest</th><th class="num">slowest</th><th class="num">p90</th><th class="num">std</th><th class="num">mean speed vs original</th><th class="num">best run mean</th><th class="num">worst run mean</th></tr></thead>
      <tbody>{table_rows(payload, "wdt_ms")}</tbody>
    </table>
  </div>

  <h2>Method</h2>
  <div class="panel">
    <p>{html.escape(payload["metadata"]["definition"])}</p>
    <p>{html.escape(payload["metadata"]["note"])}</p>
  </div>
</main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = collect(args.root, args.token_length)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv, payload)
    write_html(args.html, payload)
    print(f"wrote {args.json}")
    print(f"wrote {args.csv}")
    print(f"wrote {args.html}")


if __name__ == "__main__":
    main()
