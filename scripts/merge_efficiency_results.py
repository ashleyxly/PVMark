from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from baseline_eval.benchmark_efficiency import flatten_rows, write_csv, write_html, write_markdown
from baseline_eval.common import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge efficiency benchmark result directories.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input directories containing efficiency_results.json.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def merge_metadata(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = value
        elif key in {"hash_types", "hash_methods", "schemes"}:
            merged[key] = sorted(set(merged.get(key, [])) | set(value or []))
        elif key == "hash_wet_backend":
            existing = merged.get(key)
            if isinstance(existing, list):
                merged[key] = sorted(set(existing) | {value})
            elif existing == value:
                merged[key] = existing
            else:
                merged[key] = sorted({existing, value})
    return merged


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    merged: dict[str, Any] = {"metadata": {}, "schemes": {}}

    for input_dir in args.inputs:
        result_path = Path(input_dir) / "efficiency_results.json"
        payload = read_json(result_path)
        merged["metadata"] = merge_metadata(merged["metadata"], payload.get("metadata", {}))
        for scheme_name, scheme_result in payload.get("schemes", {}).items():
            if scheme_name in merged["schemes"]:
                raise SystemExit(f"Duplicate scheme in inputs: {scheme_name}")
            merged["schemes"][scheme_name] = scheme_result

    rows = flatten_rows(merged)
    json_path = out_dir / "efficiency_results.json"
    csv_path = out_dir / "efficiency_results.csv"
    md_path = out_dir / "efficiency_results.md"
    html_path = out_dir / "efficiency_results.html"
    write_json(json_path, merged)
    write_csv(csv_path, rows)
    write_markdown(md_path, merged, rows)
    write_html(
        html_path,
        md_path,
        rows,
        merged["metadata"]["wdt_token_counts"],
        merged["metadata"].get("wet_token_count", 200),
    )
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Wrote {html_path}", flush=True)


if __name__ == "__main__":
    main()
