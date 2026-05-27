#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


TIME_RE = re.compile(r"(?P<label>compile|setup|prove|verication|verify).*?(?P<value>[0-9]+(?:\.[0-9]+)?)", re.I)


def parse_time_file(path: Path):
    rows = []
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        match = TIME_RE.search(line)
        if match:
            rows.append({"file": str(path), "metric": match.group("label").lower(), "seconds": float(match.group("value"))})
    if not rows and text.strip():
        rows.append({"file": str(path), "metric": "raw", "bytes": path.stat().st_size})
    return rows


def parse_json_size(path: Path):
    return {"file": str(path), "bytes": path.stat().st_size}


def parse_halo2_log(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if "Running " in line and " took " in line:
            rows.append({"file": str(path), "line": line.strip()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    for root in args.inputs:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            if path.suffix == ".txt":
                rows.extend(parse_time_file(path))
                rows.extend(parse_halo2_log(path))
            elif path.suffix == ".json":
                rows.append(parse_json_size(path))
    text = json.dumps(rows, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
