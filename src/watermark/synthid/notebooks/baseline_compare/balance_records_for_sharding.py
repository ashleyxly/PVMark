from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorder payload records so modulo sharding has balanced text lengths."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--attack", default=None, help="Use this attack text length as cost if present.")
    return parser.parse_args()


def record_cost(record: dict[str, Any], attack: str | None) -> int:
    if attack and isinstance(record.get("attacks"), dict):
        text = record["attacks"].get(attack, "")
        return len(text or "")
    return len(record.get("completion_text") or record.get("original_text") or "")


def balance_records(records: list[dict[str, Any]], num_shards: int, attack: str | None) -> list[dict[str, Any]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    costs = [0 for _ in range(num_shards)]
    for record in sorted(records, key=lambda r: record_cost(r, attack), reverse=True):
        shard = min(range(num_shards), key=lambda idx: costs[idx])
        buckets[shard].append(record)
        costs[shard] += record_cost(record, attack)

    ordered: list[dict[str, Any]] = []
    max_len = max((len(bucket) for bucket in buckets), default=0)
    for row in range(max_len):
        for shard in range(num_shards):
            if row < len(buckets[shard]):
                ordered.append(buckets[shard][row])
    return ordered


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    payload = read_json(args.input)
    records = payload.get("records", [])
    ordered = balance_records(records, args.num_shards, args.attack)
    metadata = dict(payload.get("metadata", {}))
    metadata["balanced_for_num_shards"] = args.num_shards
    metadata["balanced_cost_attack"] = args.attack
    metadata["balanced_source"] = str(Path(args.input))
    write_json(args.output, {"metadata": metadata, "records": ordered})


if __name__ == "__main__":
    main()
