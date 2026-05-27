from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from common import (
    DEFAULT_GPT2_PATH,
    DEFAULT_UPV_GENERATOR,
    DEFAULT_UPV_ROOT,
    Timer,
    append_jsonl,
    completed_keys,
    load_existing_payload_records,
    record_key,
    shard_items,
    sort_records,
    write_json,
)
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect UPV watermark on attacked outputs.")
    parser.add_argument("--input", required=True, help="Attacks JSON from run_attacks.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--upv-root", default=str(DEFAULT_UPV_ROOT))
    parser.add_argument("--generator-model", default=str(DEFAULT_UPV_GENERATOR))
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=4.0)
    return parser.parse_args()


def import_upv(root: str | Path) -> Any:
    sys.path.insert(0, str(root))
    from watermark_model import Watermark  # type: ignore

    return Watermark


@contextlib.contextmanager
def torch_load_map_location_for_current_device() -> Any:
    original_load = torch.load

    def load_with_map_location(*args: Any, **kwargs: Any) -> Any:
        if not torch.cuda.is_available() and "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("cpu")
        return original_load(*args, **kwargs)

    torch.load = load_with_map_location  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_load  # type: ignore[assignment]


def main() -> None:
    args = parse_args()
    Watermark = import_upv(args.upv_root)
    with torch_load_map_location_for_current_device():
        watermark = Watermark(
            bit_number=args.bit_number,
            window_size=args.window_size,
            layers=args.layers,
            delta=args.delta,
            model_dir=args.generator_model,
            beam_size=args.beam_size,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)

    input_records = shard_items(
        payload["records"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        input_records = input_records[: args.limit]
    output_path = Path(args.output)
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r, r.get("attack"))) if args.resume else set()

    for record in input_records:
        for attack_name, text in record.get("attacks", {}).items():
            key = record_key(record, attack_name)
            if key in existing_keys:
                continue
            with Timer() as t:
                try:
                    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)[
                        "input_ids"
                    ].squeeze(0)
                    _mask, green_count, z_score = watermark.green_token_mask_and_stats(input_ids)
                    detected = bool(z_score > args.threshold)
                    error = None
                except Exception as exc:
                    green_count = None
                    z_score = None
                    detected = None
                    error = repr(exc)
            detection_record = {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": record.get("method"),
                "attack": attack_name,
                "watermarked": bool(record.get("watermarked")),
                "detected": detected,
                "score": float(z_score) if z_score is not None else None,
                "green_token_count": int(green_count) if green_count is not None else None,
                "threshold": args.threshold,
                "detection_time_sec": t.elapsed,
                "error": error,
            }
            append_jsonl(records_jsonl, detection_record)
            existing_keys.add(key)

    records = sort_records(
        load_existing_payload_records(args.output, records_jsonl, lambda r: record_key(r, r.get("attack")))
    )
    write_json(
        args.output,
        {
            "metadata": {
                "source": args.input,
                "method": "upv_public",
                "detector": "public_z_score",
                "threshold": args.threshold,
                "generator_model": args.generator_model,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "checkpoint_jsonl": str(records_jsonl),
            },
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
