from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_PUBLICLY_DETECTABLE_ROOT,
    Timer,
    append_jsonl,
    completed_keys,
    load_existing_payload_records,
    record_key,
    shard_items,
    sort_records,
    write_json,
)
from pdw_experiment import import_pdw, load_detection_material


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect PDW asymmetric watermark on attacked outputs.")
    parser.add_argument("--input", default=None, help="Attacks JSON from run_attacks.py")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdw-root", default=str(DEFAULT_PUBLICLY_DETECTABLE_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--message-length", type=int, default=None)
    parser.add_argument("--signature-segment-length", type=int, default=None)
    parser.add_argument("--bit-size", type=int, default=None)
    parser.add_argument("--max-planted-errors", type=int, default=None)
    parser.add_argument(
        "--isolate-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each PDW attacked-text detection in a child process so native segfaults become per-record errors.",
    )
    parser.add_argument("--single-timeout-sec", type=int, default=300)
    parser.add_argument("--single-detection", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def get_key_ref(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("method_metadata") or {}
    key_ref = metadata.get("pdw_key")
    return key_ref if isinstance(key_ref, dict) else None


def resolve_params(args: argparse.Namespace, crypto: Any) -> dict[str, int]:
    return {
        "message_length": args.message_length or crypto.DEFAULT_MESSAGE_LENGTH,
        "signature_segment_length": args.signature_segment_length or crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH,
        "bit_size": args.bit_size or crypto.DEFAULT_BIT_SIZE,
        "max_planted_errors": (
            args.max_planted_errors
            if args.max_planted_errors is not None
            else crypto.DEFAULT_MAX_PLANTED_ERRORS
        ),
    }


def run_single_detection(args: argparse.Namespace) -> None:
    payload = json.load(sys.stdin)
    crypto, pdw_detect, _pdw_generate = import_pdw(args.pdw_root)
    params_values = resolve_params(args, crypto)
    key_ref = payload.get("key_ref")
    if not isinstance(key_ref, dict):
        print(json.dumps({"detected": None, "error": "missing method_metadata.pdw_key"}))
        return
    pk, params = load_detection_material(args.pdw_root, key_ref)
    detected = bool(
        pdw_detect.search_for_asymmetric_watermark(
            pk,
            params,
            payload.get("text", ""),
            params_values["message_length"],
            params_values["signature_segment_length"],
            params_values["bit_size"],
            params_values["max_planted_errors"],
        )
    )
    print(json.dumps({"detected": detected, "error": None}))


def detect_in_subprocess(
    args: argparse.Namespace,
    key_ref: dict[str, Any] | None,
    text: str,
) -> tuple[bool | None, str | None]:
    if key_ref is None:
        return None, "missing method_metadata.pdw_key"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-detection",
        "--pdw-root",
        args.pdw_root,
        "--message-length",
        str(args.message_length) if args.message_length is not None else "0",
        "--signature-segment-length",
        str(args.signature_segment_length) if args.signature_segment_length is not None else "0",
        "--bit-size",
        str(args.bit_size) if args.bit_size is not None else "0",
        "--max-planted-errors",
        str(args.max_planted_errors) if args.max_planted_errors is not None else "-1",
    ]
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-detection",
        "--pdw-root",
        args.pdw_root,
    ]
    if args.message_length is not None:
        command.extend(["--message-length", str(args.message_length)])
    if args.signature_segment_length is not None:
        command.extend(["--signature-segment-length", str(args.signature_segment_length)])
    if args.bit_size is not None:
        command.extend(["--bit-size", str(args.bit_size)])
    if args.max_planted_errors is not None:
        command.extend(["--max-planted-errors", str(args.max_planted_errors)])
    try:
        result = subprocess.run(
            command,
            input=json.dumps({"key_ref": key_ref, "text": text}, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.single_timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return None, f"single detection subprocess timed out after {args.single_timeout_sec}s"
    if result.returncode != 0:
        stderr_tail = "\n".join(result.stderr.splitlines()[-8:])
        return None, f"single detection subprocess failed returncode={result.returncode}: {stderr_tail}"
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as exc:
        stdout_tail = "\n".join(result.stdout.splitlines()[-8:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-8:])
        return None, f"single detection subprocess returned invalid JSON: {repr(exc)} stdout={stdout_tail} stderr={stderr_tail}"
    return payload.get("detected"), payload.get("error")


def main() -> None:
    args = parse_args()
    if args.single_detection:
        run_single_detection(args)
        return
    if not args.input or not args.output:
        raise SystemExit("--input and --output are required unless --single-detection is set")

    crypto, pdw_detect, _pdw_generate = import_pdw(args.pdw_root)
    params_values = resolve_params(args, crypto)
    message_length = params_values["message_length"]
    signature_segment_length = params_values["signature_segment_length"]
    bit_size = params_values["bit_size"]
    max_planted_errors = params_values["max_planted_errors"]

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

    key_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    for record in input_records:
        key_ref = get_key_ref(record)
        for attack_name, text in record.get("attacks", {}).items():
            key = record_key(record, attack_name)
            if key in existing_keys:
                continue
            with Timer() as t:
                try:
                    if args.isolate_records:
                        detected, error = detect_in_subprocess(args, key_ref, text)
                    else:
                        if key_ref is None:
                            raise ValueError("missing method_metadata.pdw_key")
                        cache_key = (str(key_ref.get("pk_path")), str(key_ref.get("params_path")))
                        if cache_key not in key_cache:
                            key_cache[cache_key] = load_detection_material(args.pdw_root, key_ref)
                        pk, params = key_cache[cache_key]
                        detected = bool(
                            pdw_detect.search_for_asymmetric_watermark(
                                pk,
                                params,
                                text,
                                message_length,
                                signature_segment_length,
                                bit_size,
                                max_planted_errors,
                            )
                        )
                        error = None
                except Exception as exc:
                    detected = None
                    error = repr(exc)
            detection_record = {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": "publicly_detectable_asymmetric",
                "attack": attack_name,
                "watermarked": bool(record.get("watermarked")),
                "detected": detected,
                "score": None,
                "key_id": key_ref.get("key_id") if key_ref else None,
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
                "method": "publicly_detectable_asymmetric",
                "detector": "search_for_asymmetric_watermark",
                "message_length": message_length,
                "signature_segment_length": signature_segment_length,
                "bit_size": bit_size,
                "max_planted_errors": max_planted_errors,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "checkpoint_jsonl": str(records_jsonl),
            },
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
