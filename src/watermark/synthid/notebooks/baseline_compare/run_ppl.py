from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DEFAULT_OPT27B_PATH,
    append_jsonl,
    completed_keys,
    load_existing_payload_records,
    record_key,
    shard_items,
    sort_records,
    summarize_numbers,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PPL with a shared evaluator.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_OPT27B_PATH))
    parser.add_argument("--text-key", default="completion_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--iqr-filter", action="store_true")
    return parser.parse_args()


def load_text_records(path: str | Path, text_key: str, limit: int | None) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records: list[dict[str, Any]] = []
    for record in payload["records"]:
        text = record.get(text_key)
        if text is None and "attacks" in record:
            for attack_name, attack_text in record["attacks"].items():
                records.append(
                    {
                        "sample_id": record.get("sample_id"),
                        "q_id": record.get("q_id"),
                        "method": record.get("method"),
                        "watermarked": bool(record.get("watermarked")),
                        "attack": attack_name,
                        "text": attack_text,
                    }
                )
                if limit is not None and len(records) >= limit:
                    break
            if limit is not None and len(records) >= limit:
                break
            continue
        records.append(
            {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": record.get("method"),
                "watermarked": bool(record.get("watermarked")),
                "attack": record.get("attack"),
                "text": text or "",
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def ppl_record_key(record: dict[str, Any]) -> str:
    return record_key(record, record.get("attack"))


def compute_ppl(text: str, model: Any, tokenizer: Any, device: torch.device, max_length: int, stride: int) -> float:
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = encodings.input_ids
    if input_ids.shape[-1] <= 1:
        return float("inf")
    target_ids = input_ids.clone()
    target_ids[:, 0] = -100
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
    return float(torch.exp(outputs.loss.float()).item())


def iqr_filter(values: list[float]) -> tuple[list[float], int]:
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return [], 0
    arr = np.asarray(finite)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    filtered = [v for v in finite if lo <= v <= hi]
    return filtered, len(finite) - len(filtered)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": device},
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    records = shard_items(
        load_text_records(args.input, args.text_key, args.limit),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    records_jsonl = Path(args.output).with_suffix(Path(args.output).suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, ppl_record_key) if args.resume else set()
    for record in tqdm(records):
        if ppl_record_key(record) in existing_keys:
            continue
        try:
            ppl = compute_ppl(record["text"], model, tokenizer, device, args.max_length, args.stride)
            error = None
        except Exception as exc:
            ppl = None
            error = repr(exc)
        result = {**{k: v for k, v in record.items() if k != "text"}, "ppl": ppl, "error": error}
        append_jsonl(records_jsonl, result)
        existing_keys.add(ppl_record_key(result))

    results = sort_records(
        load_existing_payload_records(args.output, records_jsonl, ppl_record_key)
    )
    ppls = [r["ppl"] for r in results if r["ppl"] is not None]
    summary = {"raw": summarize_numbers(ppls)}
    if args.iqr_filter:
        filtered, removed = iqr_filter([float(v) for v in ppls])
        summary["iqr_filtered"] = {**summarize_numbers(filtered), "outliers_removed": removed}

    write_json(
        args.output,
        {
            "metadata": {
                "source": args.input,
                "evaluator": args.model_name_or_path,
                "text_key": args.text_key,
                "max_length": args.max_length,
                "stride": args.stride,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "checkpoint_jsonl": str(records_jsonl),
            },
            "summary": summary,
            "records": results,
        },
    )


if __name__ == "__main__":
    main()
