from __future__ import annotations
import os

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from common import (
    DEFAULT_BERT_PATH,
    Timer,
    append_jsonl,
    completed_keys,
    load_existing_payload_records,
    record_key,
    shard_items,
    sort_records,
    write_json,
)


MARKLLM_ROOT = Path(os.environ.get("PVMark_MARKLLM_ROOT", "external/MarkLLM"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MarkLLM attacks for baseline outputs.")
    parser.add_argument("--input", required=True, help="Generation JSON with records.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bert-path", default=str(DEFAULT_BERT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--word-deletion-ratio", type=float, default=0.3)
    parser.add_argument("--synonym-ratio", type=float, default=0.5)
    parser.add_argument("--context-aware-ratio", type=float, default=0.5)
    parser.add_argument("--text-key", default="completion_text")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_markllm() -> Any:
    sys.path.insert(0, str(MARKLLM_ROOT))
    from evaluation.tools.text_editor import (  # type: ignore
        ContextAwareSynonymSubstitution,
        SynonymSubstitution,
        WordDeletion,
    )

    return WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    WordDeletion, SynonymSubstitution, ContextAwareSynonymSubstitution = import_markllm()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForMaskedLM.from_pretrained(args.bert_path).to(device)
    model.eval()

    attacks = {
        "word_deletion": WordDeletion(ratio=args.word_deletion_ratio),
        "synonym_substitution": SynonymSubstitution(ratio=args.synonym_ratio),
        "context_aware_synonym_substitution": ContextAwareSynonymSubstitution(
            ratio=args.context_aware_ratio,
            tokenizer=tokenizer,
            model=model,
            device=device,
        ),
    }

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = shard_items(
        payload["records"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        records = records[: args.limit]

    records_jsonl = Path(args.output).with_suffix(Path(args.output).suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()
    for record in records:
        if record_key(record) in existing_keys:
            continue
        text = record.get(args.text_key, "")
        attack_outputs: dict[str, Any] = {}
        timing: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name, attack in attacks.items():
            with Timer() as t:
                try:
                    attack_outputs[name] = attack.edit(text)
                except Exception as exc:
                    attack_outputs[name] = ""
                    errors[name] = repr(exc)
            timing[name] = t.elapsed
        attacked_record = {
            "sample_id": record.get("sample_id"),
            "source_index": record.get("source_index"),
            "q_id": record.get("q_id"),
            "method": record.get("method"),
            "watermarked": bool(record.get("watermarked")),
            "method_metadata": record.get("method_metadata", {}),
            "original_text": text,
            "attacks": attack_outputs,
            "attack_time_sec": timing,
            "errors": errors,
        }
        append_jsonl(records_jsonl, attacked_record)
        existing_keys.add(record_key(record))

    attacked_records = sort_records(
        load_existing_payload_records(args.output, records_jsonl, lambda r: record_key(r))
    )
    write_json(
        args.output,
        {
            "metadata": {
                "source": args.input,
                "seed": args.seed,
                "bert_path": args.bert_path,
                "word_deletion_ratio": args.word_deletion_ratio,
                "synonym_ratio": args.synonym_ratio,
                "context_aware_ratio": args.context_aware_ratio,
                "text_key": args.text_key,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "checkpoint_jsonl": str(records_jsonl),
            },
            "records": attacked_records,
        },
    )


if __name__ == "__main__":
    main()
