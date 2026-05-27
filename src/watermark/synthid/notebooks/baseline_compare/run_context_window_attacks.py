from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nltk.corpus import wordnet
from tqdm import tqdm
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


ATTACK_NAME = "context_aware_synonym_substitution"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long-text-safe context-aware synonym substitution attack. "
            "This follows MarkLLM's ContextAwareSynonymSubstitution logic, but "
            "runs BERT on a local window around the masked word so [MASK] is not "
            "lost when long texts exceed BERT's max length."
        )
    )
    parser.add_argument("--input", required=True, help="Generation JSON with records.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bert-path", default=str(DEFAULT_BERT_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-aware-ratio", type=float, default=0.5)
    parser.add_argument("--text-key", default="completion_text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-radius-words", type=int, default=96)
    parser.add_argument("--max-bert-length", type=int, default=512)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_synonyms(word: str) -> list[str]:
    synonyms: set[str] = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)


class ContextWindowSubstitution:
    def __init__(
        self,
        *,
        ratio: float,
        tokenizer: BertTokenizer,
        model: BertForMaskedLM,
        device: str,
        batch_size: int,
        window_radius_words: int,
        max_bert_length: int,
    ) -> None:
        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.window_radius_words = window_radius_words
        self.max_bert_length = max_bert_length
        self._synonym_cache: dict[str, bool] = {}

    def has_wordnet_synonym(self, word: str) -> bool:
        if word not in self._synonym_cache:
            self._synonym_cache[word] = bool(get_synonyms(word))
        return self._synonym_cache[word]

    def choose_indices(self, words: list[str]) -> list[int]:
        num_words = len(words)
        if num_words == 0:
            return []
        replaceable_indices = [
            i for i, word in enumerate(words) if self.has_wordnet_synonym(word)
        ]
        num_to_replace = int(
            min(self.ratio, len(replaceable_indices) / num_words) * num_words
        )
        if num_to_replace <= 0:
            return []
        return random.sample(replaceable_indices, num_to_replace)

    def make_masked_window(self, words: list[str], target_index: int) -> str:
        radius = self.window_radius_words
        while radius >= 0:
            left = max(0, target_index - radius)
            right = min(len(words), target_index + radius + 1)
            window = list(words[left:target_index]) + ["[MASK]"] + list(words[target_index + 1 : right])
            text = " ".join(window)
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_bert_length,
            )
            mask_positions = torch.where(
                encoded["input_ids"][0] == self.tokenizer.mask_token_id
            )[0]
            if mask_positions.numel() > 0:
                return text
            radius = radius // 2 - 1
        return "[MASK]"

    def predict_replacements(self, masked_texts: list[str]) -> list[str]:
        replacements: list[str] = []
        for start in range(0, len(masked_texts), self.batch_size):
            batch_texts = masked_texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_bert_length,
            ).to(self.device)
            mask_positions = inputs["input_ids"].eq(self.tokenizer.mask_token_id)
            missing_rows = [
                idx for idx, row in enumerate(mask_positions) if not bool(row.any())
            ]
            if missing_rows:
                raise RuntimeError(
                    f"[MASK] missing after local-window tokenization for batch rows {missing_rows}"
                )
            with torch.no_grad():
                outputs = self.model(**inputs)
            for batch_idx in range(len(batch_texts)):
                pos = torch.where(mask_positions[batch_idx])[0][0].item()
                logits = outputs.logits[batch_idx, pos]
                token_id = int(torch.argmax(logits).item())
                replacements.append(self.tokenizer.convert_ids_to_tokens([token_id])[0])
        return replacements

    def edit(self, text: str) -> tuple[str, dict[str, Any]]:
        words = text.split()
        selected_indices = self.choose_indices(words)
        stats = {
            "num_words": len(words),
            "selected_count": len(selected_indices),
            "requested_ratio": self.ratio,
            "actual_selected_ratio": (len(selected_indices) / len(words)) if words else None,
            "window_radius_words": self.window_radius_words,
            "max_bert_length": self.max_bert_length,
        }
        if not words:
            stats["real_replace_count"] = 0
            return "", stats
        if not selected_indices:
            stats["real_replace_count"] = 0
            return text, stats

        masked_texts = [self.make_masked_window(words, i) for i in selected_indices]
        replacements = self.predict_replacements(masked_texts)
        for index, replacement in zip(selected_indices, replacements):
            words[index] = replacement
        stats["real_replace_count"] = len(replacements)
        stats["actual_replace_ratio"] = len(replacements) / len(words)
        return " ".join(words), stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = BertForMaskedLM.from_pretrained(args.bert_path).to(device)
    model.eval()
    attacker = ContextWindowSubstitution(
        ratio=args.context_aware_ratio,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        window_radius_words=args.window_radius_words,
        max_bert_length=args.max_bert_length,
    )

    with open(args.input, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = shard_items(
        payload["records"],
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.limit is not None:
        records = records[: args.limit]

    output_path = Path(args.output)
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()

    for record in tqdm(records):
        if record_key(record) in existing_keys:
            continue
        text = record.get(args.text_key, "") or ""
        attack_outputs: dict[str, str] = {}
        timing: dict[str, float] = {}
        errors: dict[str, str] = {}
        attack_stats: dict[str, Any] = {}
        with Timer() as t:
            try:
                attacked_text, stats = attacker.edit(text)
                attack_outputs[ATTACK_NAME] = attacked_text
                attack_stats[ATTACK_NAME] = stats
            except Exception as exc:
                attack_outputs[ATTACK_NAME] = ""
                attack_stats[ATTACK_NAME] = {
                    "num_words": len(text.split()),
                    "requested_ratio": args.context_aware_ratio,
                }
                errors[ATTACK_NAME] = repr(exc)
        timing[ATTACK_NAME] = t.elapsed
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
            "attack_stats": attack_stats,
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
                "attack": ATTACK_NAME,
                "context_aware_ratio": args.context_aware_ratio,
                "text_key": args.text_key,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "checkpoint_jsonl": str(records_jsonl),
                "window_radius_words": args.window_radius_words,
                "max_bert_length": args.max_bert_length,
                "batch_size": args.batch_size,
                "note": (
                    "Long-text-safe context-aware attack using local BERT windows; "
                    "keeps MarkLLM ratio/index-selection semantics while preventing "
                    "the [MASK] token from being removed by BERT truncation."
                ),
            },
            "records": attacked_records,
        },
    )


if __name__ == "__main__":
    main()
