from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from baseline_eval.common import (
    DEFAULT_BERT_MODEL,
    WallTimer,
    append_resume_shard_args,
    ensure_dir,
    make_failure,
    read_json,
    read_json_dir,
    remove_checkpoint_if_exists,
    select_shard,
    set_seed,
    write_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight local versions of the three text-edit attacks.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--word-deletion-ratio", type=float, default=0.3)
    parser.add_argument("--synonym-substitution-ratio", type=float, default=0.5)
    parser.add_argument("--context-aware-synonym-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--text-field", default="output_with_watermark")
    parser.add_argument("--attack-only-detected", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detected-field", default="prediction_with_watermark")
    append_resume_shard_args(parser)
    return parser.parse_args()


def word_deletion(text: str, ratio: float) -> str:
    if not text:
        return text
    return " ".join(word for word in text.split() if random.random() >= ratio)


class SynonymSubstitution:
    def __init__(self, ratio: float) -> None:
        from nltk.corpus import wordnet

        self.ratio = ratio
        self.wordnet = wordnet

    def edit(self, text: str) -> str:
        words = text.split()
        word_synonyms: dict[str, list[Any]] = {}
        replaceable_indices: list[int] = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                word_synonyms[word] = [syn for syn in self.wordnet.synsets(word) if len(syn.lemmas()) > 1]
            if word_synonyms[word]:
                replaceable_indices.append(i)
        num_to_replace = min(int(self.ratio * len(words)), len(replaceable_indices))
        if num_to_replace > 0:
            for i in random.sample(replaceable_indices, num_to_replace):
                chosen_syn = random.choice(word_synonyms[words[i]])
                words[i] = random.choice(chosen_syn.lemmas()[1:]).name().replace("_", " ")
        return " ".join(words)


class ContextAwareSynonymSubstitution:
    def __init__(self, ratio: float, tokenizer: Any, model: Any, device: str) -> None:
        from nltk.corpus import wordnet

        self.ratio = ratio
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = device
        self.wordnet = wordnet

    def _get_synonyms_from_wordnet(self, word: str) -> list[str]:
        synonyms: set[str] = set()
        for syn in self.wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace("_", " "))
        return list(synonyms)

    def edit(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        replaceable_indices = [i for i, word in enumerate(words) if self._get_synonyms_from_wordnet(word)]
        num_to_replace = int(min(self.ratio, len(replaceable_indices) / len(words)) * len(words))
        if num_to_replace <= 0:
            return text
        for i in random.sample(replaceable_indices, num_to_replace):
            masked_text = " ".join(words[:i] + [self.tokenizer.mask_token] + words[i + 1 :])
            inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            mask_position = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            if mask_position.numel() <= 0:
                continue
            with torch.inference_mode():
                outputs = self.model(**inputs)
            predictions = outputs.logits[0, int(mask_position[0].item())]
            predicted_index = int(torch.argmax(predictions).detach().cpu().item())
            words[i] = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        return " ".join(words)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    from transformers import BertForMaskedLM, BertTokenizer

    payload = read_json(args.input)
    out_path = args.output or str(Path(args.input).with_name("attacks.json"))
    out_dir = ensure_dir(Path(out_path).parent)
    sample_dir = ensure_dir(out_dir / "attack_samples")
    failure_dir = ensure_dir(out_dir / "attack_failures")

    synonym = SynonymSubstitution(args.synonym_substitution_ratio)
    context = ContextAwareSynonymSubstitution(
        args.context_aware_synonym_ratio,
        tokenizer=BertTokenizer.from_pretrained(args.bert_model),
        model=BertForMaskedLM.from_pretrained(args.bert_model).to(device),
        device=device,
    )

    source_samples = [s for s in payload["samples"] if s.get(args.text_field)]
    if args.attack_only_detected:
        source_samples = [s for s in source_samples if s.get(args.detected_field)]
    if args.max_samples is not None:
        source_samples = source_samples[: args.max_samples]
    source_samples = select_shard(source_samples, args.shard_index, args.num_shards)

    rows_by_id = read_json_dir(sample_dir, "sample_id") if args.resume else {}
    failures_by_id = read_json_dir(failure_dir, "id") if args.resume else {}
    for sample in tqdm(source_samples, desc="Light attacks"):
        sample_id = int(sample["id"])
        if args.resume and sample_id in rows_by_id:
            continue
        if args.resume and sample_id in failures_by_id and not args.retry_failures:
            continue
        try:
            text = sample[args.text_field]
            with WallTimer() as t1:
                deleted = word_deletion(text, args.word_deletion_ratio)
            with WallTimer() as t2:
                synonym_text = synonym.edit(text)
            with WallTimer() as t3:
                context_text = context.edit(text)
            row = {
                "sample_id": sample_id,
                "original_text": text,
                "word_deletion": deleted,
                "synonym_substitution": synonym_text,
                "context_aware_synonym_substitution": context_text,
                "attack_time_sec": {
                    "word_deletion": t1.elapsed,
                    "synonym_substitution": t2.elapsed,
                    "context_aware_synonym_substitution": t3.elapsed,
                },
            }
            write_checkpoint(sample_dir, sample_id, row)
            remove_checkpoint_if_exists(failure_dir, sample_id)
            rows_by_id[sample_id] = row
            failures_by_id.pop(sample_id, None)
        except Exception as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            failure = make_failure(sample_id, "light_attack_generation", exc)
            write_checkpoint(failure_dir, sample_id, failure)
            failures_by_id[sample_id] = failure
            if args.fail_fast:
                raise

    rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    failures = [failures_by_id[k] for k in sorted(failures_by_id)]
    write_json(
        out_path,
        {
            "metadata": {
                "source_generations": args.input,
                "scheme": payload.get("metadata", {}).get("scheme"),
                "seed": args.seed,
                "text_field": args.text_field,
                "attack_only_detected": args.attack_only_detected,
                "detected_field": args.detected_field if args.attack_only_detected else None,
                "attack_params": {
                    "word_deletion_ratio": args.word_deletion_ratio,
                    "synonym_substitution_ratio": args.synonym_substitution_ratio,
                    "context_aware_synonym_ratio": args.context_aware_synonym_ratio,
                    "bert_model": args.bert_model,
                },
                "num_attacked": len(rows),
                "failures": failures,
                "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            },
            "attacks": rows,
        },
    )


if __name__ == "__main__":
    main()
