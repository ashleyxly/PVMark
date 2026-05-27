from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from baseline_eval.common import (
    DEFAULT_BERT_MODEL,
    DEFAULT_MARKLLM_ROOT,
    WallTimer,
    add_repo_to_path,
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
    parser = argparse.ArgumentParser(description="Run MarkLLM text-edit attacks on detected watermarked outputs.")
    parser.add_argument("--input", required=True, help="generations.json")
    parser.add_argument("--output", default=None, help="attacks.json; defaults next to input")
    parser.add_argument("--markllm-root", default=DEFAULT_MARKLLM_ROOT)
    parser.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--word-deletion-ratio", type=float, default=0.3)
    parser.add_argument("--synonym-substitution-ratio", type=float, default=0.5)
    parser.add_argument("--context-aware-synonym-ratio", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    append_resume_shard_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    add_repo_to_path(args.markllm_root)

    from transformers import BertForMaskedLM, BertTokenizer
    from evaluation.tools.text_editor import (
        ContextAwareSynonymSubstitution,
        SynonymSubstitution,
        WordDeletion,
    )

    payload = read_json(args.input)
    out_path = args.output or str((ensure_dir(Path(args.input).parent) / "attacks.json"))
    out_dir = ensure_dir(Path(out_path).parent)
    sample_dir = ensure_dir(out_dir / "attack_samples")
    failure_dir = ensure_dir(out_dir / "attack_failures")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    attack1 = WordDeletion(ratio=args.word_deletion_ratio)
    attack2 = SynonymSubstitution(ratio=args.synonym_substitution_ratio)
    attack3 = ContextAwareSynonymSubstitution(
        ratio=args.context_aware_synonym_ratio,
        tokenizer=BertTokenizer.from_pretrained(args.bert_model),
        model=BertForMaskedLM.from_pretrained(args.bert_model).to(device),
        device=device,
    )

    detected = [
        s for s in payload["samples"] if s.get("prediction_with_watermark") and s.get("output_with_watermark")
    ]
    if args.max_samples is not None:
        detected = detected[: args.max_samples]
    detected = select_shard(detected, args.shard_index, args.num_shards)

    attack_rows_by_id = read_json_dir(sample_dir, "sample_id") if args.resume else {}
    failures_by_id = read_json_dir(failure_dir, "id") if args.resume else {}
    for sample in tqdm(detected, desc="Attacking detected WM samples"):
        sample_id = int(sample["id"])
        if args.resume and sample_id in attack_rows_by_id:
            continue
        if args.resume and sample_id in failures_by_id and not args.retry_failures:
            continue

        try:
            text = sample["output_with_watermark"]
            with WallTimer() as t1:
                word_deletion = attack1.edit(text)
            with WallTimer() as t2:
                synonym_substitution = attack2.edit(text)
            with WallTimer() as t3:
                context_aware = attack3.edit(text)
            row = {
                "sample_id": sample["id"],
                "original_text": text,
                "word_deletion": word_deletion,
                "synonym_substitution": synonym_substitution,
                "context_aware_synonym_substitution": context_aware,
                "attack_time_sec": {
                    "word_deletion": t1.elapsed,
                    "synonym_substitution": t2.elapsed,
                    "context_aware_synonym_substitution": t3.elapsed,
                },
            }
            write_checkpoint(sample_dir, sample_id, row)
            remove_checkpoint_if_exists(failure_dir, sample_id)
            attack_rows_by_id[sample_id] = row
            failures_by_id.pop(sample_id, None)
        except Exception as exc:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            failure = make_failure(sample_id, "attack_generation", exc)
            write_checkpoint(failure_dir, sample_id, failure)
            failures_by_id[sample_id] = failure
            if args.fail_fast:
                raise

    attack_rows = [attack_rows_by_id[k] for k in sorted(attack_rows_by_id)]
    failures = [failures_by_id[k] for k in sorted(failures_by_id)]

    result = {
        "metadata": {
            "source_generations": args.input,
            "scheme": payload.get("metadata", {}).get("scheme"),
            "seed": args.seed,
            "attacked_only_initially_detected": True,
            "attack_params": {
                "word_deletion_ratio": args.word_deletion_ratio,
                "synonym_substitution_ratio": args.synonym_substitution_ratio,
                "context_aware_synonym_ratio": args.context_aware_synonym_ratio,
                "bert_model": args.bert_model,
            },
            "num_attacked": len(attack_rows),
            "failures": failures,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
        },
        "attacks": attack_rows,
    }
    write_json(out_path, result)


if __name__ == "__main__":
    main()
