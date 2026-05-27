from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from baseline_eval.common import (
    DEFAULT_PPL_MODEL,
    append_resume_shard_args,
    ensure_dir,
    make_failure,
    read_json,
    read_json_dir,
    remove_checkpoint_if_exists,
    select_shard,
    write_checkpoint,
    write_json,
)


CLEAN_SECTIONS = [
    ("clean.without_watermark", "without_watermark"),
    ("clean.with_watermark", "with_watermark"),
]
ATTACK_SECTIONS = [
    "original_text",
    "word_deletion",
    "synonym_substitution",
    "context_aware_synonym_substitution",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute OPT perplexity for clean and attacked texts.")
    parser.add_argument("--generations", required=True)
    parser.add_argument("--attacks", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--ppl-model", default=DEFAULT_PPL_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-length", type=int, default=None)
    append_resume_shard_args(parser)
    return parser.parse_args()


def safe_section(section: str) -> str:
    return section.replace(".", "__")


def load_ppl_model(model_path: str, device: str, load_fp16: bool) -> tuple[Any, Any, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for PPL evaluation; rerun on a GPU or pass --device cpu explicitly.")

    kwargs: dict[str, Any] = {}
    if load_fp16:
        kwargs["torch_dtype"] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device)
    model.eval()
    return model, tokenizer, device


def build_items(generations: dict[str, Any], attacks: dict[str, Any] | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sample in generations["samples"]:
        sample_id = int(sample["id"])
        items.append(
            {
                "section": "clean.without_watermark",
                "sample_id": sample_id,
                "text": sample.get("output_without_watermark", ""),
            }
        )
        items.append(
            {
                "section": "clean.with_watermark",
                "sample_id": sample_id,
                "text": sample.get("output_with_watermark", ""),
            }
        )
    if attacks:
        for row in attacks["attacks"]:
            sample_id = int(row["sample_id"])
            for field in ATTACK_SECTIONS:
                items.append(
                    {
                        "section": f"attacks.{field}",
                        "sample_id": sample_id,
                        "text": row.get(field, ""),
                    }
                )
    return items


def compute_batch(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    device: str,
    max_length: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    nonempty_positions: list[int] = []
    nonempty_texts: list[str] = []
    for pos, text in enumerate(texts):
        if isinstance(text, str) and text.strip():
            nonempty_positions.append(pos)
            nonempty_texts.append(text)
        else:
            results.append({"perplexity": None, "loss": None, "token_count": 0})

    computed: dict[int, dict[str, Any]] = {}
    if nonempty_texts:
        tok_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "add_special_tokens": True,
        }
        if max_length is not None:
            tok_kwargs.update({"truncation": True, "max_length": max_length})
        encoded = tokenizer(nonempty_texts, **tok_kwargs)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        token_counts = shift_mask.sum(dim=1)
        loss_sums = (losses * shift_mask).sum(dim=1)
        mean_losses = loss_sums / token_counts.clamp(min=1)

        for pos, loss, token_count in zip(nonempty_positions, mean_losses, token_counts):
            count = int(token_count.item())
            if count <= 0:
                computed[pos] = {"perplexity": None, "loss": None, "token_count": 0}
                continue
            loss_value = float(loss.item())
            computed[pos] = {
                "perplexity": float(math.exp(min(loss_value, 100.0))),
                "loss": loss_value,
                "token_count": count,
            }

    merged: list[dict[str, Any]] = []
    empty_iter = iter(results)
    for pos in range(len(texts)):
        if pos in computed:
            merged.append(computed[pos])
        else:
            merged.append(next(empty_iter))
    return merged


def process_items(
    items: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    device: str,
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
    item_root = ensure_dir(output_dir / "ppl_items")
    failure_root = ensure_dir(output_dir / "ppl_failures")

    rows_by_section: dict[str, dict[int, dict[str, Any]]] = {}
    failures_by_section: dict[str, dict[int, dict[str, Any]]] = {}
    for item in items:
        section = item["section"]
        rows_by_section.setdefault(section, {})
        failures_by_section.setdefault(section, {})

    for section in rows_by_section:
        if args.resume:
            rows_by_section[section] = read_json_dir(item_root / safe_section(section), "sample_id")
            failures_by_section[section] = read_json_dir(failure_root / safe_section(section), "id")

    pending: list[dict[str, Any]] = []
    for item in items:
        section = item["section"]
        sample_id = int(item["sample_id"])
        if args.resume and sample_id in rows_by_section[section]:
            continue
        if args.resume and sample_id in failures_by_section[section] and not args.retry_failures:
            continue
        pending.append(item)

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(pending), batch_size), desc="PPL batches"):
        batch = pending[start : start + batch_size]
        try:
            batch_results = compute_batch(
                [item["text"] for item in batch],
                model,
                tokenizer,
                device,
                args.max_length,
            )
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_results = []
            for item in batch:
                try:
                    batch_results.extend(
                        compute_batch([item["text"]], model, tokenizer, device, args.max_length)
                    )
                except Exception as exc:
                    failure = make_failure(
                        int(item["sample_id"]),
                        f"ppl.{item['section']}",
                        exc,
                        extra={"section": item["section"]},
                    )
                    failure_dir = failure_root / safe_section(item["section"])
                    write_checkpoint(failure_dir, int(item["sample_id"]), failure)
                    failures_by_section[item["section"]][int(item["sample_id"])] = failure
                    if args.fail_fast:
                        raise
                    batch_results.append(None)

        for item, result in zip(batch, batch_results):
            if result is None:
                continue
            section = item["section"]
            sample_id = int(item["sample_id"])
            row = {
                "section": section,
                "sample_id": sample_id,
                "perplexity": result["perplexity"],
                "loss": result["loss"],
                "token_count": result["token_count"],
            }
            write_checkpoint(item_root / safe_section(section), sample_id, row)
            remove_checkpoint_if_exists(failure_root / safe_section(section), sample_id)
            rows_by_section[section][sample_id] = row
            failures_by_section[section].pop(sample_id, None)

    failures = [
        failure
        for section in sorted(failures_by_section)
        for _, failure in sorted(failures_by_section[section].items())
    ]
    return rows_by_section, failures


def section_result(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[k] for k in sorted(rows)]
    values = [row["perplexity"] for row in ordered if row.get("perplexity") is not None]
    return {
        "perplexities": values,
        "mean_perplexity": float(sum(values) / len(values)) if values else None,
        "num_texts": len(values),
        "items": ordered,
    }


def main() -> None:
    args = parse_args()
    generations = read_json(args.generations)
    attacks = read_json(args.attacks) if args.attacks else None
    out = Path(args.output or str(Path(args.generations).with_name("ppl.json")))
    out_dir = ensure_dir(out.parent)

    items = build_items(generations, attacks)
    items = select_shard(items, args.shard_index, args.num_shards)

    model, tokenizer, device = load_ppl_model(args.ppl_model, args.device, args.load_fp16)
    rows_by_section, failures = process_items(items, out_dir, args, model, tokenizer, device)

    result: dict[str, Any] = {
        "metadata": {
            "scheme": generations.get("metadata", {}).get("scheme"),
            "source_generations": args.generations,
            "source_attacks": args.attacks,
            "ppl_model": args.ppl_model,
            "batch_size": args.batch_size,
            "device": args.device,
            "load_fp16": args.load_fp16,
            "max_length": args.max_length,
            "failures": failures,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
        },
        "clean": {
            clean_key: section_result(rows_by_section.get(section, {}))
            for section, clean_key in CLEAN_SECTIONS
        },
    }

    if attacks:
        result["attacks"] = {
            field: section_result(rows_by_section.get(f"attacks.{field}", {}))
            for field in ATTACK_SECTIONS
        }

    write_json(out, result)


if __name__ == "__main__":
    main()
