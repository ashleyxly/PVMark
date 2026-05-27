from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GPT2_PATH,
    DEFAULT_UPV_GENERATOR,
    DEFAULT_UPV_ROOT,
    Timer,
    append_jsonl,
    build_generation_payload,
    completed_keys,
    ensure_dir,
    load_eli5_prompts,
    load_existing_payload_records,
    record_key,
    set_seed,
    shard_items,
    sort_records,
    token_count,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UPV baseline on ELI5 prompts.")
    parser.add_argument("--mode", choices=["generate", "detect", "full"], default="full")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--upv-root", default=str(DEFAULT_UPV_ROOT))
    parser.add_argument("--generator-model", default=str(DEFAULT_UPV_GENERATOR))
    parser.add_argument("--output-dir", default="tests/baseline_comparison/upv_gpt2")
    parser.add_argument("--input", default=None, help="Generation JSON for detect mode.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--llm-name", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=4.0)
    return parser.parse_args()


def import_upv(root: str | Path) -> tuple[Any, Any, Any]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    from watermark_model import CustomLogitsProcessor, Watermark, WatermarkLogitsProcessor  # type: ignore

    return Watermark, WatermarkLogitsProcessor, CustomLogitsProcessor


def load_model_and_tokenizer(model_name_or_path: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


@contextlib.contextmanager
def torch_load_map_location_for_current_device() -> Any:
    """Make UPV's torch.load(model_dir) work when CUDA is unavailable.

    The upstream UPV code loads checkpoints with torch.load(model_dir), and some
    provided checkpoints were serialized from CUDA tensors. In CPU-only smoke
    tests this needs map_location='cpu'. Keep the monkey patch tightly scoped so
    it does not affect model loading elsewhere.
    """

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


def make_watermark(Watermark: Any, args: argparse.Namespace) -> Any:
    with torch_load_map_location_for_current_device():
        return Watermark(
            bit_number=args.bit_number,
            window_size=args.window_size,
            layers=args.layers,
            delta=args.delta,
            model_dir=args.generator_model,
            beam_size=args.beam_size,
        )


def make_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": False,
    }
    if args.beam_size > 0:
        kwargs.update({"num_beams": args.beam_size})
    else:
        kwargs.update(
            {
                "do_sample": True,
                "top_k": args.top_k,
                "temperature": args.temperature,
            }
        )
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
    return kwargs


def generate(args: argparse.Namespace) -> Path:
    Watermark, WatermarkLogitsProcessor, CustomLogitsProcessor = import_upv(args.upv_root)
    set_seed(args.seed)
    prompts = shard_items(
        load_eli5_prompts(args.dataset_path, args.limit),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    watermark = make_watermark(Watermark, args)
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        delta=args.delta,
        model=watermark.model,
        window_size=args.window_size,
        cache=watermark.cache,
        bit_number=args.bit_number,
        beam_size=args.beam_size,
        llm_name=args.llm_name,
    )
    custom_processor = CustomLogitsProcessor(llm_name=args.llm_name)
    generation_kwargs = make_generation_kwargs(args)
    output_dir = ensure_dir(args.output_dir)
    generation_path = output_dir / "generations.json"
    records_jsonl = output_dir / "generations.records.jsonl"
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()

    for item in prompts:
        if record_key({**item, "watermarked": True}) in existing_keys and record_key({**item, "watermarked": False}) in existing_keys:
            continue
        prompt = item["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        with Timer() as t:
            try:
                wm_output = model.generate(
                    **inputs,
                    logits_processor=LogitsProcessorList([watermark_processor]),
                    **generation_kwargs,
                )
                wm_completion_ids = wm_output[:, prompt_len:]
                wm_completion_text = tokenizer.batch_decode(wm_completion_ids, skip_special_tokens=True)[0]
                wm_full_text = tokenizer.batch_decode(wm_output, skip_special_tokens=True)[0]
                wm_completion_count = int(wm_completion_ids.shape[-1])
                wm_error = None
            except Exception as exc:
                wm_completion_text = ""
                wm_full_text = prompt
                wm_completion_count = 0
                wm_error = repr(exc)
        wm_record = {
            **item,
            "method": "upv_public",
            "watermarked": True,
            "prompt_template": "raw_title",
            "completion_text": wm_completion_text,
            "full_text": wm_full_text,
            "prompt_token_count": int(prompt_len),
            "completion_token_count": wm_completion_count,
            "generation_time_sec": t.elapsed,
            "method_metadata": {
                "bit_number": args.bit_number,
                "window_size": args.window_size,
                "layers": args.layers,
                "delta": args.delta,
                "beam_size": args.beam_size,
                "generator_model": args.generator_model,
                "generation_error": wm_error,
            },
        }
        if record_key(wm_record) not in existing_keys:
            append_jsonl(records_jsonl, wm_record)
            existing_keys.add(record_key(wm_record))

        with Timer() as t:
            try:
                uwm_output = model.generate(
                    **inputs,
                    logits_processor=LogitsProcessorList([custom_processor]),
                    **generation_kwargs,
                )
                uwm_completion_ids = uwm_output[:, prompt_len:]
                uwm_completion_text = tokenizer.batch_decode(uwm_completion_ids, skip_special_tokens=True)[0]
                uwm_full_text = tokenizer.batch_decode(uwm_output, skip_special_tokens=True)[0]
                uwm_completion_count = int(uwm_completion_ids.shape[-1])
                uwm_error = None
            except Exception as exc:
                uwm_completion_text = ""
                uwm_full_text = prompt
                uwm_completion_count = 0
                uwm_error = repr(exc)
        uwm_record = {
            **item,
            "method": "upv_public",
            "watermarked": False,
            "prompt_template": "raw_title",
            "completion_text": uwm_completion_text,
            "full_text": uwm_full_text,
            "prompt_token_count": int(prompt_len),
            "completion_token_count": uwm_completion_count,
            "generation_time_sec": t.elapsed,
            "method_metadata": {
                "bit_number": args.bit_number,
                "window_size": args.window_size,
                "layers": args.layers,
                "delta": args.delta,
                "beam_size": args.beam_size,
                "generator_model": args.generator_model,
                "generation_error": uwm_error,
            },
        }
        if record_key(uwm_record) not in existing_keys:
            append_jsonl(records_jsonl, uwm_record)
            existing_keys.add(record_key(uwm_record))

    records = sort_records(load_existing_payload_records(generation_path, records_jsonl, lambda r: record_key(r)))
    payload = build_generation_payload(
        method="upv_public",
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        seed=args.seed,
        records=records,
        generation_config={
            **generation_kwargs,
            "original_setting_note": "UPV original sampling uses top_k=20 and temperature=0.7; top_p is omitted unless explicitly set.",
        },
        extra_metadata={
            "upv_root": args.upv_root,
            "generator_model": args.generator_model,
            "threshold": args.threshold,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "checkpoint_jsonl": str(records_jsonl),
        },
    )
    write_json(generation_path, payload)

    if args.mode == "full":
        detection_path = output_dir / "detection.json"
        detect_generation_file(args, generation_path, detection_path, watermark=watermark, tokenizer=tokenizer)
    return generation_path


def detect_generation_file(
    args: argparse.Namespace,
    generation_path: str | Path,
    output_path: str | Path,
    *,
    watermark: Any | None = None,
    tokenizer: Any | None = None,
) -> Path:
    Watermark, _WatermarkLogitsProcessor, _CustomLogitsProcessor = import_upv(args.upv_root)
    if watermark is None:
        watermark = make_watermark(Watermark, args)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token

    import json

    with open(generation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    output_path = Path(output_path)
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r))

    for record in payload["records"]:
        if record_key(record) in existing_keys:
            continue
        text = record.get("completion_text", "")
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
            "method": "upv_public",
            "watermarked": bool(record.get("watermarked")),
            "detected": detected,
            "score": float(z_score) if z_score is not None else None,
            "green_token_count": int(green_count) if green_count is not None else None,
            "threshold": args.threshold,
            "detection_time_sec": t.elapsed,
            "error": error,
        }
        append_jsonl(records_jsonl, detection_record)
        existing_keys.add(record_key(record))

    detection_records = sort_records(
        load_existing_payload_records(output_path, records_jsonl, lambda r: record_key(r))
    )
    write_json(
        output_path,
        {
            "metadata": {
                "method": "upv_public",
                "detector": "public_z_score",
                "threshold": args.threshold,
                "bit_number": args.bit_number,
                "window_size": args.window_size,
                "generator_model": args.generator_model,
            },
            "records": detection_records,
        },
    )
    return Path(output_path)


def main() -> None:
    args = parse_args()
    if args.mode in {"generate", "full"}:
        generate(args)
    else:
        if not args.input:
            raise SystemExit("--input is required for detect mode")
        output_dir = ensure_dir(args.output_dir)
        detect_generation_file(args, args.input, output_dir / "detection.json")


if __name__ == "__main__":
    main()
