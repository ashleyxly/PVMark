from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GPT2_PATH,
    DEFAULT_PUBLICLY_DETECTABLE_ROOT,
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
    parser = argparse.ArgumentParser(
        description="Run publicly-detectable-watermark asymmetric baseline on ELI5 prompts."
    )
    parser.add_argument("--mode", choices=["generate", "detect", "full"], default="full")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--pdw-root", default=str(DEFAULT_PUBLICLY_DETECTABLE_ROOT))
    parser.add_argument("--output-dir", default="tests/baseline_comparison/pdw_gpt2")
    parser.add_argument("--input", default=None, help="Generation JSON for detect mode.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-type", default="multinomial", choices=["argmax", "multinomial", "nucleus"])
    parser.add_argument("--num-tokens", type=int, default=200, help="Used for UWM/plain generation.")
    parser.add_argument("--message-length", type=int, default=None)
    parser.add_argument("--signature-segment-length", type=int, default=None)
    parser.add_argument("--bit-size", type=int, default=None)
    parser.add_argument("--max-planted-errors", type=int, default=None)
    parser.add_argument(
        "--key-mode",
        choices=["shared", "per_sample"],
        default="shared",
        help="Use one PDW keypair for the whole run, or one keypair per ELI5 sample.",
    )
    parser.add_argument(
        "--key-dir",
        default=None,
        help="Directory for PDW sk/pk/params pickle files. Defaults to <output-dir>/pdw_keys.",
    )
    parser.add_argument("--continue-until-stop-token", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def import_pdw(root: str | Path) -> tuple[Any, Any, Any]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    sys.path.insert(0, str(root))
    import crypto  # type: ignore
    import detect  # type: ignore
    import generate  # type: ignore

    return crypto, detect, generate


def load_model_and_tokenizer(model_name_or_path: str, load_in_4bit: bool = False) -> tuple[Any, Any]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        max_length=getattr(model.config, "max_position_embeddings", None),
        truncation=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def install_context_limit_guard(pdw_generate: Any, model: Any) -> None:
    """Prevent GPT-style position embedding overrun from becoming a CUDA assert."""
    original_sample_token = pdw_generate.sample_token

    def guarded_sample_token(
        model_arg: Any,
        tokenizer: Any,
        inputs: torch.Tensor,
        past: Any,
        attn: torch.Tensor,
        vocab_size: int,
        sample_type: str,
        embedded_first_message_signature_pair: bool = False,
        top_p: float = 0.9,
        temperature: float = 0.9,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        max_positions = (
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_sequence_length", None)
        )
        if max_positions is not None:
            context_length = int(attn.shape[-1] if attn is not None else inputs.shape[-1])
            if past is not None and context_length >= int(max_positions):
                raise RuntimeError(
                    "PDW generation reached model context limit before sampling next token: "
                    f"context_length={context_length}, max_position_embeddings={int(max_positions)}"
                )
            if past is None and int(inputs.shape[-1]) > int(max_positions):
                raise RuntimeError(
                    "PDW prompt exceeds model context limit: "
                    f"prompt_length={int(inputs.shape[-1])}, max_position_embeddings={int(max_positions)}"
                )
        return original_sample_token(
            model_arg,
            tokenizer,
            inputs,
            past,
            attn,
            vocab_size,
            sample_type,
            embedded_first_message_signature_pair,
            top_p,
            temperature,
        )

    pdw_generate.sample_token = guarded_sample_token


def make_key_paths(args: argparse.Namespace, output_dir: Path, sample_id: int) -> tuple[str, str, str, dict[str, Any]]:
    key_dir = ensure_dir(args.key_dir or (output_dir / "pdw_keys"))
    key_id = "shared" if args.key_mode == "shared" else f"sample_{sample_id:06d}"
    sk_path = key_dir / f"{key_id}_sk.pickle"
    pk_path = key_dir / f"{key_id}_pk.pickle"
    params_path = key_dir / f"{key_id}_params.pickle"
    public_ref = {
        "key_mode": args.key_mode,
        "key_id": key_id,
        "pk_path": str(pk_path),
        "params_path": str(params_path),
    }
    return str(sk_path), str(pk_path), str(params_path), public_ref


def load_detection_material(pdw_root: str | Path, key_ref: dict[str, Any]) -> tuple[Any, Any]:
    import_pdw(pdw_root)
    from petlib.pack import decode  # type: ignore

    pk_path = key_ref.get("pk_path")
    params_path = key_ref.get("params_path")
    if not pk_path or not params_path:
        raise ValueError("missing pk_path/params_path in pdw_key metadata")
    with open(pk_path, "rb") as f:
        pk = decode(pickle.load(f))
    with open(params_path, "rb") as f:
        G = decode(pickle.load(f))
    params = (G, G.order(), G.gen1(), G.gen2(), G.pair)
    return pk, params


def generate(args: argparse.Namespace) -> Path:
    crypto, _detect, pdw_generate = import_pdw(args.pdw_root)
    message_length = args.message_length or crypto.DEFAULT_MESSAGE_LENGTH
    signature_segment_length = args.signature_segment_length or crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH
    bit_size = args.bit_size or crypto.DEFAULT_BIT_SIZE
    max_planted_errors = (
        args.max_planted_errors
        if args.max_planted_errors is not None
        else crypto.DEFAULT_MAX_PLANTED_ERRORS
    )

    set_seed(args.seed)
    prompts = shard_items(
        load_eli5_prompts(args.dataset_path, args.limit),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.load_in_4bit)
    install_context_limit_guard(pdw_generate, model)
    output_dir = ensure_dir(args.output_dir)
    generation_path = output_dir / "generations.json"
    records_jsonl = output_dir / "generations.records.jsonl"
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()

    for item in prompts:
        if record_key({**item, "watermarked": True}) in existing_keys and record_key({**item, "watermarked": False}) in existing_keys:
            continue
        prompt = item["prompt"]
        sk_path, pk_path, params_path, key_ref = make_key_paths(args, output_dir, item["sample_id"])

        with Timer() as t:
            try:
                (
                    wm_text,
                    wm_tokens,
                    pk,
                    params,
                    num_tokens,
                    num_planted_errors,
                ) = pdw_generate.generate_text_asymmetric(
                    prompt,
                    model,
                    tokenizer,
                    args.sample_type,
                    message_length,
                    signature_segment_length,
                    bit_size,
                    max_planted_errors,
                    sk_path,
                    pk_path,
                    params_path,
                    args.continue_until_stop_token,
                )
                wm_error = None
            except Exception as exc:
                wm_text = ""
                wm_tokens = torch.empty(0, dtype=torch.long)
                pk = None
                params = None
                num_tokens = 0
                num_planted_errors = None
                wm_error = repr(exc)
        wm_record = {
            **item,
            "method": "publicly_detectable_asymmetric",
            "watermarked": True,
            "prompt_template": "raw_title",
            "completion_text": wm_text,
            "full_text": prompt + wm_text,
            "prompt_token_count": token_count(tokenizer, prompt),
            "completion_token_count": int(num_tokens),
            "generation_time_sec": t.elapsed,
            "method_metadata": {
                "sample_type": args.sample_type,
                "message_length": message_length,
                "signature_segment_length": signature_segment_length,
                "bit_size": bit_size,
                "max_planted_errors": max_planted_errors,
                "num_planted_errors": num_planted_errors,
                "generation_error": wm_error,
                "pdw_key": key_ref,
            },
        }
        if record_key(wm_record) not in existing_keys:
            append_jsonl(records_jsonl, wm_record)
            existing_keys.add(record_key(wm_record))

        with Timer() as t:
            try:
                uwm_text, uwm_tokens = pdw_generate.generate_text_plain(
                    prompt,
                    args.num_tokens,
                    model,
                    tokenizer,
                    args.sample_type,
                )
                uwm_token_count = int(len(uwm_tokens))
                uwm_error = None
            except Exception as exc:
                uwm_text = ""
                uwm_token_count = 0
                uwm_error = repr(exc)
        uwm_record = {
            **item,
            "method": "publicly_detectable_asymmetric",
            "watermarked": False,
            "prompt_template": "raw_title",
            "completion_text": uwm_text,
            "full_text": prompt + uwm_text,
            "prompt_token_count": token_count(tokenizer, prompt),
            "completion_token_count": uwm_token_count,
            "generation_time_sec": t.elapsed,
            "method_metadata": {
                "sample_type": args.sample_type,
                "num_tokens": args.num_tokens,
                "generation_error": uwm_error,
                "pdw_key": key_ref,
            },
        }
        if record_key(uwm_record) not in existing_keys:
            append_jsonl(records_jsonl, uwm_record)
            existing_keys.add(record_key(uwm_record))

    records = sort_records(
        load_existing_payload_records(generation_path, records_jsonl, lambda r: record_key(r))
    )
    payload = build_generation_payload(
        method="publicly_detectable_asymmetric",
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        seed=args.seed,
        records=[strip_runtime_fields(r) for r in records],
        generation_config={
            "wm_sample_type": args.sample_type,
            "uwm_sample_type": args.sample_type,
            "uwm_num_tokens": args.num_tokens,
            "message_length": message_length,
            "signature_segment_length": signature_segment_length,
            "bit_size": bit_size,
            "max_planted_errors": max_planted_errors,
            "key_mode": args.key_mode,
            "key_dir": str(args.key_dir or (output_dir / "pdw_keys")),
            "continue_until_stop_token": args.continue_until_stop_token,
        },
        extra_metadata={
            "pdw_root": args.pdw_root,
            "note": "PDW public detection material is stored in per-record method_metadata.pdw_key. Secret sk files are kept only for reproducible local generation.",
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "checkpoint_jsonl": str(records_jsonl),
        },
    )
    write_json(generation_path, payload)

    if args.mode == "full":
        detection_path = output_dir / "detection.json"
        detect_records(args, records, detection_path)
    return generation_path


def strip_runtime_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if not k.startswith("_runtime_")}


def get_record_key_ref(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("method_metadata") or {}
    key_ref = metadata.get("pdw_key")
    return key_ref if isinstance(key_ref, dict) else None


def detect_records(args: argparse.Namespace, records: list[dict[str, Any]], output_path: str | Path) -> Path:
    crypto, pdw_detect, _generate = import_pdw(args.pdw_root)
    message_length = args.message_length or crypto.DEFAULT_MESSAGE_LENGTH
    signature_segment_length = args.signature_segment_length or crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH
    bit_size = args.bit_size or crypto.DEFAULT_BIT_SIZE
    max_planted_errors = (
        args.max_planted_errors
        if args.max_planted_errors is not None
        else crypto.DEFAULT_MAX_PLANTED_ERRORS
    )

    output_path = Path(output_path)
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if getattr(args, "resume", True) else set()
    last_pk = None
    last_params = None
    key_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
    for record in records:
        if record_key(record) in existing_keys:
            continue
        key_ref = get_record_key_ref(record)
        if record.get("watermarked"):
            last_pk = record.get("_runtime_pk")
            last_params = record.get("_runtime_params")
            pk = last_pk
            params = last_params
        else:
            pk = last_pk
            params = last_params
        text = record.get("completion_text", "")
        with Timer() as t:
            try:
                if (pk is None or params is None) and key_ref is not None:
                    cache_key = (str(key_ref.get("pk_path")), str(key_ref.get("params_path")))
                    if cache_key not in key_cache:
                        key_cache[cache_key] = load_detection_material(args.pdw_root, key_ref)
                    pk, params = key_cache[cache_key]
                if pk is None or params is None:
                    detected = None
                    error = "missing pdw pk/params"
                else:
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
            "watermarked": bool(record.get("watermarked")),
            "detected": detected,
            "score": None,
            "key_id": key_ref.get("key_id") if key_ref else None,
            "detection_time_sec": t.elapsed,
            "error": error,
        }
        append_jsonl(records_jsonl, detection_record)
        existing_keys.add(record_key(record))
    detection_records = sort_records(
        load_existing_payload_records(output_path, records_jsonl, lambda r: record_key(r))
    )
    payload = {
        "metadata": {
            "method": "publicly_detectable_asymmetric",
            "detector": "search_for_asymmetric_watermark",
            "message_length": message_length,
            "signature_segment_length": signature_segment_length,
            "bit_size": bit_size,
            "max_planted_errors": max_planted_errors,
            "key_mode": args.key_mode,
        },
        "records": detection_records,
    }
    write_json(output_path, payload)
    return Path(output_path)


def main() -> None:
    args = parse_args()
    if args.mode in {"generate", "full"}:
        generate(args)
    else:
        if not args.input:
            raise SystemExit("--input is required for detect mode")
        output_dir = ensure_dir(args.output_dir)
        import json

        with open(args.input, "r", encoding="utf-8") as f:
            payload = json.load(f)
        detect_records(args, payload["records"], output_dir / "detection.json")


if __name__ == "__main__":
    main()
