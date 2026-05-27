from __future__ import annotations

import argparse
import fcntl
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from baseline_eval.common import (
    DEFAULT_PDW_ROOT,
    DEFAULT_RESULTS_ROOT,
    GenerationSample,
    WallTimer,
    add_repo_to_path,
    append_common_generation_args,
    ensure_dir,
    load_c4_records,
    load_causal_lm_and_tokenizer,
    make_generation_payload,
    make_failure,
    prepare_prompt,
    read_json_dir,
    remove_checkpoint_if_exists,
    set_seed,
    summarize_clean_detection,
    write_checkpoint,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run publicly-detectable asymmetric watermark baseline.")
    append_common_generation_args(parser)
    parser.set_defaults(
        output_dir=f"{DEFAULT_RESULTS_ROOT}/publicly_detectable/opt1.3b_c4_num100",
        prompt_max_length=1024,
    )
    parser.add_argument("--pdw-root", default=DEFAULT_PDW_ROOT)
    parser.add_argument("--sample-type", default="multinomial", choices=["argmax", "multinomial", "nucleus"])
    parser.add_argument("--message-length", type=int, default=None)
    parser.add_argument("--signature-segment-length", type=int, default=None)
    parser.add_argument("--bit-size", type=int, default=None)
    parser.add_argument("--max-planted-errors", type=int, default=None)
    parser.add_argument("--continue-until-stop-token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-time-before-plant-error", type=int, default=None)
    parser.add_argument("--plain-length-policy", default="match_watermarked_tokens", choices=["match_watermarked_tokens", "fixed_200"])
    parser.add_argument("--key-dir", default=None, help="Shared PDW key directory; use one shared dir across parallel shards.")
    parser.add_argument("--include-ids", default="", help="Comma-separated dataset ids to process after shard selection.")
    return parser.parse_args()


def seeded_for_pdw(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_pdw_keys(key_dir: Path, crypto_module: Any) -> tuple[str, str, str]:
    from petlib.pack import encode

    key_dir = ensure_dir(key_dir)
    sk_path = key_dir / "sk.pickle"
    pk_path = key_dir / "pk.pickle"
    params_path = key_dir / "params.pickle"
    lock_path = key_dir / "keys.lock"

    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if sk_path.exists() and pk_path.exists() and params_path.exists():
            return str(sk_path), str(pk_path), str(params_path)

        sk, pk, params = crypto_module.bls_generate_openssl()
        G = params[0]
        for path, value in [(sk_path, sk), (pk_path, pk), (params_path, G)]:
            tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
            with open(tmp_path, "wb") as f:
                pickle.dump(encode(value), f)
            os.replace(tmp_path, path)
        return str(sk_path), str(pk_path), str(params_path)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(out_dir / "generation_samples")
    failure_dir = ensure_dir(out_dir / "generation_failures")
    key_dir = ensure_dir(args.key_dir if args.key_dir else out_dir / "keys")
    add_repo_to_path(args.pdw_root)

    import crypto
    import detect
    import generate

    if args.max_time_before_plant_error is not None:
        generate.MAX_TIME_BEFORE_PLANT_ERROR = args.max_time_before_plant_error
        generate.MAX_TIME_BEFORE_GIVE_UP_SAMPLE_VALID_TOKEN = args.max_time_before_plant_error

    message_length = args.message_length if args.message_length is not None else crypto.DEFAULT_MESSAGE_LENGTH
    signature_segment_length = (
        args.signature_segment_length
        if args.signature_segment_length is not None
        else crypto.DEFAULT_SIGNATURE_SEGMENT_LENGTH
    )
    bit_size = args.bit_size if args.bit_size is not None else crypto.DEFAULT_BIT_SIZE
    max_planted_errors = (
        args.max_planted_errors
        if args.max_planted_errors is not None
        else crypto.DEFAULT_MAX_PLANTED_ERRORS
    )

    sk_path, pk_path, params_path = ensure_pdw_keys(key_dir, crypto)

    set_seed(args.seed)
    model, tokenizer, device = load_causal_lm_and_tokenizer(
        args.model, use_gpu=args.use_gpu, load_fp16=args.load_fp16
    )

    records = load_c4_records(args.dataset, args.max_samples, args.shard_index, args.num_shards)
    if args.include_ids.strip():
        include_ids = {int(item) for item in args.include_ids.split(",") if item.strip()}
        records = [row for row in records if int(row["id"]) in include_ids]
    samples_by_id = read_json_dir(sample_dir, "id") if args.resume else {}
    failures_by_id = read_json_dir(failure_dir, "id") if args.resume else {}

    for row in tqdm(records, desc="PDW generation"):
        sample_id = int(row["id"])
        if args.resume and sample_id in samples_by_id:
            continue
        if args.resume and sample_id in failures_by_id and not args.retry_failures:
            continue

        try:
            sample_seed = int(args.seed) + sample_id
            prompt_inputs = prepare_prompt(tokenizer, row["input_text"], device, args.prompt_max_length)
            prompt = tokenizer.batch_decode(prompt_inputs["input_ids"], skip_special_tokens=True)[0]

            seeded_for_pdw(sample_seed)
            with WallTimer() as gen_wm_timer:
                (
                    wm_text,
                    wm_tokens,
                    pk,
                    params,
                    counter,
                    planted_errors,
                ) = generate.generate_text_asymmetric(
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

            plain_tokens_to_generate = int(wm_tokens.numel())
            if args.plain_length_policy == "fixed_200":
                plain_tokens_to_generate = args.max_new_tokens

            seeded_for_pdw(sample_seed)
            with WallTimer() as gen_plain_timer:
                plain_text, plain_tokens = generate.generate_text_plain(
                    prompt,
                    plain_tokens_to_generate,
                    model,
                    tokenizer,
                    args.sample_type,
                )

            with WallTimer() as det_wm_timer:
                wm_pred = bool(
                    detect.search_for_asymmetric_watermark(
                        pk,
                        params,
                        wm_text,
                        message_length,
                        signature_segment_length,
                        bit_size,
                        max_planted_errors,
                    )
                )
            with WallTimer() as det_plain_timer:
                plain_pred = bool(
                    detect.search_for_asymmetric_watermark(
                        pk,
                        params,
                        plain_text,
                        message_length,
                        signature_segment_length,
                        bit_size,
                        max_planted_errors,
                    )
                )

            sample = GenerationSample(
                id=sample_id,
                input_text=row["input_text"],
                reference_text_removed=row["reference_text_removed"],
                output_without_watermark=plain_text,
                output_with_watermark=wm_text,
                token_count_without_watermark=int(plain_tokens.numel()),
                token_count_with_watermark=int(wm_tokens.numel()),
                char_count_without_watermark=len(plain_text),
                char_count_with_watermark=len(wm_text),
                score_without_watermark=None,
                score_with_watermark=None,
                prediction_without_watermark=plain_pred,
                prediction_with_watermark=wm_pred,
                generation_time_without_watermark_sec=gen_plain_timer.elapsed,
                generation_time_with_watermark_sec=gen_wm_timer.elapsed,
                detection_time_without_watermark_sec=det_plain_timer.elapsed,
                detection_time_with_watermark_sec=det_wm_timer.elapsed,
                extra={
                    "sample_type": args.sample_type,
                    "message_length": message_length,
                    "signature_segment_length": signature_segment_length,
                    "bit_size": bit_size,
                    "max_planted_errors": max_planted_errors,
                    "num_planted_errors": int(planted_errors),
                    "asymmetric_sample_counter": int(counter),
                    "plain_length_policy": args.plain_length_policy,
                },
            )
            sample_dict = sample.to_dict()
            write_checkpoint(sample_dir, sample_id, sample_dict)
            remove_checkpoint_if_exists(failure_dir, sample_id)
            samples_by_id[sample_id] = sample_dict
            failures_by_id.pop(sample_id, None)
        except Exception as exc:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            failure = make_failure(sample_id, "publicly_detectable_generation", exc)
            write_checkpoint(failure_dir, sample_id, failure)
            failures_by_id[sample_id] = failure
            if args.fail_fast:
                raise

    sample_dicts = [samples_by_id[k] for k in sorted(samples_by_id)]
    failures = [failures_by_id[k] for k in sorted(failures_by_id)]
    payload = make_generation_payload(
        "publicly_detectable",
        args,
        sample_dicts,
        extra_metadata={
            "clean_detection_summary": summarize_clean_detection(sample_dicts),
            "failures": failures,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            "scheme_metadata": {
                "sk_path": sk_path,
                "pk_path": pk_path,
                "params_path": params_path,
                "message_length": message_length,
                "signature_segment_length": signature_segment_length,
                "bit_size": bit_size,
                "max_planted_errors": max_planted_errors,
                "note": "Asymmetric PDW output length is determined by full signature embedding, not max_new_tokens.",
            },
        },
    )
    write_json(out_dir / "generations.json", payload)
    write_json(out_dir / "metrics_summary.json", payload["metadata"]["clean_detection_summary"])


if __name__ == "__main__":
    main()
