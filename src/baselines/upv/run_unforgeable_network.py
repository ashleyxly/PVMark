from __future__ import annotations

import argparse
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import LogitsProcessorList

from baseline_eval.common import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_UPV_ROOT,
    GenerationSample,
    WallTimer,
    add_repo_to_path,
    append_common_generation_args,
    bool_from_score,
    count_tokens,
    decode_new_tokens,
    ensure_dir,
    load_c4_records,
    load_causal_lm_and_tokenizer,
    make_failure,
    make_generation_payload,
    prepare_prompt,
    read_json_dir,
    remove_checkpoint_if_exists,
    set_seed,
    summarize_clean_detection,
    write_checkpoint,
    write_json,
)
from baseline_eval.upv_network import UpvNetworkDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UPV with the network-based detector on current C4 prompts.")
    append_common_generation_args(parser)
    parser.set_defaults(
        output_dir=f"{DEFAULT_RESULTS_ROOT}/unforgeable_network/opt1.3b_c4_num100",
    )
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument(
        "--generator-model",
        default=f"{DEFAULT_UPV_ROOT}/experiments/main_experiments/generator_model/combine_model.pt",
    )
    parser.add_argument("--network-detector-model", required=True)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--network-threshold", type=float, default=0.5)
    parser.add_argument("--key-z-threshold", type=float, default=1.0)
    parser.add_argument("--llm-name", default="opt-1.3b")
    parser.add_argument("--sampling-temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--detector-fixed-length", type=int, default=200)
    parser.add_argument("--device", default=None, help="Explicit torch device, e.g. cuda:2.")
    return parser.parse_args()


def key_score_ids(watermark: Any, token_ids: torch.Tensor, threshold: float) -> tuple[Optional[float], bool]:
    if token_ids.ndim > 1:
        token_ids = token_ids.squeeze(0)
    if token_ids.numel() <= watermark.min_prefix_len:
        return None, False
    _, _, z_score = watermark.green_token_mask_and_stats(token_ids.detach().cpu())
    z = float(z_score)
    return z, bool_from_score(z, threshold)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(out_dir / "generation_samples")
    failure_dir = ensure_dir(out_dir / "generation_failures")
    add_repo_to_path(args.upv_root)

    from watermark_model import CustomLogitsProcessor, Watermark, WatermarkLogitsProcessor

    set_seed(args.seed)
    model, tokenizer, device = load_causal_lm_and_tokenizer(
        args.model, use_gpu=args.use_gpu, load_fp16=args.load_fp16, device=args.device
    )

    watermark = Watermark(
        bit_number=args.bit_number,
        window_size=args.window_size,
        layers=args.layers,
        delta=args.delta,
        model_dir=args.generator_model,
        beam_size=args.beam_size,
    )
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
    network_detector = UpvNetworkDetector(
        upv_root=args.upv_root,
        detector_model=args.network_detector_model,
        tokenizer_path=args.model,
        bit_number=args.bit_number,
        layers=args.layers,
        fixed_length=args.detector_fixed_length,
        threshold=args.network_threshold,
        device=device,
    )

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.beam_size == 0:
        gen_kwargs.update({"do_sample": True, "top_k": args.top_k, "temperature": args.sampling_temp})
    else:
        gen_kwargs.update({"num_beams": args.beam_size})

    records = load_c4_records(args.dataset, args.max_samples, args.shard_index, args.num_shards)
    samples_by_id = read_json_dir(sample_dir, "id") if args.resume else {}
    failures_by_id = read_json_dir(failure_dir, "id") if args.resume else {}
    for row in tqdm(records, desc="UPV network generation"):
        sample_id = int(row["id"])
        if args.resume and sample_id in samples_by_id:
            continue
        if args.resume and sample_id in failures_by_id and not args.retry_failures:
            continue

        try:
            tokd_input = prepare_prompt(tokenizer, row["input_text"], device, args.prompt_max_length)
            prompt_len = tokd_input["input_ids"].shape[-1]

            sample_seed = int(args.seed) + sample_id
            set_seed(sample_seed)
            with WallTimer() as gen_plain_timer, torch.inference_mode():
                plain_output = model.generate(
                    **tokd_input,
                    logits_processor=LogitsProcessorList([custom_processor]),
                    **gen_kwargs,
                )
            plain_new_ids = plain_output[:, prompt_len:]

            set_seed(sample_seed)
            wm_generate_kwargs = dict(gen_kwargs)
            if args.no_repeat_ngram_size > 0:
                wm_generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            with WallTimer() as gen_wm_timer, torch.inference_mode():
                wm_output = model.generate(
                    **tokd_input,
                    logits_processor=LogitsProcessorList([watermark_processor]),
                    **wm_generate_kwargs,
                )
            wm_new_ids = wm_output[:, prompt_len:]

            plain_text = decode_new_tokens(tokenizer, plain_new_ids)
            wm_text = decode_new_tokens(tokenizer, wm_new_ids)

            with WallTimer() as det_plain_timer:
                plain_score, plain_pred = network_detector.score_text(plain_text)
            with WallTimer() as det_wm_timer:
                wm_score, wm_pred = network_detector.score_text(wm_text)

            key_plain_score, key_plain_pred = key_score_ids(watermark, plain_new_ids, args.key_z_threshold)
            key_wm_score, key_wm_pred = key_score_ids(watermark, wm_new_ids, args.key_z_threshold)

            sample = GenerationSample(
                id=sample_id,
                input_text=row["input_text"],
                reference_text_removed=row["reference_text_removed"],
                output_without_watermark=plain_text,
                output_with_watermark=wm_text,
                token_count_without_watermark=int(plain_new_ids.numel()),
                token_count_with_watermark=int(wm_new_ids.numel()),
                char_count_without_watermark=len(plain_text),
                char_count_with_watermark=len(wm_text),
                score_without_watermark=plain_score,
                score_with_watermark=wm_score,
                prediction_without_watermark=plain_pred,
                prediction_with_watermark=wm_pred,
                generation_time_without_watermark_sec=gen_plain_timer.elapsed,
                generation_time_with_watermark_sec=gen_wm_timer.elapsed,
                detection_time_without_watermark_sec=det_plain_timer.elapsed,
                detection_time_with_watermark_sec=det_wm_timer.elapsed,
                extra={
                    "sampling_config": gen_kwargs,
                    "wm_no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "retokenized_plain_count": count_tokens(tokenizer, plain_text),
                    "retokenized_wm_count": count_tokens(tokenizer, wm_text),
                    "detector_type": "network",
                    "key_based_z_score_without_watermark": key_plain_score,
                    "key_based_z_score_with_watermark": key_wm_score,
                    "key_based_prediction_without_watermark": key_plain_pred,
                    "key_based_prediction_with_watermark": key_wm_pred,
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
            failure = make_failure(sample_id, "unforgeable_network_generation", exc)
            write_checkpoint(failure_dir, sample_id, failure)
            failures_by_id[sample_id] = failure
            if args.fail_fast:
                raise

    sample_dicts = [samples_by_id[k] for k in sorted(samples_by_id)]
    failures = [failures_by_id[k] for k in sorted(failures_by_id)]
    payload = make_generation_payload(
        "unforgeable_network",
        args,
        sample_dicts,
        extra_metadata={
            "clean_detection_summary": summarize_clean_detection(sample_dicts),
            "failures": failures,
            "shard": {"index": args.shard_index, "num_shards": args.num_shards},
            "scheme_metadata": {
                "bit_number": args.bit_number,
                "window_size": args.window_size,
                "layers": args.layers,
                "delta": args.delta,
                "network_threshold": args.network_threshold,
                "network_detector_model": args.network_detector_model,
                "detector_fixed_length": args.detector_fixed_length,
                "generator_model": args.generator_model,
                "key_z_threshold_for_reference_only": args.key_z_threshold,
                "detector_type": "network",
            },
        },
    )
    write_json(out_dir / "generations.json", payload)
    write_json(out_dir / "metrics_summary.json", payload["metadata"]["clean_detection_summary"])


if __name__ == "__main__":
    main()
