from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList

from path_config import GEN_MODEL, c4_prompt_file, kgw_result_file
from watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and detect hash-based KGW outputs.")
    parser.add_argument("--model_name_or_path", default=GEN_MODEL)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--input_num", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--prompt_max_length", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--generation_seed", type=int, default=20242024)
    parser.add_argument("--use_sampling", type=str2bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--n_beams", type=int, default=1)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--seeding_scheme", default="simple_1")
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--normalizers", default="")
    parser.add_argument("--ignore_repeated_bigrams", type=str2bool, default=False)
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--select_green_tokens", type=str2bool, default=True)
    parser.add_argument("--seed_separately", type=str2bool, default=True)
    parser.add_argument("--load_fp16", type=str2bool, default=False)
    parser.add_argument(
        "--hash_type",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4, 5],
        help="0=SHA256, 1=BLAKE2b, 2=KECCAK256, 3=Poseidon, 4=Poseidon2, 5=MiMC.",
    )
    parser.add_argument(
        "--hash_method",
        type=int,
        default=2,
        choices=[2, 4],
        help="Fixed variants only: 2=TwoToOneFixed, 4=ThreeToOneFixed.",
    )
    parser.add_argument("--skip_model_load", type=str2bool, default=False)
    return parser.parse_args()


def load_model(args: argparse.Namespace):
    is_seq2seq = any(model_type in args.model_name_or_path for model_type in ["t5", "T0"])
    is_decoder_only = any(model_type in args.model_name_or_path for model_type in ["gpt", "opt", "bloom"])
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif is_decoder_only:
        kwargs: dict[str, Any] = {}
        if args.load_fp16:
            kwargs.update({"torch_dtype": torch.float16, "device_map": "auto"})
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    if not args.load_fp16:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    return model, tokenizer, device, is_decoder_only


def build_detector(args: argparse.Namespace, tokenizer, device: str) -> WatermarkDetector:
    return WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers,
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
        hash_type=args.hash_type,
        hash_method=args.hash_method,
    )


def score_text(detector: WatermarkDetector, text: str) -> float | None:
    if len(text) <= detector.min_prefix_len:
        return None
    score = detector.detect(text)
    value = score.get("z_score")
    return float(value) if value is not None else None


def generate_pair(
    prompt: str,
    args: argparse.Namespace,
    model,
    tokenizer,
    device: str,
    is_decoder_only: bool,
) -> tuple[str, str, str]:
    processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
        hash_method=args.hash_method,
        hash_type=args.hash_type,
    )
    gen_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.use_sampling:
        gen_kwargs.update({"do_sample": True, "top_k": 0, "temperature": args.sampling_temp})
    else:
        gen_kwargs.update({"num_beams": args.n_beams})

    if args.prompt_max_length is None:
        max_positions = getattr(model.config, "max_position_embeddings", 2048)
        args.prompt_max_length = max_positions - args.max_new_tokens

    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.prompt_max_length,
    ).to(device)
    redecoded_prompt = tokenizer.batch_decode(tokenized["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    plain = model.generate(**tokenized, **gen_kwargs)
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    watermarked = model.generate(
        **tokenized,
        logits_processor=LogitsProcessorList([processor]),
        **gen_kwargs,
    )

    if is_decoder_only:
        prompt_len = tokenized["input_ids"].shape[-1]
        plain = plain[:, prompt_len:]
        watermarked = watermarked[:, prompt_len:]

    plain_text = tokenizer.batch_decode(plain, skip_special_tokens=True)[0]
    watermarked_text = tokenizer.batch_decode(watermarked, skip_special_tokens=True)[0]
    return redecoded_prompt, plain_text, watermarked_text


def load_prompts(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    prompts = []
    for row in rows:
        if "text_shortened" in row:
            prompts.append(
                {
                    "input_text": row.get("text_shortened", ""),
                    "reference_text_removed": row.get("text_removed", ""),
                }
            )
    return prompts


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def result_path(args: argparse.Namespace, part: int) -> Path:
    if args.output_json:
        path = Path(args.output_json)
        if args.save_interval > 0 and part > 1:
            return path.with_name(f"{path.stem}_part_{part}{path.suffix}")
        return path
    return kgw_result_file(args.input_num, args.hash_type, args.hash_method, part)


def main() -> None:
    args = parse_args()
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    args.normalizers = [item for item in args.normalizers.split(",") if item]

    input_path = Path(args.input_json) if args.input_json else c4_prompt_file(args.input_num)
    prompts = load_prompts(input_path)
    if args.skip_model_load:
        print(f"Loaded {len(prompts)} prompts from {input_path}")
        return

    model, tokenizer, device, is_decoder_only = load_model(args)
    detector = build_detector(args, tokenizer, device)

    chunk: dict[str, list[Any]] = {
        "z_score_without_watermark": [],
        "z_score_with_watermark": [],
        "input_text": [],
        "reference_text_removed": [],
        "output_without_watermark": [],
        "output_with_watermark": [],
    }
    part = 1

    for index, row in enumerate(tqdm(prompts), start=1):
        redecoded_prompt, plain_text, wm_text = generate_pair(
            row["input_text"],
            args,
            model,
            tokenizer,
            device,
            is_decoder_only,
        )
        chunk["z_score_without_watermark"].append(score_text(detector, plain_text))
        chunk["z_score_with_watermark"].append(score_text(detector, wm_text))
        chunk["input_text"].append(redecoded_prompt)
        chunk["reference_text_removed"].append(row["reference_text_removed"])
        chunk["output_without_watermark"].append(plain_text)
        chunk["output_with_watermark"].append(wm_text)

        if args.save_interval > 0 and index % args.save_interval == 0:
            write_json(result_path(args, part), chunk)
            chunk = {key: [] for key in chunk}
            part += 1

    if chunk["input_text"]:
        write_json(result_path(args, part), chunk)


if __name__ == "__main__":
    main()
