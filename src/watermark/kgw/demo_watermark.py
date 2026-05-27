"""
KGW Watermark Demo — Embed and detect watermarks using hash-based KGW.

Usage:
    python demo_watermark.py --model_name_or_path facebook/opt-1.3b
    python demo_watermark.py --model_name_or_path facebook/opt-1.3b --prompt "The quick brown fox"
    python demo_watermark.py --model_name_or_path facebook/opt-1.3b --max_new_tokens 200 --gamma 0.25 --delta 2.0

Requires: hash_rustlib (built from src/hash_function/hash-function via maturin/setuptools-rust)
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


DEFAULT_PROMPT = (
    "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
    "species of turtle native to the brackish coastal tidal marshes of the "
    "Northeastern and southern United States, and in Bermuda. It belongs "
    "to the monotypic genus Malaclemys. It has one of the largest ranges of "
    "all turtles in North America, stretching as far south as the Florida Keys "
    "and as far north as Cape Cod. The name 'terrapin' is derived from the "
    "Algonquian word torope. It applies to Malaclemys terrapin in both "
    "British English and American English."
)


def parse_args():
    parser = argparse.ArgumentParser(description="KGW watermark embedding and detection demo")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.25,
                        help="Fraction of vocab in the green list")
    parser.add_argument("--delta", type=float, default=2.0,
                        help="Logit bias for green list tokens")
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--seeding_scheme", type=str, default="simple_1")
    parser.add_argument("--detection_z_threshold", type=float, default=4.0)
    parser.add_argument("--normalizers", type=str, default="",
                        help="Comma-separated normalizer names (e.g. 'unicode,homoglyphs')")
    parser.add_argument("--ignore_repeated_bigrams", action="store_true")
    parser.add_argument("--select_green_tokens", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (auto-detected if not set)")
    return parser.parse_args()


def load_model(model_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_with_watermark(model, tokenizer, prompt, args, device):
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        select_green_tokens=args.select_green_tokens,
    )

    tokd_input = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
        max_length=model.config.max_position_embeddings - args.max_new_tokens,
    ).to(device)

    torch.manual_seed(args.generation_seed)
    output_without = model.generate(
        **tokd_input, max_new_tokens=args.max_new_tokens,
        do_sample=True, top_k=0, temperature=args.sampling_temp,
    )

    torch.manual_seed(args.generation_seed)
    output_with = model.generate(
        **tokd_input, max_new_tokens=args.max_new_tokens,
        logits_processor=LogitsProcessorList([watermark_processor]),
        do_sample=True, top_k=0, temperature=args.sampling_temp,
    )

    prompt_len = tokd_input["input_ids"].shape[-1]
    decoded_without = tokenizer.batch_decode(output_without[:, prompt_len:], skip_special_tokens=True)[0]
    decoded_with = tokenizer.batch_decode(output_with[:, prompt_len:], skip_special_tokens=True)[0]

    return decoded_without, decoded_with


def detect_watermark(text, tokenizer, args, device):
    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=args.detection_z_threshold,
        normalizers=args.normalizers.split(",") if args.normalizers else [],
        ignore_repeated_bigrams=args.ignore_repeated_bigrams,
        select_green_tokens=args.select_green_tokens,
    )
    if len(text) - 1 > detector.min_prefix_len:
        return detector.detect(text)
    return {"error": "text too short for detection"}


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Model: {args.model_name_or_path}")
    print(f"Device: {device}")
    print(f"gamma={args.gamma}, delta={args.delta}, seed={args.generation_seed}")
    print()

    model, tokenizer = load_model(args.model_name_or_path, device)
    print(f"Vocab size: {len(tokenizer)}")
    print()

    print("=" * 60)
    print("Generating text with and without watermark...")
    print("=" * 60)
    text_without, text_with = generate_with_watermark(model, tokenizer, args.prompt, args, device)

    print()
    print("--- WITHOUT WATERMARK ---")
    print(text_without)
    print()
    score_without = detect_watermark(text_without, tokenizer, args, device)
    print(f"Detection: {score_without}")
    print()

    print("--- WITH WATERMARK ---")
    print(text_with)
    print()
    score_with = detect_watermark(text_with, tokenizer, args, device)
    print(f"Detection: {score_with}")
    print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Without watermark: z={score_without.get('z_score', 'N/A'):.2f}, "
          f"prediction={score_without.get('prediction', 'N/A')}")
    print(f"  With watermark:    z={score_with.get('z_score', 'N/A'):.2f}, "
          f"prediction={score_with.get('prediction', 'N/A')}")


if __name__ == "__main__":
    main()
