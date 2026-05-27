from __future__ import annotations
import os

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import upv_network_detector as und  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time UPV network detector inference.")
    parser.add_argument("--output-dir", default="tests/baseline_comparison/upv_network_detector_gpt2_eli5")
    parser.add_argument("--model-name-or-path", default=os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
    parser.add_argument("--generations", default="tests/baseline_comparison/upv_gpt2/generations.json")
    parser.add_argument("--attacks", default="tests/baseline_comparison/upv_gpt2/attacks.json")
    parser.add_argument(
        "--subnet",
        default=os.environ.get("PVMark_UPV_SUBNET", str(Path(os.environ.get("PVMark_UPV_ROOT", "external/unforgeable_watermark")) / "experiments/main_experiments/generator_model/sub_net.pt")),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--z-value", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    return parser.parse_args()


def time_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
    *,
    runs: int,
) -> dict[str, Any]:
    warmup_texts = texts[: min(len(texts), 512)]
    for _ in range(args.warmup):
        und.predict_texts(model, tokenizer, warmup_texts, args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    durations: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        scores = und.predict_texts(model, tokenizer, texts, args, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if len(scores) != len(texts):
            raise RuntimeError(f"expected {len(texts)} scores, got {len(scores)}")
        durations.append(time.perf_counter() - start)

    return {
        "num_texts": len(texts),
        "runs": runs,
        "durations_sec": durations,
        "mean_total_sec": statistics.mean(durations),
        "median_total_sec": statistics.median(durations),
        "mean_sec_per_text": statistics.mean(durations) / len(texts),
        "median_sec_per_text": statistics.median(durations) / len(texts),
        "texts_per_sec_mean": len(texts) / statistics.mean(durations),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = und.resolve_device(args.device)
    tokenizer = und.load_tokenizer(args.model_name_or_path)
    model = und.load_detector(args, device)

    with open(args.generations, "r", encoding="utf-8") as f:
        generation_records = json.load(f)["records"]
    normal_texts = [r.get("completion_text", "") for r in generation_records]

    with open(args.attacks, "r", encoding="utf-8") as f:
        attack_records = json.load(f)["records"]
    attack_texts: list[str] = []
    by_attack: dict[str, list[str]] = {}
    for record in attack_records:
        for attack_name, text in record.get("attacks", {}).items():
            attack_texts.append(text)
            by_attack.setdefault(attack_name, []).append(text)

    output = {
        "metadata": {
            "checkpoint": str(und.checkpoint_path(args)),
            "model_name_or_path": args.model_name_or_path,
            "device": str(device),
            "batch_size": args.eval_batch_size,
            "note": "Wall-clock batched inference timing includes tokenization, feature construction, and network forward pass.",
        },
        "normal_2000": time_texts(model, tokenizer, normal_texts, args, device, runs=args.runs),
        "attacks_6000": time_texts(model, tokenizer, attack_texts, args, device, runs=args.runs),
        "by_attack": {
            name: time_texts(model, tokenizer, texts, args, device, runs=max(3, args.runs // 2))
            for name, texts in sorted(by_attack.items())
        },
    }

    out_path = output_dir / f"efficiency_network_z{und.format_z(args.z_value)}_timing.json"
    und.write_json(out_path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
