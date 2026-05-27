from __future__ import annotations
import os

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
REPO_ROOT = NOTEBOOKS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(NOTEBOOKS_DIR))
import test_detect_time as synthid_bench  # noqa: E402

from synthid_text import logits_processing  # noqa: E402
from synthid_text import synthid_mixin  # noqa: E402

from time_hash_synthid_efficiency import (  # noqa: E402
    batched_embedding_replay_outputs,
    clear_hash_caches,
    collect_embedding_sequence_outputs,
    configure_hash_backend,
    resolve_device,
)


DEFAULT_MODEL = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check sequential GPU MiMC WET replay against batched replay."
    )
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--score-atol", type=float, default=1e-6)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def synthid_config(device: torch.device) -> dict:
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    return config


@torch.no_grad()
def main() -> None:
    args = parse_args()
    configure_hash_backend(
        hash_type=5,
        fused_g_values=True,
        fused_detect_g_values=False,
        fast_context_mask=False,
        gpu_hash_backend=True,
        gpu_fused_score_update=True,
        gpu_fused_history_update=False,
        cuda_cpp_online_wet=False,
        cpu_update_scores=False,
        compile_update_scores=False,
    )
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
    model.eval()

    input_ids, _ = synthid_bench.build_detection_batch(
        tokenizer=tokenizer,
        news_text=synthid_bench.DEFAULT_NEWS_TEXT,
        batch_size=args.batch_size,
        token_length=args.token_length,
        device=device,
    )
    score_bank = model(input_ids).logits[:, : args.token_length, :].detach()

    clear_hash_caches()
    sequential_processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    seq_scores, seq_indices, seq_original = collect_embedding_sequence_outputs(
        sequential_processor,
        input_ids,
        score_bank,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    clear_hash_caches()
    batched_processor = logits_processing.SynthIDLogitsProcessor(
        **synthid_config(device),
        top_k=args.top_k,
        temperature=args.temperature,
    )
    batch_scores, batch_indices, batch_original = batched_embedding_replay_outputs(
        batched_processor,
        input_ids,
        score_bank,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()

    index_equal = bool(torch.equal(seq_indices, batch_indices))
    original_max_diff = float((seq_original - batch_original).abs().max().item())
    score_max_diff = float((seq_scores - batch_scores).abs().max().item())
    score_allclose = bool(
        torch.allclose(seq_scores, batch_scores, atol=args.score_atol, rtol=0.0)
    )

    result = {
        "hash_type": logits_processing.HASH_TYPE,
        "gpu_hash_backend": logits_processing.GPU_HASH_BACKEND,
        "gpu_fused_score_update": logits_processing.GPU_FUSED_SCORE_UPDATE,
        "token_length": args.token_length,
        "batch_size": args.batch_size,
        "top_k": args.top_k,
        "top_k_indices_equal": index_equal,
        "original_top_k_score_max_diff": original_max_diff,
        "updated_score_max_diff": score_max_diff,
        "updated_score_allclose": score_allclose,
        "score_atol": args.score_atol,
    }
    print(result)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not index_equal or original_max_diff != 0.0 or not score_allclose:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
