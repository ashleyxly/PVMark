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

sys.path.insert(0, str(SCRIPT_DIR))
from common import ensure_dir, write_json  # noqa: E402
from time_hash_synthid_efficiency import (  # noqa: E402
    collect_embedding_sequence_outputs,
    configure_hash_backend,
    resolve_device,
    synthid_config,
)

from synthid_text import logits_processing, synthid_mixin  # noqa: E402


DEFAULT_MODEL = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
DEFAULT_OUTPUT = Path(
    "tests/baseline_comparison/hash_synthid_variants_true_online_20260525"
) / "variant_online_equivalence.json"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Verify true-online WET equivalence for hash variants."
  )
  parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
  parser.add_argument("--model-name-or-path", default=str(DEFAULT_MODEL))
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--token-length", type=int, default=64)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--hash-types", type=int, nargs="+", default=[3, 4, 5])
  parser.add_argument("--score-atol", type=float, default=1e-3)
  return parser.parse_args()


def run_path(
    *,
    hash_type: int,
    device: torch.device,
    input_ids: torch.LongTensor,
    score_bank: torch.FloatTensor,
    compile_update_scores: bool,
    gpu_hash_backend: bool,
    gpu_fused_score_update: bool,
    gpu_fused_history_update: bool,
):
  configure_hash_backend(
      hash_type=hash_type,
      fused_g_values=True,
      fused_detect_g_values=False,
      fast_context_mask=False,
      gpu_hash_backend=gpu_hash_backend,
      gpu_fused_score_update=gpu_fused_score_update,
      gpu_fused_history_update=gpu_fused_history_update,
      cuda_cpp_online_wet=False,
      cpu_update_scores=False,
      compile_update_scores=compile_update_scores,
  )
  processor = logits_processing.SynthIDLogitsProcessor(
      **synthid_config(device),
      top_k=40,
      temperature=1.0,
  )
  return collect_embedding_sequence_outputs(processor, input_ids, score_bank)


def compare(reference, candidate, score_atol: float) -> dict:
  ref_scores, ref_indices, ref_original = reference
  cand_scores, cand_indices, cand_original = candidate
  diff = torch.max(torch.abs(ref_scores - cand_scores)).item()
  original_diff = torch.max(torch.abs(ref_original - cand_original)).item()
  return {
      "top_k_indices_equal": bool(torch.equal(ref_indices, cand_indices)),
      "original_top_k_score_max_diff": float(original_diff),
      "updated_score_max_diff": float(diff),
      "updated_score_allclose": bool(
          torch.allclose(ref_scores, cand_scores, atol=score_atol, rtol=0.0)
      ),
      "score_atol": score_atol,
  }


def main() -> None:
  args = parse_args()
  device = resolve_device(args.device)
  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to(device)
  model.eval()

  input_ids, available_tokens = synthid_bench.build_detection_batch(
      tokenizer=tokenizer,
      news_text=synthid_bench.DEFAULT_NEWS_TEXT,
      batch_size=args.batch_size,
      token_length=args.token_length,
      device=device,
  )
  with torch.no_grad():
    score_bank = model(input_ids).logits[:, : args.token_length, :].detach()

  output = {
      "token_length": args.token_length,
      "batch_size": args.batch_size,
      "available_tokens": available_tokens,
      "reference": "Rust fused g-values + default torch score update",
      "comparisons": {},
  }

  for hash_type in args.hash_types:
    reference = run_path(
        hash_type=hash_type,
        device=device,
        input_ids=input_ids,
        score_bank=score_bank,
        compile_update_scores=False,
        gpu_hash_backend=False,
        gpu_fused_score_update=False,
        gpu_fused_history_update=False,
    )
    output["comparisons"][str(hash_type)] = {}
    compiled = run_path(
        hash_type=hash_type,
        device=device,
        input_ids=input_ids,
        score_bank=score_bank,
        compile_update_scores=True,
        gpu_hash_backend=False,
        gpu_fused_score_update=False,
        gpu_fused_history_update=False,
    )
    output["comparisons"][str(hash_type)]["compile_update_scores"] = compare(
        reference,
        compiled,
        args.score_atol,
    )
    gpu_score = run_path(
        hash_type=hash_type,
        device=device,
        input_ids=input_ids,
        score_bank=score_bank,
        compile_update_scores=False,
        gpu_hash_backend=(hash_type == 5),
        gpu_fused_score_update=True,
        gpu_fused_history_update=(hash_type == 5),
    )
    output["comparisons"][str(hash_type)]["gpu_fused_score_update"] = compare(
        reference,
        gpu_score,
        args.score_atol,
    )
    if hash_type == 5:
      rust_mimc = run_path(
          hash_type=hash_type,
          device=device,
          input_ids=input_ids,
          score_bank=score_bank,
          compile_update_scores=False,
          gpu_hash_backend=False,
          gpu_fused_score_update=False,
          gpu_fused_history_update=False,
      )
      output["comparisons"][str(hash_type)]["mimc_gpu_vs_rust"] = compare(
          rust_mimc,
          gpu_score,
          args.score_atol,
      )

  out_path = Path(args.output)
  ensure_dir(out_path.parent)
  write_json(out_path, output)
  print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
