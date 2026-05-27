from __future__ import annotations
import os

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

from synthid_text import cuda_hash_cpp, gpu_hash, logits_processing, synthid_mixin  # noqa: E402
from time_hash_synthid_efficiency import (  # noqa: E402
    _build_online_contexts,
    clear_hash_caches,
    configure_hash_backend,
    resolve_device,
)


@torch.no_grad()
def main() -> None:
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
    device = resolve_device("cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")
    config = dict(synthid_mixin.DEFAULT_WATERMARKING_CONFIG)
    config["device"] = device
    processor = logits_processing.SynthIDLogitsProcessor(
        **config,
        top_k=40,
        temperature=1.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("PVMark_GPT2_MODEL", "gpt2")
    ).to(device)
    model.eval()
    token_length = 16
    batch_size = 1
    input_ids, _ = synthid_bench.build_detection_batch(
        tokenizer=tokenizer,
        news_text=synthid_bench.DEFAULT_NEWS_TEXT,
        batch_size=batch_size,
        token_length=token_length,
        device=device,
    )
    score_bank = model(input_ids).logits[:, :token_length, :].detach()
    replay_scores = torch.cat((score_bank[:, :1, :], score_bank[:, : token_length - 1, :]), dim=1)
    top_k_result = torch.topk(replay_scores, k=processor.top_k, dim=2)
    scores_top_k = top_k_result.values.contiguous()
    top_k_indices = top_k_result.indices.contiguous()
    contexts = _build_online_contexts(input_ids, processor.ngram_len)
    flat_contexts = contexts.reshape(token_length * batch_size, -1).contiguous()
    flat_indices = top_k_indices.permute(1, 0, 2).reshape(
        token_length * batch_size, processor.top_k
    ).contiguous()
    flat_scores = scores_top_k.permute(1, 0, 2).reshape(
        token_length * batch_size, processor.top_k
    ).contiguous()

    clear_hash_caches()
    numba_g, numba_context = gpu_hash.compute_g_values_use_mimc_gpu_split_context(
        flat_contexts,
        flat_indices,
        processor._rust_keys,
        dtype=None,
        return_context_tensor=True,
    )
    numba_rep = gpu_hash.compute_batched_repetition_flags_gpu(
        numba_context,
        token_length,
        batch_size,
        processor.context_history_size,
    )
    numba_scores = gpu_hash.update_scores_gpu(flat_scores, numba_g, numba_rep)
    cpp_context, cpp_g, cpp_rep = cuda_hash_cpp.debug_batched_mimc_cpp(
        flat_contexts,
        flat_indices,
        flat_scores,
        processor._rust_keys,
        token_length,
        batch_size,
        processor.context_history_size,
    )
    cpp_scores = cuda_hash_cpp.compute_batched_updated_scores_use_mimc_cpp(
        flat_contexts,
        flat_indices,
        flat_scores,
        processor._rust_keys,
        token_length,
        batch_size,
        processor.context_history_size,
    )
    torch.cuda.synchronize()

    print("context_equal", torch.equal(numba_context, cpp_context))
    print("g_equal", torch.equal(numba_g, cpp_g))
    print("rep_equal", torch.equal(numba_rep, cpp_rep))
    print("score_max_diff", float((numba_scores - cpp_scores).abs().max().item()))
    if not torch.equal(numba_context, cpp_context):
        idx = torch.nonzero(numba_context != cpp_context)[0]
        print("first_context_diff", idx.tolist(), int(numba_context[tuple(idx)].item()), int(cpp_context[tuple(idx)].item()))
    if not torch.equal(numba_g, cpp_g):
        idx = torch.nonzero(numba_g != cpp_g)[0]
        print("first_g_diff", idx.tolist(), int(numba_g[tuple(idx)].item()), int(cpp_g[tuple(idx)].item()))
    if not torch.equal(numba_rep, cpp_rep):
        idx = torch.nonzero(numba_rep != cpp_rep)[0]
        print("first_rep_diff", idx.tolist(), int(numba_rep[tuple(idx)].item()), int(cpp_rep[tuple(idx)].item()))
    diff = (numba_scores - cpp_scores).abs()
    if float(diff.max().item()) != 0.0:
        idx = torch.nonzero(diff == diff.max())[0]
        print(
            "max_score_diff_idx",
            idx.tolist(),
            float(numba_scores[tuple(idx)].item()),
            float(cpp_scores[tuple(idx)].item()),
            float(flat_scores[tuple(idx)].item()),
        )


if __name__ == "__main__":
    main()
