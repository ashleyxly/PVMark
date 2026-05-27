
import os 
from collections.abc import Sequence
import enum
import gc

import datasets
import huggingface_hub
from synthid_text import detector_mean
from synthid_text import logits_processing
from synthid_text import synthid_mixin
from synthid_text import detector_bayesian
import tensorflow as tf
import torch
import tqdm
import transformers
import numpy as np

import json

import argparse
from argparse import Namespace



def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=os.environ.get("PVMark_GPT2_MODEL", "gpt2"),
        
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--hash_type",
        type=int,
        default=3,
        help="Hash Type",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("PVMark_ELI5_SELECT_TEST", "experiment_data/prompts/select_test.json"),
        help="Data json path",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--outputs_len",
        type=int,
        default=200,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=125,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top_K tokens",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.99,
        help="Top_P",
    )
    args = parser.parse_args()
    return args



class ModelName(enum.Enum):
  GPT2 = os.environ.get("PVMark_GPT2_MODEL", "gpt2")
  GEMMA_2B = os.environ.get("PVMark_GEMMA_2B_MODEL", "google/gemma-2b-it")
  GEMMA_7B = os.environ.get("PVMark_GEMMA_7B_MODEL", "google/gemma-7b-it")



def load_model(
    model_name: ModelName,
    expected_device: torch.device,
    enable_watermarking: bool = False,
) -> transformers.PreTrainedModel:
  if model_name == ModelName.GPT2:
    model_cls = (
        synthid_mixin.SynthIDGPT2LMHeadModel
        if enable_watermarking
        else transformers.GPT2LMHeadModel
    )
    model = model_cls.from_pretrained(model_name.value, device_map='auto')
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
  else:
    model_cls = (
        synthid_mixin.SynthIDGemmaForCausalLM
        if enable_watermarking
        else transformers.GemmaForCausalLM
    )
    model = model_cls.from_pretrained(
        model_name.value,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )


  return model


def _compute_perplexity(
    outputs: torch.LongTensor,
    scores: torch.FloatTensor,
    eos_token_mask: torch.LongTensor,
    watermarked: bool = False,
) -> float:
  """Compute perplexity given the model outputs and the logits."""
  len_offset = len(scores)
  if watermarked:
    nll_scores = scores
  else:
    nll_scores = [
        torch.gather(
            -torch.log(torch.nn.Softmax(dim=1)(sc)),
            1,
            outputs[:, -len_offset + idx, None],
        )
        for idx, sc in enumerate(scores)
    ]
  nll_sum = torch.nan_to_num(
      torch.squeeze(torch.stack(nll_scores, dim=1), dim=2)
      * eos_token_mask.long(),
      posinf=0,
  )
  nll_sum = nll_sum.sum(dim=1)
  nll_mean = nll_sum / eos_token_mask.sum(dim=1)
  return nll_mean.sum(dim=0)


def _process_raw_prompt(prompt: Sequence[str], MODEL_NAME, tokenizer) -> str:
  """Add chat template to the raw prompt."""
  if MODEL_NAME == ModelName.GPT2:
    return prompt.decode().strip('"')
  else:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt.decode().strip('"')}],
        tokenize=False,
        add_generation_prompt=True,
    )


import jaxlib
def tensor_to_list(tensor):
    if isinstance(tensor, (jaxlib.xla_extension.ArrayImpl, torch.Tensor)):
        return tensor.tolist()  # Convert to a list if it's a tensor-like object
    return tensor  # Return as is if it's not a tensor


    




    
    


def main(args):
    DEVICE = (
        torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    )
    CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG
    
    MODEL_NAME = ModelName(args.model_name_or_path)
    if MODEL_NAME is not ModelName.GPT2:
        huggingface_hub.notebook_login()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME.value)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    logits_processor = logits_processing.SynthIDLogitsProcessor(
        **CONFIG, top_k=args.top_k, temperature=args.temperature
    )
    
    BATCH_SIZE = args.batch_size
    OUTPUTS_LEN = args.outputs_len
    TOP_K = args.top_k
    TOP_P = args.top_p
    TEMPERATURE = args.temperature
    NUM_BATCHES = args.num_batches
    
    
    with open(args.data_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    prompts = [d['title'] for d in data if 'title' in d]
    
    
    print("############ WM Model ######################")
    
    wm_model = load_model(MODEL_NAME, expected_device=DEVICE, enable_watermarking=True)
    wm_g_values = []
    wm_eos_masks = []
    wm_outputs = []
    wm_outputs_text = []
    wm_perplexities = []
    wm_combined_mask = []
    wm_mean_scores = []
    wm_weighted_mean_scores = []
    wm_context_repetition_mask = []
    
    for batch_id in tqdm.tqdm(range(NUM_BATCHES)):
        batch_prompt = prompts[
            batch_id * BATCH_SIZE:(batch_id + 1) * BATCH_SIZE]
        batch_prompt = [_process_raw_prompt(prompt.encode(), MODEL_NAME=MODEL_NAME, tokenizer=tokenizer) for prompt in batch_prompt]
        inputs = tokenizer(
            batch_prompt,
            return_tensors='pt',
            padding=True,
        ).to(DEVICE)
        _, inputs_len = inputs['input_ids'].shape

        with torch.no_grad():
            outputs = wm_model.generate(
                **inputs,
                do_sample=True,
                max_length=inputs_len + OUTPUTS_LEN,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                return_dict_in_generate=True,
                output_scores=True,
            )

        scores = outputs.scores
        outputs = outputs.sequences
        org_eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs[:, inputs_len:],
            eos_token_id=tokenizer.eos_token_id,
        )
        eos_token_mask = org_eos_token_mask[:, CONFIG['ngram_len'] - 1 :]
        
        context_repetition_mask = logits_processor.compute_context_repetition_mask(
            input_ids=outputs[:, inputs_len:],
        )
        combined_mask = context_repetition_mask * eos_token_mask

        wm_perplexities.append(_compute_perplexity(outputs, scores, org_eos_token_mask, watermarked=True))

        g_values = logits_processor.compute_g_values(
            input_ids=outputs[:, inputs_len:],
        )
        
        mean_scores = detector_mean.mean_score(
            g_values.cpu().numpy(), combined_mask.cpu().numpy()
        )
    
        weighted_mean_scores = detector_mean.weighted_mean_score(
            g_values.cpu().numpy(), combined_mask.cpu().numpy()
        )

        wm_g_values.append(g_values.cpu())
        wm_eos_masks.append(eos_token_mask.cpu())
        wm_outputs.append(outputs.cpu())
        wm_combined_mask.append(combined_mask.cpu())
        wm_mean_scores.append(mean_scores)
        wm_weighted_mean_scores.append(weighted_mean_scores)
        wm_context_repetition_mask.append(context_repetition_mask.cpu())
        
        gc.collect()
        torch.cuda.empty_cache()
        
    wm_perplexities_cpu = [p.cpu() if p.is_cuda else p for p in wm_perplexities]
    final_perplexity = torch.exp(
        torch.tensor(np.sum(wm_perplexities_cpu), dtype=torch.float32) / (BATCH_SIZE * NUM_BATCHES)
    )
    for i, output in enumerate(wm_outputs):
        wm_outputs_text.append(tokenizer.batch_decode(output, skip_special_tokens=True))
        
    data_to_save = {
        "wm_g_values": [tensor_to_list(g) for g in wm_g_values],
        "wm_eos_masks": [tensor_to_list(mask) for mask in wm_eos_masks],
        "wm_outputs": [tensor_to_list(output) for output in wm_outputs],
        "wm_combined_mask": [tensor_to_list(mask) for mask in wm_combined_mask],
        "wm_mean_scores": [tensor_to_list(scores) for scores in wm_mean_scores],
        "wm_weighted_mean_scores": [tensor_to_list(scores) for scores in wm_weighted_mean_scores],
        "wm_context_repetition_mask": [tensor_to_list(mask) for mask in wm_context_repetition_mask],
        "wm_perplexities": [tensor_to_list(p) for p in wm_perplexities],
        "wm_outputs_text": wm_outputs_text,
        "final_perplexity": final_perplexity.item()
    }
    
    
    output_json_path = 'tests/WM_UWM/Type_{}_{}_Org_WM_results.json'.format(args.hash_type, MODEL_NAME)
    
    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(data_to_save, outfile, ensure_ascii=False, indent=0)
        
    print(f"结果已保存至 {output_json_path}")
    
    del wm_model, wm_g_values, wm_eos_masks, wm_outputs, wm_outputs_text, wm_perplexities, wm_combined_mask, wm_mean_scores, wm_weighted_mean_scores, wm_context_repetition_mask
    gc.collect()
    torch.cuda.empty_cache()
    
    print("############ Done ######################")
    
    


if __name__ == "__main__":
    args = parse_args()
    main(args)    
    