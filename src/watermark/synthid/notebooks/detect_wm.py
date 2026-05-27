
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
        default=os.environ.get("PVMark_GEMMA_7B_MODEL", "google/gemma-7b-it"),
        
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

def flatten_json(json_file, key):
    flattened_list = []
    with open(json_file, 'r') as file:
        data = json.load(file)
        wm_outputs_text = data.get(key, [])
        for sub_list in wm_outputs_text:
            flattened_list.extend(sub_list)
    return flattened_list


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
    
    
    print("data_path: ", args.data_path)
    word_deletion_texts = flatten_json(args.data_path, 'word_deletion')
    synonym_substitution_texts = flatten_json(args.data_path, 'synonym_substitution')
    context_aware_synonym_substitution_texts = flatten_json(args.data_path, 'context_aware_synonym_substitution')
    
    word_deletion_mean_scores_list = []
    word_deletion_weighted_mean_scores_list = []
    synonym_substitution_mean_scores_list = []
    synonym_substitution_weighted_mean_scores_list = []
    context_aware_synonym_substitution_mean_scores_list = []
    context_aware_synonym_substitution_weighted_mean_scores_list = []
    
    
    for batch_id in tqdm.tqdm(range(NUM_BATCHES)):
        batch_word_deletion_texts = word_deletion_texts[ batch_id * BATCH_SIZE : (batch_id + 1) * BATCH_SIZE ]
        batch_synonym_substitution_texts = synonym_substitution_texts[ batch_id * BATCH_SIZE : (batch_id + 1) * BATCH_SIZE ]
        batch_context_aware_synonym_substitution_texts = context_aware_synonym_substitution_texts[ batch_id * BATCH_SIZE : (batch_id + 1) * BATCH_SIZE ]
        
        if len(batch_word_deletion_texts) < BATCH_SIZE:
            padding_needed = BATCH_SIZE - len(batch_word_deletion_texts)
            batch_word_deletion_texts += [word_deletion_texts[-1]] * padding_needed
            batch_synonym_substitution_texts += [synonym_substitution_texts[-1]] * padding_needed
            batch_context_aware_synonym_substitution_texts += [context_aware_synonym_substitution_texts[-1]] * padding_needed
            
        inputs_word_deletion = tokenizer(
            batch_word_deletion_texts,
            return_tensors='pt',
            padding=True,
        ).to(DEVICE)
        inputs_synonym_substitution = tokenizer(
            batch_synonym_substitution_texts,
            return_tensors='pt',
            padding=True,
        ).to(DEVICE)
        inputs_context_aware_synonym_substitution = tokenizer(
            batch_context_aware_synonym_substitution_texts,
            return_tensors='pt',
            padding=True,
        ).to(DEVICE)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            word_deletion_eos_token_mask = logits_processor.compute_eos_token_mask(
                input_ids=inputs_word_deletion['input_ids'],
                eos_token_id=tokenizer.eos_token_id,
            )
        except:
            continue
        word_deletion_eos_token_mask = word_deletion_eos_token_mask[:, CONFIG['ngram_len'] - 1 :]
        try:
            word_deletion_context_repetition_mask = logits_processor.compute_context_repetition_mask(
                input_ids=inputs_word_deletion['input_ids'],
            )   
        except:
            continue
            
        word_deletion_combined_mask = word_deletion_context_repetition_mask * word_deletion_eos_token_mask
        word_deletion_g_values = logits_processor.compute_g_values(
            input_ids=inputs_word_deletion['input_ids'],
        )
        word_deletion_mean_scores = detector_mean.mean_score(
            word_deletion_g_values.cpu().numpy(), word_deletion_combined_mask.cpu().numpy()
        )
        word_deletion_weighted_mean_scores = detector_mean.weighted_mean_score(
            word_deletion_g_values.cpu().numpy(), word_deletion_combined_mask.cpu().numpy()
        )
        word_deletion_mean_scores_list.append(word_deletion_mean_scores)
        word_deletion_weighted_mean_scores_list.append(word_deletion_weighted_mean_scores)
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
        
            synonym_substitution_eos_token_mask = logits_processor.compute_eos_token_mask(
                input_ids=inputs_synonym_substitution['input_ids'],
                eos_token_id=tokenizer.eos_token_id,
            )
            synonym_substitution_eos_token_mask = synonym_substitution_eos_token_mask[:, CONFIG['ngram_len'] - 1 :]
            synonym_substitution_context_repetition_mask = logits_processor.compute_context_repetition_mask(
                input_ids=inputs_synonym_substitution['input_ids'],
            )
            synonym_substitution_combined_mask = synonym_substitution_context_repetition_mask * synonym_substitution_eos_token_mask
            synonym_substitution_g_values = logits_processor.compute_g_values(
                input_ids=inputs_synonym_substitution['input_ids'],
            )
            synonym_substitution_mean_scores = detector_mean.mean_score(
                synonym_substitution_g_values.cpu().numpy(), synonym_substitution_combined_mask.cpu().numpy()
            )
            synonym_substitution_weighted_mean_scores = detector_mean.weighted_mean_score(
                synonym_substitution_g_values.cpu().numpy(), synonym_substitution_combined_mask.cpu().numpy()
            )
        except:
            continue
        synonym_substitution_mean_scores_list.append(synonym_substitution_mean_scores)
        synonym_substitution_weighted_mean_scores_list.append(synonym_substitution_weighted_mean_scores)
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            context_aware_synonym_substitution_eos_token_mask = logits_processor.compute_eos_token_mask(
                input_ids=inputs_context_aware_synonym_substitution['input_ids'],
                eos_token_id=tokenizer.eos_token_id,
            )
            context_aware_synonym_substitution_eos_token_mask = context_aware_synonym_substitution_eos_token_mask[:, CONFIG['ngram_len'] - 1 :]
            context_aware_synonym_substitution_context_repetition_mask = logits_processor.compute_context_repetition_mask(
                input_ids=inputs_context_aware_synonym_substitution['input_ids'],
            )
            context_aware_synonym_substitution_combined_mask = context_aware_synonym_substitution_context_repetition_mask * context_aware_synonym_substitution_eos_token_mask
            context_aware_synonym_substitution_g_values = logits_processor.compute_g_values(
                input_ids=inputs_context_aware_synonym_substitution['input_ids'],
            )
            context_aware_synonym_substitution_mean_scores = detector_mean.mean_score(
                context_aware_synonym_substitution_g_values.cpu().numpy(), context_aware_synonym_substitution_combined_mask.cpu().numpy()
            )
            context_aware_synonym_substitution_weighted_mean_scores = detector_mean.weighted_mean_score(
                context_aware_synonym_substitution_g_values.cpu().numpy(), context_aware_synonym_substitution_combined_mask.cpu().numpy()
            )
        except:
            continue
        context_aware_synonym_substitution_mean_scores_list.append(context_aware_synonym_substitution_mean_scores)
        context_aware_synonym_substitution_weighted_mean_scores_list.append(context_aware_synonym_substitution_weighted_mean_scores)
        
        
        gc.collect()
        torch.cuda.empty_cache()
        
    data_to_save = {
        "word_deletion_mean_scores": [tensor_to_list(score) for score in word_deletion_mean_scores_list],
        "word_deletion_weighted_mean_scores": [tensor_to_list(score) for score in word_deletion_weighted_mean_scores_list],
        "synonym_substitution_mean_scores": [tensor_to_list(score) for score in synonym_substitution_mean_scores_list],
        "synonym_substitution_weighted_mean_scores": [tensor_to_list(score) for score in synonym_substitution_weighted_mean_scores_list],
        "context_aware_synonym_substitution_mean_scores": [tensor_to_list(score) for score in context_aware_synonym_substitution_mean_scores_list],
        "context_aware_synonym_substitution_weighted_mean_scores": [tensor_to_list(score) for score in context_aware_synonym_substitution_weighted_mean_scores_list],
    }
    
    
    output_json_path = 'tests/WM_UWM/Attack/Detect/Detect_Attack_Type_{}_{}_Org_WM_results.json'.format(args.hash_type, MODEL_NAME)
    
    with open(output_json_path, 'w', encoding='utf-8') as outfile:
        json.dump(data_to_save, outfile, ensure_ascii=False, indent=0)
        
    print(f"结果已保存至 {output_json_path}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("############ Done ######################")
    
    


if __name__ == "__main__":
    args = parse_args()
    main(args)    
    