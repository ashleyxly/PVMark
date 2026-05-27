import os 
from collections.abc import Sequence
import enum
import time

import datasets
import huggingface_hub
from synthid_text import detector_mean
from synthid_text import logits_processing
from synthid_text import synthid_mixin
import tensorflow as tf
import transformers
import numpy as np
import json

import torch
from tqdm import tqdm

IS_DEBUG = False

class ModelName(enum.Enum):
  GPT2 = os.environ.get("PVMark_GPT2_MODEL", "gpt2")
  GEMMA_2B = os.environ.get("PVMark_GEMMA_2B_MODEL", "google/gemma-2b-it")
  GEMMA_7B = os.environ.get("PVMark_GEMMA_7B_MODEL", "google/gemma-7b-it")
  GEMMA_2_9B = os.environ.get("PVMark_GEMMA_2_9B_MODEL", "google/gemma-2-9b-it")



model_name = os.environ.get("PVMark_GEMMA_2_9B_MODEL", "google/gemma-2-9b-it")
MODEL_NAME = ModelName(model_name)

if MODEL_NAME is not ModelName.GPT2:
  huggingface_hub.notebook_login()
  

DEVICE = (
    torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)

CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG

BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1024
TEMPERATURE = 0.5
TOP_K = 40
TOP_P = 0.99
PPL_BATCH_SIZE = 32  # 批量计算PPL时每个批次的文本数量


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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name.value,
        device_map={"": DEVICE},  # Force all layers to same device
        torch_dtype=torch.bfloat16,
    )

  if str(model.device) != str(expected_device):
    raise ValueError('Model device not as expected.')

  return model


def process_json_files_2(org_text):
    original_text_list = []
    
    with open(org_text, 'r') as file:
        data = json.load(file)
        wm_outputs_text = data.get('nonwm_outputs_text', [])
        for sub_list in wm_outputs_text:
            original_text_list.extend(sub_list)
    
    return original_text_list

def is_text_valid(text):
    """Check if the text has at least one token."""
    encodings = tokenizer(
            text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
    ).to("cuda")

    attn_masks = encodings["attention_mask"]
    return torch.ge(attn_masks.sum(1), 1).item()  # Returns True if valid, else False


def compute_ppl_batch(texts_batch, model, tokenizer, device, max_length=None, stride=512):
    """批量计算PPL，显著提升性能"""
    if max_length is None:
        max_length = min(1024, model.config.max_position_embeddings)  # 限制最大长度
    
    batch_results = []
    
    for text in texts_batch:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        input_ids = encodings.input_ids
        seq_len = input_ids.size(1)
        
        if seq_len <= 1:
            batch_results.append(float("inf"))
            continue
            
        nlls = []
        
        if seq_len <= max_length:
            target_ids = input_ids.clone()
            target_ids[:, 0] = -100  # 第一个token不参与损失计算
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                loss = outputs.loss
                nlls.append(loss)
        else:
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                
                if end_loc - begin_loc <= 1:
                    continue
                    
                window_input_ids = input_ids[:, begin_loc:end_loc]
                target_ids = window_input_ids.clone()
                target_ids[:, 0] = -100
                
                with torch.no_grad():
                    outputs = model(window_input_ids, labels=target_ids)
                    loss = outputs.loss
                    nlls.append(loss)
                
                if end_loc >= seq_len:
                    break
        
        if nlls:
            nlls_float = [nll.float().item() for nll in nlls]
            if IS_DEBUG:  # 调试所有文本
                print(f"Debug: nlls_float = {nlls_float}")
                print(f"Debug: avg_nll = {sum(nlls_float) / len(nlls_float)}")
            avg_nll = sum(nlls_float) / len(nlls_float)
            ppl = torch.exp(torch.tensor(avg_nll))
            batch_results.append(ppl.item())
        else:
            batch_results.append(float("inf"))
    
    return batch_results



local_model_path = os.environ.get("PVMark_GEMMA_2_9B_MODEL", "google/gemma-2-9b-it")
tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = load_model(MODEL_NAME, DEVICE, enable_watermarking=False)

if IS_DEBUG:
    print(f"Model class: {type(model).__name__}")
    print(f"Model config: {model.config}")
    print(f"Model device: {model.device}")
    print(f"Is watermarked: {hasattr(model, 'logits_processor')}")
model.to(DEVICE)

hash_type = 4
model_name_str = "GPT2_UWM"


all_results = []

files_2 = 'tests/WM_UWM/Type_{}_ModelName.{}_results.json'.format(hash_type, model_name_str)

actual_model_name = model_name.split('/')[-1]  # Extract "gemma-2-9b-it" from path
res_json_file_path='tests/WM_UWM/Only_PPL/PPL_Type_{}_ModelName.{}_ActualModel.{}_NoOutliers_results.json'.format(hash_type, model_name_str, actual_model_name)

org_text = process_json_files_2(files_2)


valid_texts = [sub for sub in org_text if is_text_valid(sub)]
if IS_DEBUG:
    print("Total valid texts:", len(valid_texts))
    print("Sample text:", valid_texts[0][:200] if valid_texts else "No valid texts")

valid_texts = valid_texts  # 处理所有文本

if IS_DEBUG:
    print(f"Model name: {model.name_or_path}")
    print(f"Tokenizer name: {tokenizer.name_or_path}")
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    test_text = "Hello, world!"
    test_tokens = tokenizer(test_text, return_tensors="pt")
    print(f"Test text tokens: {test_tokens.input_ids}")
    print(f"Test text decoded: {tokenizer.decode(test_tokens.input_ids[0])}")

if IS_DEBUG:
    print("Processing texts...")

all_ppls = []
for text in tqdm(valid_texts):
    ppl_batch = compute_ppl_batch([text], model, tokenizer, DEVICE, max_length=1024, stride=512)
    all_ppls.extend(ppl_batch)

if all_ppls:
    avg_ppl = sum(all_ppls) / len(all_ppls)
    min_ppl = min(all_ppls)
    max_ppl = max(all_ppls)
    
    print(f"Original Average PPL: {avg_ppl}")
    print(f"Original Min PPL: {min_ppl}")
    print(f"Original Max PPL: {max_ppl}")
    
    import numpy as np
    ppl_array = np.array(all_ppls)
    Q1 = np.percentile(ppl_array, 25)
    Q3 = np.percentile(ppl_array, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    filtered_ppls = [ppl for ppl in all_ppls if lower_bound <= ppl <= upper_bound]
    removed_count = len(all_ppls) - len(filtered_ppls)
    
    filtered_avg_ppl = sum(filtered_ppls) / len(filtered_ppls)
    filtered_min_ppl = min(filtered_ppls)
    filtered_max_ppl = max(filtered_ppls)
    
    print(f"Filtered Average PPL: {filtered_avg_ppl}")
    print(f"Filtered Min PPL: {filtered_min_ppl}")
    print(f"Filtered Max PPL: {filtered_max_ppl}")
    print(f"Removed {removed_count} outliers")
    
    total_texts = len(org_text)
    valid_count = len(valid_texts)
    
    result_data = {
        "total_texts": total_texts,
        "valid_texts": valid_count,
        "outliers_removed": removed_count,
        "original_average_ppl": avg_ppl,
        "original_min_ppl": min_ppl,
        "original_max_ppl": max_ppl,
        "filtered_average_ppl": filtered_avg_ppl,
        "filtered_min_ppl": filtered_min_ppl,
        "filtered_max_ppl": filtered_max_ppl,
        "ppl_results": filtered_ppls
    }
    
    print(f"Results prepared: {valid_count} texts processed")
else:
    print("No valid texts to compute PPL")
    result_data = None

if result_data:
    import os
    
    os.makedirs(os.path.dirname(res_json_file_path), exist_ok=True)
    
    with open(res_json_file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {res_json_file_path}")
    print(f"File size: {os.path.getsize(res_json_file_path) / 1024:.2f} KB")
else:
    print("No results to save")