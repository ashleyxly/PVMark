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

IS_DEBUG = True

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

  if str(model.device) != str(expected_device):
    raise ValueError('Model device not as expected.')

  return model


def process_json_files_2(org_text):
    original_text_list = []
    
    with open(org_text, 'r') as file:
        data = json.load(file)
        wm_outputs_text = data.get('wm_outputs_text', [])
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
            if IS_DEBUG and len(batch_results) == 0:  # 只调试第一个文本
                print(f"Debug: nlls_float = {nlls_float}")
                print(f"Debug: avg_nll = {sum(nlls_float) / len(nlls_float)}")
            avg_nll = sum(nlls_float) / len(nlls_float)
            ppl = torch.exp(torch.tensor(avg_nll))
            batch_results.append(ppl.item())
        else:
            batch_results.append(float("inf"))
    
    return batch_results



local_model_path = os.environ.get("PVMark_GEMMA_7B_MODEL", "google/gemma-7b-it")
tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = load_model(MODEL_NAME, DEVICE, enable_watermarking=False)
model.to(DEVICE)

hash_type = 4
model_name_str = "GEMMA_7B_WM"


all_results = []

files_2 = 'tests/WM_UWM/Type_{}_ModelName.{}_results.json'.format(hash_type, model_name_str)

res_json_file_path='tests/WM_UWM/Only_PPL/PPL_Type_{}_ModelName.{}_results.json'.format(hash_type, model_name_str)

org_text = process_json_files_2(files_2)


valid_texts = [sub for sub in org_text if is_text_valid(sub)]
if IS_DEBUG:
    print("Total valid texts:", len(valid_texts))

valid_texts = valid_texts[:5]  # 可以限制数量进行测试

if IS_DEBUG:
    print("Processing texts...")



if IS_DEBUG:
    print("max_length", model.config.max_position_embeddings)


USE_EVALUATE_LIBRARY = True  # 设为False以使用更快的标准方法

if USE_EVALUATE_LIBRARY:
    try:
        import evaluate
        from tqdm import tqdm
        
        if IS_DEBUG:
            print("Loading evaluate perplexity metric...")
            start_time = time.time()
        
        perplexity = evaluate.load("perplexity", module_type="metric")
        
        if IS_DEBUG:
            load_time = time.time() - start_time
            print(f"Perplexity metric loaded in {load_time:.2f} seconds")
        
        def calculate_ppl_with_evaluate(texts, batch_size=1):
            """使用evaluate库计算PPL，分批处理以避免内存不足"""
            if IS_DEBUG:
                print(f"Calculating PPL for {len(texts)} texts using evaluate library...")
                print(f"Model ID: {model_name if isinstance(model_name, str) else model_name.value}")
                print(f"Batch size: {batch_size}")
                start_time = time.time()
            
            all_results = []
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    batch_results = perplexity.compute(
                        model_id=model_name if isinstance(model_name, str) else model_name.value,
                        predictions=batch_texts,
                        batch_size=1,  # evaluate库内部的batch_size
                    )
                    
                    all_results.extend(batch_results['perplexities'])
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Out of memory in batch {i}, reducing batch size...")
                        if batch_size > 1:
                            return calculate_ppl_with_evaluate(texts, batch_size // 2)
                        else:
                            print("Cannot reduce batch size further, skipping this text")
                            all_results.append(float("inf"))
                    else:
                        raise e
                
                except Exception as e:
                    print(f"Error processing batch {i}: {e}")
                    all_results.extend([float("inf")] * len(batch_texts))
            
            valid_results = [r for r in all_results if r != float("inf")]
            if valid_results:
                mean_perplexity = sum(valid_results) / len(valid_results)
            else:
                mean_perplexity = float("inf")
            
            if IS_DEBUG:
                compute_time = time.time() - start_time
                print(f"PPL calculation completed in {compute_time:.2f} seconds")
                print(f"Valid results: {len(valid_results)}/{len(all_results)}")
                print(f"Mean perplexity: {mean_perplexity}")
            
            return mean_perplexity
        
        ppl = calculate_ppl_with_evaluate(valid_texts, batch_size=1)
        
    except ImportError:
        print("evaluate库未安装，使用标准方法计算PPL")
        USE_EVALUATE_LIBRARY = False

if not USE_EVALUATE_LIBRARY:
    
    def calculate_ppl(texts, model, tokenizer, device, max_length=1024, stride=512):
        """使用标准方法计算PPL，优化内存使用"""
        results = []
        
        if IS_DEBUG:
            print(f"Calculating PPL for {len(texts)} texts using standard method...")
            start_time = time.time()
        
        for text_idx, text in enumerate(tqdm(texts, desc="Calculating PPL")):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length
                ).to(device)
                
                input_ids = encodings.input_ids
                seq_len = input_ids.size(1)
                
                if seq_len <= 1:
                    results.append(float("inf"))
                    del encodings, input_ids
                    continue
                    
                nlls = []
                
                for i in range(0, seq_len - 1, stride):
                    begin_loc = i
                    end_loc = min(i + max_length, seq_len)
                    
                    window_input_ids = input_ids[:, begin_loc:end_loc]
                    target_ids = window_input_ids.clone()
                    
                    target_ids[:, 0] = -100
                    
                    with torch.no_grad():
                        outputs = model(window_input_ids, labels=target_ids)
                        neg_log_likelihood = outputs.loss
                        
                    nlls.append(neg_log_likelihood)
                    
                    del window_input_ids, target_ids, outputs
                    
                    if end_loc >= seq_len:
                        break
                
                if nlls:
                    avg_nll = torch.stack(nlls).mean()
                    ppl = torch.exp(avg_nll)
                    results.append(ppl.item())
                else:
                    results.append(float("inf"))
                
                del encodings, input_ids, nlls
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Out of memory processing text {text_idx}, skipping...")
                    results.append(float("inf"))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                else:
                    raise e
            except Exception as e:
                if IS_DEBUG:
                    print(f"Error processing text {text_idx}: {e}")
                results.append(float("inf"))
        
        if IS_DEBUG:
            total_time = time.time() - start_time
            print(f"PPL calculation completed in {total_time:.2f} seconds")
            print(f"Average PPL: {sum(results) / len(results) if results else 'N/A'}")
        
        return results
    
    ppl_results = calculate_ppl(valid_texts, model, tokenizer, DEVICE, max_length=1024, stride=512)
    
    if ppl_results:
        valid_ppls = [p for p in ppl_results if p != float("inf")]
        if valid_ppls:
            ppl = torch.tensor(valid_ppls).mean()
        else:
            ppl = float("inf")
    else:
        ppl = float("inf")


try:
    if isinstance(ppl, torch.Tensor):
        ppl_value = round(ppl.item(), 4)
    else:
        ppl_value = round(float(ppl), 4)
    print("Org Text: ", ppl_value)
except Exception as e:
    print(f"Error calculating PPL: {e}")
    print("Org Text: inf")