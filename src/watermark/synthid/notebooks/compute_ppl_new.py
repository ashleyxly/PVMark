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



model_name = os.environ.get("PVMark_GEMMA_2B_MODEL", "google/gemma-2b-it")
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


def process_json_files_2(org_text, attack_text):
    original_text_list = []
    word_deletion_list = []
    synonym_substitution_list = []
    context_aware_synonym_substitution_list = []
    
    with open(org_text, 'r') as file:
        data = json.load(file)
        wm_outputs_text = data.get('wm_outputs_text', [])
        for sub_list in wm_outputs_text:
            original_text_list.extend(sub_list)
    
    with open(attack_text, 'r') as file:
        data = json.load(file)
        word_deletion_text = data.get('word_deletion', [])
        for sub_list in word_deletion_text:
            word_deletion_list.extend(sub_list)

        synonym_substitution_text = data.get('synonym_substitution', [])
        for sub_list in synonym_substitution_text:
            synonym_substitution_list.extend(sub_list)
        
        context_aware_synonym_substitution_text = data.get('context_aware_synonym_substitution', [])
        for sub_list in context_aware_synonym_substitution_text:
            context_aware_synonym_substitution_list.extend(sub_list)
    
    return original_text_list, word_deletion_list, synonym_substitution_list, context_aware_synonym_substitution_list

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



local_model_path = os.environ.get("PVMark_GEMMA_2B_MODEL", "google/gemma-2b-it")
tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = load_model(MODEL_NAME, DEVICE, enable_watermarking=False)
model.to(DEVICE)

hash_type = 3
ModelName = "GEMMA_2B_Org_WM"


all_results = []

files_2 = 'tests/WM_UWM/Type_{}_ModelName.{}_results.json'.format(hash_type, ModelName)
attack_text_path = 'tests/WM_UWM/Attack/Attack_Type_{}_ModelName.{}_results.json'.format(hash_type, ModelName)
res_json_file_path='tests/WM_UWM/Attack/PPL/PPL_new/PPL_Attack_Type_{}_ModelName.{}_results.json'.format(hash_type, ModelName)

org_text, attack1_text, attack2_text, attack3_text = process_json_files_2(files_2, attack_text_path)


valid_texts = [sub for sub in org_text if is_text_valid(sub)]
valid_texts = valid_texts[:5]
if IS_DEBUG:
    print("len(valid_texts)", len(valid_texts))

encodings = tokenizer(
    valid_texts,
    padding=True,
    return_tensors="pt",
).to(DEVICE)

max_length = model.config.max_position_embeddings
stride = 1
seq_len = encodings.input_ids.size(1)

if IS_DEBUG:
    print("max_length", max_length)
    print("seq_len", seq_len)


nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # 可能与最后一个步骤上的步幅不同
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

if len(nlls) > 0:
    ppl = torch.exp(torch.stack(nlls).mean())
else:
    ppl = float("inf")  # 处理空输入的情况


print("Org Text: ", ppl.item())