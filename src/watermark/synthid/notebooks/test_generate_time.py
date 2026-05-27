
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




class ModelName(enum.Enum):
  GPT2 = os.environ.get("PVMark_GPT2_MODEL", "gpt2")
  GEMMA_2B = os.environ.get("PVMark_GEMMA_2B_MODEL", "google/gemma-2b-it")
  GEMMA_7B = 'google/gemma-7b-it'


model_name = os.environ.get("PVMark_GPT2_MODEL", "gpt2")
MODEL_NAME = ModelName(model_name)

if MODEL_NAME is not ModelName.GPT2:
  huggingface_hub.notebook_login()


DEVICE = (
    torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
)
DEVICE

import os
CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG
CONFIG


BATCH_SIZE = 8
NUM_BATCHES = 320
OUTPUTS_LEN = 1
TEMPERATURE = 1.0
TOP_K = 40
TOP_P = 0.99

local_model_path = os.environ.get("PVMark_GPT2_MODEL", "gpt2")

tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

logits_processor = logits_processing.SynthIDLogitsProcessor(
    **CONFIG, top_k=TOP_K, temperature=TEMPERATURE
)



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


def _process_raw_prompt(prompt: Sequence[str]) -> str:
  """Add chat template to the raw prompt."""
  if MODEL_NAME == ModelName.GPT2:
    return prompt.decode().strip('"')
  else:
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt.decode().strip('"')}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _detect_watermark_and_print(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    scores: torch.Tensor,
) -> None:
    """Detect watermark and print the scores."""
    
    _, inputs_len = inputs['input_ids'].shape
    outputs_org = outputs
    outputs = outputs[:, inputs_len:]

    eos_token_mask = logits_processor.compute_eos_token_mask(
        input_ids=outputs,
        eos_token_id=tokenizer.eos_token_id,
    )[:, CONFIG['ngram_len'] - 1 :]

    context_repetition_mask = logits_processor.compute_context_repetition_mask(
        input_ids=outputs,
    )
    combined_mask = context_repetition_mask * eos_token_mask

    g_values = logits_processor.compute_g_values(
        input_ids=outputs,
    )

    mean_scores = detector_mean.mean_score(
        g_values.cpu().numpy(), combined_mask.cpu().numpy()
    )
    
    weighted_mean_scores = detector_mean.weighted_mean_score(
        g_values.cpu().numpy(), combined_mask.cpu().numpy()
    )
    
    return mean_scores, weighted_mean_scores


gc.collect()

batch_size = 1
example_inputs = [
    'I enjoy walking with my cute dog',
    'I am from New York',
    'The test was not so very hard after all',
    "I don't think they can score twice in so short a time",
]
example_inputs = example_inputs * (int(batch_size / 4) + 1)
example_inputs = example_inputs[:batch_size]

inputs = tokenizer(
    example_inputs,
    return_tensors='pt',
    padding=True,
).to(DEVICE)

_, inputs_len = inputs['input_ids'].shape

model = load_model(MODEL_NAME, expected_device=DEVICE, enable_watermarking=True)
torch.manual_seed(0)

gc.collect()
torch.cuda.empty_cache()

import time
start_time = time.time()
for i in range(10):
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=OUTPUTS_LEN,
        top_k=40,
        return_dict_in_generate=True,
        output_scores=True,
    )
end_time = time.time()
mean_time = (end_time - start_time) / 10
print(f"Mean time: {mean_time}")
exit()
    
scores = outputs.scores
outputs = outputs.sequences


print('WM_Output:\n' + 100 * '-')
for i, output in enumerate(outputs):
  print(tokenizer.decode(output, skip_special_tokens=True))
  print(100 * '-')

del model
gc.collect()
torch.cuda.empty_cache()

uwm_model = load_model(MODEL_NAME, expected_device=DEVICE, enable_watermarking=False)
uwm_outputs = uwm_model.generate(
    **inputs,
    do_sample=True,
    temperature=TEMPERATURE,
    max_length=OUTPUTS_LEN,
    top_k=40,
    return_dict_in_generate=True,
    output_scores=True,
)

uwm_scores = uwm_outputs.scores
uwm_outputs = uwm_outputs.sequences


print('UWM_Output:\n' + 100 * '-')
for i, output in enumerate(uwm_outputs):
  print(tokenizer.decode(output, skip_special_tokens=True))
  print(100 * '-')

del uwm_model
gc.collect()
torch.cuda.empty_cache()



wm_mean_scores, wm_weighted_mean_scores = _detect_watermark_and_print(
    inputs=inputs,
    outputs=outputs,
    logits_processor=logits_processor,
    scores=scores,
    
)

uwm_mean_scores, uwm_weighted_mean_scores = _detect_watermark_and_print(
    inputs=inputs,
    outputs=uwm_outputs,
    logits_processor=logits_processor,
    scores=uwm_scores,
)

print('Mean scores for watermarked responses: ', wm_mean_scores)
print('Mean scores for unwatermarked responses: ', uwm_mean_scores)
print('Weighted Mean scores for watermarked responses: ', wm_weighted_mean_scores)
print('Weighted Mean scores for unwatermarked responses: ', uwm_weighted_mean_scores)


exit()






eli5_prompts = datasets.load_dataset(os.environ.get("PVMark_ELI5_DATASET", "eli5"))

gc.collect()

model = load_model(MODEL_NAME, expected_device=DEVICE)
torch.manual_seed(0)

nonwm_g_values = []
nonwm_eos_masks = []
nonwm_outputs = []
perplexities = []

for batch_id in tqdm.tqdm(range(NUM_BATCHES)):
  prompts = eli5_prompts['train']['title'][
      batch_id * BATCH_SIZE:(batch_id + 1) * BATCH_SIZE]
  prompts = [_process_raw_prompt(prompt.encode()) for prompt in prompts]
  inputs = tokenizer(
      prompts,
      return_tensors='pt',
      padding=True,
  ).to(DEVICE)
  _, inputs_len = inputs['input_ids'].shape

  outputs = model.generate(
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
  eos_token_mask = logits_processor.compute_eos_token_mask(
      input_ids=outputs[:, inputs_len:],
      eos_token_id=tokenizer.eos_token_id,
  )

  perplexities.append(_compute_perplexity(outputs, scores, eos_token_mask))

  g_values = logits_processor.compute_g_values(
      input_ids=outputs[:, inputs_len:],
  )

  nonwm_g_values.append(g_values.cpu())
  nonwm_eos_masks.append(eos_token_mask.cpu())
  nonwm_outputs.append(outputs.cpu())

  del inputs, prompts, eos_token_mask, g_values, outputs

del model, nonwm_g_values, nonwm_eos_masks, nonwm_outputs
gc.collect()

final_perplexity = torch.exp(np.sum(perplexities) / (BATCH_SIZE * NUM_BATCHES))
print(f"Perplexity of unwatermarked model: {final_perplexity}")

gc.collect()

model = load_model(MODEL_NAME, expected_device=DEVICE, enable_watermarking=True)
torch.manual_seed(0)

wm_outputs = []
wm_g_values = []
wm_eos_masks = []
perplexities = []

for batch_id in tqdm.tqdm(range(NUM_BATCHES)):
  prompts = eli5_prompts['train']['title'][
      batch_id * BATCH_SIZE:(batch_id + 1) * BATCH_SIZE]
  prompts = [_process_raw_prompt(prompt.encode()) for prompt in prompts]
  inputs = tokenizer(
      prompts,
      return_tensors='pt',
      padding=True,
  ).to(DEVICE)
  _, inputs_len = inputs['input_ids'].shape

  outputs = model.generate(
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

  eos_token_mask = logits_processor.compute_eos_token_mask(
      input_ids=outputs[:, inputs_len:],
      eos_token_id=tokenizer.eos_token_id,
  )

  perplexities.append(_compute_perplexity(outputs, scores, eos_token_mask, watermarked=True))

  g_values = logits_processor.compute_g_values(
      input_ids=outputs[:, inputs_len:],
  )
  wm_outputs.append(outputs.cpu())
  wm_g_values.append(g_values.cpu())
  wm_eos_masks.append(eos_token_mask.cpu())

  del outputs, scores, inputs, prompts, eos_token_mask, g_values

del model, wm_outputs, wm_g_values, wm_eos_masks
gc.collect()

final_perplexity = torch.exp(
    torch.Tensor(np.sum(perplexities)) / (BATCH_SIZE * NUM_BATCHES)
)
print(f"Perplexity of watermarked model: {final_perplexity}")


exit()


NUM_NEGATIVES = 10000
POS_BATCH_SIZE = 32
NUM_POS_BATCHES = 313
NEG_BATCH_SIZE = 32
POS_TRUNCATION_LENGTH = 200
NEG_TRUNCATION_LENGTH = 200
MAX_PADDED_LENGTH = 1000
TEMPERATURE = 1.0



def generate_responses(example_inputs, enable_watermarking):
  inputs = tokenizer(
      example_inputs,
      return_tensors='pt',
      padding=True,
  ).to(DEVICE)

  gc.collect()
  torch.cuda.empty_cache()

  model = load_model(
      MODEL_NAME,
      expected_device=DEVICE,
      enable_watermarking=enable_watermarking,
  )
  torch.manual_seed(0)
  _, inputs_len = inputs['input_ids'].shape

  outputs = model.generate(
      **inputs,
      do_sample=True,
      max_length=inputs_len + OUTPUTS_LEN,
      temperature=TEMPERATURE,
      top_k=TOP_K,
      top_p=TOP_P,
  )

  outputs = outputs[:, inputs_len:]

  eos_token_mask = logits_processor.compute_eos_token_mask(
      input_ids=outputs,
      eos_token_id=tokenizer.eos_token_id,
  )[:, CONFIG['ngram_len'] - 1 :]

  context_repetition_mask = logits_processor.compute_context_repetition_mask(
      input_ids=outputs,
  )

  combined_mask = context_repetition_mask * eos_token_mask

  g_values = logits_processor.compute_g_values(
      input_ids=outputs,
  )

  return g_values, combined_mask


example_inputs = [
    'I enjoy walking with my cute dog',
    'I am from New York',
    'The test was not so very hard after all',
    "I don't think they can score twice in so short a time",
]

wm_g_values, wm_mask = generate_responses(
    example_inputs, enable_watermarking=True
)
uwm_g_values, uwm_mask = generate_responses(
    example_inputs, enable_watermarking=False
)




wm_mean_scores = detector_mean.mean_score(
    wm_g_values.cpu().numpy(), wm_mask.cpu().numpy()
)
uwm_mean_scores = detector_mean.mean_score(
    uwm_g_values.cpu().numpy(), uwm_mask.cpu().numpy()
)

print('Mean scores for watermarked responses: ', wm_mean_scores)
print('Mean scores for unwatermarked responses: ', uwm_mean_scores)


wm_weighted_mean_scores = detector_mean.weighted_mean_score(
    wm_g_values.cpu().numpy(), wm_mask.cpu().numpy()
)
uwm_weighted_mean_scores = detector_mean.weighted_mean_score(
    uwm_g_values.cpu().numpy(), uwm_mask.cpu().numpy()
)

print(
    'Weighted Mean scores for watermarked responses: ', wm_weighted_mean_scores
)
print(
    'Weighted Mean scores for unwatermarked responses: ',
    uwm_weighted_mean_scores,
)



gc.collect()
torch.cuda.empty_cache()

model = load_model(MODEL_NAME, expected_device=DEVICE, enable_watermarking=True)
torch.manual_seed(0)

eli5_prompts = datasets.load_dataset("Pavithree/eli5")

wm_outputs = []

for batch_id in tqdm.tqdm(range(NUM_POS_BATCHES)):
  prompts = eli5_prompts['train']['title'][
      batch_id * POS_BATCH_SIZE:(batch_id + 1) * POS_BATCH_SIZE]
  prompts = [_process_raw_prompt(prompt.encode()) for prompt in prompts]
  inputs = tokenizer(
      prompts,
      return_tensors='pt',
      padding=True,
  ).to(DEVICE)
  _, inputs_len = inputs['input_ids'].shape

  outputs = model.generate(
      **inputs,
      do_sample=True,
      max_length=inputs_len + OUTPUTS_LEN,
      temperature=TEMPERATURE,
      top_k=TOP_K,
      top_p=TOP_P,
  )

  wm_outputs.append(outputs[:, inputs_len:])

  del outputs, inputs, prompts

del model
gc.collect()
torch.cuda.empty_cache()


dataset, info = tfds.load('wikipedia/20230601.en', split='train', with_info=True)

dataset = dataset.take(10000)

df = tfds.as_dataframe(dataset, info)
ds = tf.data.Dataset.from_tensor_slices(dict(df))
tf.random.set_seed(0)
ds = ds.shuffle(buffer_size=10_000)
ds = ds.batch(batch_size=1)

tokenized_uwm_outputs = []
lengths = []
batched = []
padded_length = 2500
for i, batch in tqdm.tqdm(enumerate(ds)):
  responses = [val.decode() for val in batch['text'].numpy()]
  inputs = tokenizer(
      responses,
      return_tensors='pt',
      padding=True,
  ).to(DEVICE)
  line = inputs['input_ids'].cpu().numpy()[0].tolist()
  if len(line) >= padded_length:
    line = line[:padded_length]
  else:
    line = line + [
        tokenizer.eos_token_id for _ in range(padded_length - len(line))
    ]
  batched.append(torch.tensor(line, dtype=torch.long, device=DEVICE)[None, :])
  if len(batched) == NEG_BATCH_SIZE:
    tokenized_uwm_outputs.append(torch.cat(batched, dim=0))
    batched = []
  if i > NUM_NEGATIVES:
    break

bayesian_detector, test_loss = (
    detector_bayesian.BayesianDetector.train_best_detector(
        tokenized_wm_outputs=wm_outputs,
        tokenized_uwm_outputs=tokenized_uwm_outputs,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        torch_device=DEVICE,
        max_padded_length=MAX_PADDED_LENGTH,
        pos_truncation_length=POS_TRUNCATION_LENGTH,
        neg_truncation_length=NEG_TRUNCATION_LENGTH,
        verbose=True,
        learning_rate=3e-3,
        n_epochs=100,
        l2_weights=np.zeros((1,)),
    )
)



wm_bayesian_scores = bayesian_detector.score(
    wm_g_values.cpu().numpy(), wm_mask.cpu().numpy()
)
uwm_bayesian_scores = bayesian_detector.score(
    uwm_g_values.cpu().numpy(), uwm_mask.cpu().numpy()
)

print('Bayesian scores for watermarked responses: ', wm_bayesian_scores)
print('Bayesian scores for unwatermarked responses: ', uwm_bayesian_scores)


