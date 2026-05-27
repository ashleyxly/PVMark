# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Logit processor for supporting watermarking in HF model."""

from collections import Counter, deque
from collections.abc import Sequence

import torch
import transformers
import logging
# logging.basicConfig(level=logging.DEBUG, filename="logits_processing.log")
_LOGGER = logging.getLogger(__name__)
_LOGGER.debug("logits_processing This is a debug message")

from synthid_text import hashing_function
from synthid_text import gpu_hash

import functools
from concurrent.futures import ThreadPoolExecutor
import hash_rustlib
import numpy as np
from typing import Optional

RUST_LIB = True
RUST_FUSED_G_VALUES = False
RUST_FUSED_DETECT_G_VALUES = False
RUST_FAST_CONTEXT_MASK = False
GPU_HASH_BACKEND = False
GPU_FUSED_SCORE_UPDATE = False
GPU_FUSED_HISTORY_UPDATE = False
CUDA_CPP_ONLINE_WET = False
CPU_UPDATE_SCORES = False
COMPILE_UPDATE_SCORES = False
DEBUG_PRINT = False
DEBUG_PRINT_LOGITS = False
DETECT_DEBUG_PRINT = False
IS_LCG = False
HASH_TYPE = 4

@functools.lru_cache(maxsize=1000)
def compute_keys_use_LCG_from_rustlib(n_minus_1_grams, indices, keys):
  (hash_result, hash_result_with_just_context) = hash_rustlib._compute_keys_use_LCG(n_minus_1_grams, indices, keys)
  return (hash_result, hash_result_with_just_context)

@functools.lru_cache(maxsize=1000)
def invoke_sample_g_values_use_LCG_from_rustlib(ngrams_keys, field_prime):
  g_value = hash_rustlib._sample_g_values_use_LCG(ngrams_keys, field_prime)
  return g_value

@functools.lru_cache(maxsize=1000)
def compute_ngram_keys_use_LCG_from_rustlib(ngrams_keys, keys):
  hash_result = hash_rustlib._compute_ngram_keys_use_LCG(ngrams_keys, keys)
  return hash_result

@functools.lru_cache(maxsize=1000)
def compute_g_values_use_LCG_from_rustlib(n_minus_1_grams, indices, keys, field_prime):
  return hash_rustlib._compute_g_values_use_LCG(
      n_minus_1_grams, indices, keys, field_prime
  )


@functools.lru_cache(maxsize=1000)
def compute_keys_use_hash_from_rustlib(n_minus_1_grams, indices, keys):
  (hash_result, hash_result_with_just_context) = hash_rustlib._compute_keys_use_hash(n_minus_1_grams, indices, keys, HASH_TYPE)
  return (hash_result, hash_result_with_just_context)

@functools.lru_cache(maxsize=1000)
def invoke_sample_g_values_use_hash_from_rustlib(ngrams_keys, field_prime):
  g_value = hash_rustlib._sample_g_values_use_hash(ngrams_keys, field_prime, HASH_TYPE)
  return g_value

@functools.lru_cache(maxsize=1000)
def compute_ngram_keys_use_hash_from_rustlib(ngrams_keys, keys):
  hash_result = hash_rustlib._compute_ngram_keys_use_hash(ngrams_keys, keys, HASH_TYPE)
  return hash_result

@functools.lru_cache(maxsize=1000)
def compute_g_values_use_hash_from_rustlib(n_minus_1_grams, indices, keys, field_prime):
  return hash_rustlib._compute_g_values_use_hash(
      n_minus_1_grams, indices, keys, field_prime, HASH_TYPE
  )

def _decode_flat_g_values(flat_result):
  g_values, context_hash, batch_size, candidate_size, depth = flat_result
  g_values = np.frombuffer(g_values, dtype=np.uint8).reshape(
      batch_size, candidate_size, depth
  )
  return g_values, context_hash


def _decode_flat_detect_g_values(flat_result):
  g_values, batch_size, num_ngrams, depth = flat_result
  g_values = np.frombuffer(g_values, dtype=np.uint8).reshape(
      batch_size, num_ngrams, depth
  )
  return g_values


def _as_numpy_int64_contiguous(tensor: torch.Tensor) -> np.ndarray:
  if isinstance(tensor, np.ndarray):
    return np.ascontiguousarray(tensor, dtype=np.int64)
  return np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.int64)


def _context_history_key(hash_value) -> str:
  """Normalize Rust context hashes without parsing 254-bit decimal strings."""
  if isinstance(hash_value, str):
    return hash_value
  if isinstance(hash_value, bytes):
    return hash_value.decode("ascii")
  if isinstance(hash_value, np.generic):
    return str(hash_value.item())
  return str(hash_value)

@functools.lru_cache(maxsize=1000)
def compute_g_values_use_poseidon_fast_from_rustlib(
    n_minus_1_grams, indices, keys
):
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_poseidon_fast_flat(
      n_minus_1_grams, indices, keys
  ))

@functools.lru_cache(maxsize=1000)
def compute_g_values_use_poseidon2_fast_from_rustlib(
    n_minus_1_grams, indices, keys
):
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_poseidon2_fast_flat(
      n_minus_1_grams, indices, keys
  ))

@functools.lru_cache(maxsize=1000)
def compute_g_values_use_mimc_fast_from_rustlib(
    n_minus_1_grams, indices, keys
):
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_mimc_fast_flat(
      n_minus_1_grams, indices, keys
  ))


def compute_g_values_use_poseidon_fast_from_rustlib_buffer(
    n_minus_1_grams: torch.Tensor, indices: torch.Tensor, keys
):
  ngrams_np = _as_numpy_int64_contiguous(n_minus_1_grams)
  indices_np = _as_numpy_int64_contiguous(indices)
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_poseidon_fast_flat_i64(
      ngrams_np,
      indices_np,
      keys,
      ngrams_np.shape[0],
      ngrams_np.shape[1],
      indices_np.shape[1],
  ))


def compute_g_values_use_poseidon2_fast_from_rustlib_buffer(
    n_minus_1_grams: torch.Tensor, indices: torch.Tensor, keys
):
  ngrams_np = _as_numpy_int64_contiguous(n_minus_1_grams)
  indices_np = _as_numpy_int64_contiguous(indices)
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_poseidon2_fast_flat_i64(
      ngrams_np,
      indices_np,
      keys,
      ngrams_np.shape[0],
      ngrams_np.shape[1],
      indices_np.shape[1],
  ))


def compute_g_values_use_mimc_fast_from_rustlib_buffer(
    n_minus_1_grams: torch.Tensor, indices: torch.Tensor, keys
):
  ngrams_np = _as_numpy_int64_contiguous(n_minus_1_grams)
  indices_np = _as_numpy_int64_contiguous(indices)
  return _decode_flat_g_values(hash_rustlib._compute_g_values_use_mimc_fast_flat_i64(
      ngrams_np,
      indices_np,
      keys,
      ngrams_np.shape[0],
      ngrams_np.shape[1],
      indices_np.shape[1],
  ))


def compute_g_values_use_mimc_gpu_buffer(
    n_minus_1_grams: torch.Tensor,
    indices: torch.Tensor,
    keys,
    dtype: Optional[torch.dtype] = None,
    return_context_tensor: bool = False,
):
  return gpu_hash.compute_g_values_use_mimc_gpu(
      n_minus_1_grams,
      indices,
      keys,
      dtype=dtype,
      return_context_tensor=return_context_tensor,
  )


def compute_detect_g_values_use_poseidon_fast_from_rustlib_buffer(
    input_ids: torch.Tensor, keys, ngram_len: int
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return _decode_flat_detect_g_values(
      hash_rustlib._compute_detect_g_values_use_poseidon_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
      )
  )


def compute_detect_g_values_use_poseidon2_fast_from_rustlib_buffer(
    input_ids: torch.Tensor, keys, ngram_len: int
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return _decode_flat_detect_g_values(
      hash_rustlib._compute_detect_g_values_use_poseidon2_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
      )
  )


def compute_detect_g_values_use_mimc_fast_from_rustlib_buffer(
    input_ids: torch.Tensor, keys, ngram_len: int
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return _decode_flat_detect_g_values(
      hash_rustlib._compute_detect_g_values_use_mimc_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
      )
  )


def compute_context_hashes_lcg_from_rustlib_buffer(
    input_ids: torch.Tensor, context_len: int
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return hash_rustlib._compute_context_hashes_lcg_flat_i64(
      input_np,
      input_np.shape[0],
      input_np.shape[1],
      context_len,
  )


def compute_context_repetition_mask_lcg_from_rustlib_buffer(
    input_ids: torch.Tensor, context_len: int, context_history_size: int
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  mask_bytes, batch_size, num_contexts = (
      hash_rustlib._compute_context_repetition_mask_lcg_flat_i64(
          input_np,
          input_np.shape[0],
          input_np.shape[1],
          context_len,
          context_history_size,
      )
  )
  mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(
      batch_size, num_contexts
  )
  return mask.astype(np.bool_, copy=False)


def compute_weighted_mean_score_use_poseidon_fast_from_rustlib_buffer(
    input_ids: torch.Tensor,
    keys,
    ngram_len: int,
    context_history_size: int,
    eos_token_id: int,
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return np.asarray(
      hash_rustlib._compute_weighted_mean_score_use_poseidon_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
          context_history_size,
          int(eos_token_id),
      ),
      dtype=np.float32,
  )


def compute_weighted_mean_score_use_poseidon2_fast_from_rustlib_buffer(
    input_ids: torch.Tensor,
    keys,
    ngram_len: int,
    context_history_size: int,
    eos_token_id: int,
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return np.asarray(
      hash_rustlib._compute_weighted_mean_score_use_poseidon2_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
          context_history_size,
          int(eos_token_id),
      ),
      dtype=np.float32,
  )


def compute_weighted_mean_score_use_mimc_fast_from_rustlib_buffer(
    input_ids: torch.Tensor,
    keys,
    ngram_len: int,
    context_history_size: int,
    eos_token_id: int,
):
  input_np = _as_numpy_int64_contiguous(input_ids)
  return np.asarray(
      hash_rustlib._compute_weighted_mean_score_use_mimc_fast_flat_i64(
          input_np,
          keys,
          input_np.shape[0],
          input_np.shape[1],
          ngram_len,
          context_history_size,
          int(eos_token_id),
      ),
      dtype=np.float32,
  )




def list_to_tuple_3d(list_3d):
  return tuple(tuple(tuple(inner) for inner in outer) for outer in list_3d)
      

def update_scores(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
) -> torch.FloatTensor:
  """Updates scores using the g values.

  We assume that the scores are in the log space.
  Args:
    scores: Scores (batch_size, vocab_size).
    g_values: G values (batch_size, vocab_size, depth).

  Returns:
    Updated scores (batch_size, vocab_size).
  """
  _, _, depth = g_values.shape
  device = scores.device
  
  # print("depth", depth)
  _LOGGER.debug("depth: %s", depth)

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("g_values_at_depth: %s", g_values_at_depth)
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("g_mass_at_depth: %s", g_mass_at_depth)
    probs.mul_(1 + g_values_at_depth - g_mass_at_depth)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("probs: %s", probs)

  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


def update_scores_cpu(
    scores: torch.FloatTensor,
    g_values,
) -> torch.FloatTensor:
  """Updates tiny top-k scores on CPU to avoid many small CUDA kernels."""
  scores_np = scores.detach().cpu().numpy().astype(np.float64, copy=False)
  if hasattr(g_values, "detach"):
    g_values_np = g_values.detach().cpu().numpy().astype(np.float64, copy=False)
  else:
    g_values_np = np.asarray(g_values, dtype=np.float64)

  shifted_scores = scores_np - np.max(scores_np, axis=1, keepdims=True)
  probs = np.exp(shifted_scores)
  probs /= np.sum(probs, axis=1, keepdims=True)

  for i in range(g_values_np.shape[-1]):
    g_values_at_depth = g_values_np[:, :, i]
    g_mass_at_depth = np.sum(
        g_values_at_depth * probs, axis=1, keepdims=True
    )
    probs *= 1 + g_values_at_depth - g_mass_at_depth

  with np.errstate(divide="ignore", invalid="ignore"):
    log_probs = np.log(probs)
  log_probs = np.where(np.isfinite(log_probs), log_probs, -1e12)
  return torch.from_numpy(log_probs).to(device=scores.device, dtype=scores.dtype)


def update_scores_no_logging(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
) -> torch.FloatTensor:
  """Log-free update_scores variant suitable for torch.compile."""
  _, _, depth = g_values.shape
  device = scores.device
  probs = torch.softmax(scores, dim=1)
  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    probs.mul_(1 + g_values_at_depth - g_mass_at_depth)
  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


def update_scores_distortionary(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
    num_leaves: int,
) -> torch.FloatTensor:
  """Update scores using the g values for distortionary tournament watermarking.

  We assume that the scores are in the log space.
  Args:
    scores: Scores (batch_size, vocab_size).
    g_values: G values (batch_size, vocab_size, depth).
    num_leaves: Number of leaves per node in the tournament tree.

  Returns:
    Updated scores (batch_size, vocab_size).
  """
  _, _, depth = g_values.shape
  device = scores.device

  probs = torch.softmax(scores, dim=1)

  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    coeff_not_in_g = (1 - g_mass_at_depth)**(num_leaves - 1)
    coeff_in_g = (1 - (1 - g_mass_at_depth)**(num_leaves)) / g_mass_at_depth
    coeffs = torch.where(
        torch.logical_and(g_values_at_depth == 1, probs > 0),
        coeff_in_g, coeff_not_in_g)
    probs.mul_(coeffs)

  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


def update_scores_distortionary_cpu(
    scores: torch.FloatTensor,
    g_values,
    num_leaves: int,
) -> torch.FloatTensor:
  """CPU implementation of distortionary score updates for tiny top-k tensors."""
  scores_np = scores.detach().cpu().numpy().astype(np.float64, copy=False)
  if hasattr(g_values, "detach"):
    g_values_np = g_values.detach().cpu().numpy().astype(np.float64, copy=False)
  else:
    g_values_np = np.asarray(g_values, dtype=np.float64)

  shifted_scores = scores_np - np.max(scores_np, axis=1, keepdims=True)
  probs = np.exp(shifted_scores)
  probs /= np.sum(probs, axis=1, keepdims=True)

  for i in range(g_values_np.shape[-1]):
    g_values_at_depth = g_values_np[:, :, i]
    g_mass_at_depth = np.sum(
        g_values_at_depth * probs, axis=1, keepdims=True
    )
    coeff_not_in_g = (1 - g_mass_at_depth) ** (num_leaves - 1)
    coeff_in_g = (
        1 - (1 - g_mass_at_depth) ** num_leaves
    ) / g_mass_at_depth
    coeffs = np.where(
        np.logical_and(g_values_at_depth == 1, probs > 0),
        coeff_in_g,
        coeff_not_in_g,
    )
    probs *= coeffs

  with np.errstate(divide="ignore", invalid="ignore"):
    log_probs = np.log(probs)
  log_probs = np.where(np.isfinite(log_probs), log_probs, -1e12)
  return torch.from_numpy(log_probs).to(device=scores.device, dtype=scores.dtype)


def update_scores_distortionary_no_logging(
    scores: torch.FloatTensor,
    g_values: torch.FloatTensor,
    num_leaves: int,
) -> torch.FloatTensor:
  """Log-free distortionary update variant suitable for torch.compile."""
  _, _, depth = g_values.shape
  device = scores.device
  probs = torch.softmax(scores, dim=1)
  for i in range(depth):
    g_values_at_depth = g_values[:, :, i]
    g_mass_at_depth = (g_values_at_depth * probs).sum(axis=1, keepdims=True)
    coeff_not_in_g = (1 - g_mass_at_depth)**(num_leaves - 1)
    coeff_in_g = (1 - (1 - g_mass_at_depth)**(num_leaves)) / g_mass_at_depth
    coeffs = torch.where(
        torch.logical_and(g_values_at_depth == 1, probs > 0),
        coeff_in_g, coeff_not_in_g)
    probs.mul_(coeffs)
  log_probs = torch.log(probs)
  log_probs = torch.where(
      torch.isfinite(log_probs), log_probs, torch.tensor(-1e12, device=device)
  )
  return log_probs


try:
  _compiled_update_scores = torch.compile(
      update_scores_no_logging,
      mode="reduce-overhead",
      fullgraph=True,
  )
  _compiled_update_scores_distortionary = torch.compile(
      update_scores_distortionary_no_logging,
      mode="reduce-overhead",
      fullgraph=True,
  )
except Exception:
  _compiled_update_scores = update_scores_no_logging
  _compiled_update_scores_distortionary = update_scores_distortionary_no_logging


class SynthIDState:
  """SynthID watermarking state."""

  def __init__(
      self,
      batch_size: int,
      ngram_len: int,
      context_history_size: int,
      device: torch.device,
  ):
    """Initializes the state.

    Args:
      batch_size: Batch size.
      ngram_len: Ngram length.
      context_history_size: Size of the tensor to keep track of seen contexts.
      device: Device to use.
    """
    self.context = torch.zeros(
        (batch_size, ngram_len - 1),
        dtype=torch.int64,
        device=device,
    )
    self.context_history = torch.zeros(
        (batch_size, context_history_size),
        dtype=torch.int64,
        device=device,
    )
    self.context_history_python = [
        deque(["0"] * context_history_size, maxlen=context_history_size)
        for _ in range(batch_size)
    ]
    self.context_history_counts = [
        Counter({"0": context_history_size})
        for _ in range(batch_size)
    ]
    self.context_cpu = np.zeros(
        (batch_size, ngram_len - 1),
        dtype=np.int64,
    )
    self.context_history_gpu_limbs = torch.zeros(
        (batch_size, context_history_size, gpu_hash.LIMBS),
        dtype=torch.int32,
        device=device,
    )
    self.repeated_flags_gpu_zero = torch.zeros(
        (batch_size,),
        dtype=torch.uint8,
        device=device,
    )
    self.context_history_gpu_index = 0
    self.num_calls = 0


class SynthIDLogitsProcessor(transformers.LogitsProcessor):
  """SynthID watermarking logits processor.

  Logits processor updates the provided scores based on the binary g values
  assigned to each possible ngram and watermarking key combination hashed into
  an int64 keys.

  A random sampling table is pre-computed and modulo table size is applied to
  map from ngram keys (int64) to g values.
  """

  def __init__(
      self,
      *,
      ngram_len: int,
      keys: Sequence[int],
      sampling_table_size: int,
      sampling_table_seed: int,
      context_history_size: int,
      temperature: float,
      top_k: int,
      device: torch.device,
      skip_first_ngram_calls: bool = False,
      apply_top_k: bool = True,
      num_leaves: int = 2
  ):
    """Initializes the logits processor.

    Args:
      ngram_len: Ngram length.
      keys: A sequence of watermarking keys, one for each depth.
      sampling_table_size: Size of the sampling table.
      sampling_table_seed: Random seed to generate the sampling table.
      context_history_size: Size of the tensor to keep track of seen contexts.
      temperature: Temperature to use for scaling the scores.
      top_k: Top k to use for sampling the scores.
      device: Device to use.
      skip_first_ngram_calls: Whether to skip first ngram calls.
      apply_top_k: Whether to apply top k to the scores.
      num_leaves: Number of leaves per node in the tournament tree.
    """
    self.ngram_len = ngram_len
    self.keys = torch.tensor(keys, device=device)
    self._rust_keys = tuple(int(key) for key in keys)
    self._rust_context_seed = "1"
    self._rust_field_prime = (
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    )

    generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
    # A random sampling table is pre-computed and modulo table size is applied
    # to map from a hash of ngram keys to g values, this is similar to the
    # hashtable implementation used in
    # https://github.com/facebookresearch/three_bricks. We note that the
    # hashing employed in this repository is different from that used to
    # watermark the Gemini App, and hence the detectors trained based on the
    # hashing in this repository will not transfer to text generated by
    # the Gemini App.
    self.sampling_table = torch.randint(
        low=0,
        high=2,
        size=(sampling_table_size,),
        generator=generator,
        device=device,
    )
    self.context_history_size = context_history_size
    self.device = device
    self.state = None
    self.skip_first_ngram_calls = skip_first_ngram_calls
    self.apply_top_k = apply_top_k

    # Check validity of temperature.
    if not (isinstance(temperature, float) and temperature > 0):
      except_msg = (
          f"`temperature` (={temperature}) has to be a strictly positive float,"
          " otherwise your next token scores will be invalid."
      )
      if isinstance(temperature, float) and temperature == 0.0:
        except_msg += (
            " If you're looking for greedy decoding strategies, set"
            " `do_sample=False`."
        )
      raise ValueError(except_msg)

    self.temperature = temperature

    self._num_leaves = num_leaves

    # Check validity of top_k.
    if not (isinstance(top_k, int) and top_k > 1):
      raise ValueError(f"`top_k` has to be > 1, but is {top_k}")

    self.top_k = top_k

  def _init_state(self, batch_size: int):
    """Initializes the state."""
    self.state = SynthIDState(
        batch_size=batch_size,
        ngram_len=self.ngram_len,
        context_history_size=self.context_history_size,
        device=self.device,
    )

  @torch.no_grad
  def __call__(
      self,
      input_ids: torch.LongTensor,
      scores: torch.FloatTensor,
  ) -> tuple[torch.FloatTensor, torch.LongTensor]:
    raise NotImplementedError(
        "__call__ is not implemented for watermarking logits processor."
    )

  @torch.no_grad
  def watermarked_call(
      self,
      input_ids: torch.LongTensor,
      scores: torch.FloatTensor,
  ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
    """Calls the logits processor statefully.

    This function computes top_k internally and returns the indices mapping
    from top_k scores to dense scores.

    Args:
      input_ids: Input token ids (batch_size, inputs_len).
      scores: Scores (batch_size, vocab_size).

    Returns:
      Tuple of
        Watermarked updated scores (batch_size, top_k)
        Top k indices (batch_size, top_k).
        original scores for perplexity calculations (batch_size, top_k)
    """
    if DEBUG_PRINT_LOGITS:
      print("shape of input_ids -> ", input_ids.shape)
      print("shape of scores -> ", scores.shape)
      print("scores -> ", scores)
    self._check_input_ids_shape(input_ids)
    scores_processed = scores / self.temperature
    top_k_result = torch.topk(scores_processed, k=self.top_k, dim=1)
    batch_size, vocab_size = scores.shape

    if self.apply_top_k:
      scores_top_k = top_k_result.values
      # scores_top_k shape [batch_size, top_k]
      top_k_indices = top_k_result.indices
      # top_k_indices shape [batch_size, top_k]
    else:
      scores_top_k = scores_processed
      top_k_indices = torch.stack([
          torch.arange(vocab_size, device=self.device)
          for _ in range(batch_size)
      ])
    # print("scores_top_k", scores_top_k)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("scores_top_k: %s", scores_top_k)
    # print("score_top_k shape -> ", scores_top_k.shape)

    device = scores.device
    if str(device) != str(self.device):
      raise ValueError(
          "SynthIDLogitsProcessor received inputs with unexpected device.",
      )

    state_was_none = self.state is None
    if state_was_none:
      # Initialize watermarking state if it does not exist.
      self._init_state(batch_size)
    else:
      # Append last input id (which is the input id added in last call) to the
      # previous context so we have the context to be used for current
      # watermarking.
      self.state.context = torch.concat(
          (self.state.context, input_ids[:, -1:]),
          dim=1,
      )
      self.state.context = self.state.context[:, 1:]

    assert self.state is not None
    if RUST_LIB and RUST_FUSED_G_VALUES and not GPU_HASH_BACKEND and not state_was_none:
      last_token_cpu = input_ids[:, -1].detach().cpu().numpy().astype(
          np.int64,
          copy=False,
      )
      self.state.context_cpu[:, :-1] = self.state.context_cpu[:, 1:]
      self.state.context_cpu[:, -1] = last_token_cpu

    self.state.num_calls += 1

    # Don't watermark the first ngram_len - 1 tokens if set.
    if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
      return scores_top_k, top_k_indices, scores_top_k

    
    # print("self.state.context", self.state.context)
    # print("top_k_indices", top_k_indices)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("self.state.context: %s", self.state.context)
      _LOGGER.debug("top_k_indices: %s", top_k_indices)
    # print("self.state.context shape -> ", self.state.context.shape)
    # print("top_k_indices shape -> ", top_k_indices.shape)
    
    
    use_mimc_gpu_wet = (
        RUST_LIB
        and RUST_FUSED_G_VALUES
        and GPU_HASH_BACKEND
        and HASH_TYPE == 5
        and gpu_hash.is_available()
        and self._num_leaves == 2
        and scores_top_k.dtype == torch.float32
    )
    use_gpu_score_update = (
        GPU_FUSED_SCORE_UPDATE
        and gpu_hash.is_available()
        and scores_top_k.is_cuda
        and scores_top_k.dtype == torch.float32
        and use_mimc_gpu_wet
    )

    # 2. Generate random keys and g values for each ngram key combination.
    if use_mimc_gpu_wet:
      g_values = None
      hash_result_with_just_context = None
      ngram_keys = None
    elif RUST_LIB and RUST_FUSED_G_VALUES:
      g_values, hash_result_with_just_context = self._compute_g_values(
          self.state.context_cpu
          if not GPU_HASH_BACKEND
          else self.state.context,
          top_k_indices,
          None if (use_gpu_score_update or CPU_UPDATE_SCORES) else scores_top_k.dtype,
      )
      ngram_keys = None
    else:
      ngram_keys, hash_result_with_just_context = self._compute_keys(
          self.state.context, top_k_indices
      )
    # ngram_keys shape [batch_size, top_k, depth]
    # print("ngram_keys", ngram_keys)
    # print("hash_result_with_just_context", hash_result_with_just_context)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("ngram_keys: %s", ngram_keys)
      _LOGGER.debug(
          "hash_result_with_just_context: %s", hash_result_with_just_context
      )
    # print("type of ngram_keys -> ", type(ngram_keys))
    # print("type of hash_result_with_just_context -> ", type(hash_result_with_just_context))
    # print("len of ngram_keys -> ", len(ngram_keys))
    # print("len of hash_result_with_just_context -> ", len(hash_result_with_just_context))
    # print("len of ngram_keys[0] -> ", len(ngram_keys[0]))
    # print("ngram_keys shape -> ", ngram_keys.shape)
    # print("hash_result_with_just_context shape -> ", hash_result_with_just_context.shape)

    
    # 3. Sample g values.
    if not (RUST_LIB and RUST_FUSED_G_VALUES):
      g_values = self.sample_g_values(ngram_keys)
    # g_values shape [batch_size, top_k, depth]
    
    if DEBUG_PRINT:
      print("g_values shape -> ", g_values.shape)
      # 先将[batch_size, top_k, depth]形状的张量通过切片操作取到[batch_size, top_k]部分，并展平为一维张量
      flat_g_values = g_values[0, 0, :].flatten()

      # 统计元素为1的个数
      count_ones = torch.sum(flat_g_values == 1).item()
      # 统计元素为0的个数
      count_zeros = torch.sum(flat_g_values == 0).item()

      print(f"元素为1的个数: {count_ones}")
      print(f"元素为0的个数: {count_zeros}")
    # print("g_values", g_values)
    if _LOGGER.isEnabledFor(logging.DEBUG):
      _LOGGER.debug("g_values: %s", g_values)
    # print("g_values shape -> ", g_values.shape)

    # print("num_leaves", self._num_leaves)
    _LOGGER.debug("num_leaves: %s", self._num_leaves)

    # 4. Modify scores.
    if use_mimc_gpu_wet:
      if CUDA_CPP_ONLINE_WET and GPU_FUSED_SCORE_UPDATE:
        from synthid_text import cuda_hash_cpp

        updated_scores = cuda_hash_cpp.compute_online_updated_scores_use_mimc_cpp(
            self.state.context,
            top_k_indices,
            scores_top_k,
            self.state.context_history_gpu_limbs,
            self.state.context_history_gpu_index,
            self._rust_keys,
        )
        self.state.context_history_gpu_index = (
            self.state.context_history_gpu_index + 1
        ) % self.context_history_size
        return updated_scores, top_k_indices, scores_top_k

      if GPU_FUSED_HISTORY_UPDATE:
        g_values, repeated_flags_gpu = (
            gpu_hash.compute_g_values_and_repetition_use_mimc_gpu(
                self.state.context,
                top_k_indices,
                self._rust_keys,
                self.state.context_history_gpu_limbs,
                self.state.context_history_gpu_index,
                dtype=None if GPU_FUSED_SCORE_UPDATE else scores_top_k.dtype,
            )
        )
        self.state.context_history_gpu_index = (
            self.state.context_history_gpu_index + 1
        ) % self.context_history_size
      else:
        g_values, repeated_flags_gpu = (
            compute_g_values_use_mimc_gpu_buffer(
              self.state.context,
              top_k_indices,
              self._rust_keys,
              dtype=None if GPU_FUSED_SCORE_UPDATE else scores_top_k.dtype,
              return_context_tensor=True,
            )
        )
        repeated_flags_gpu = gpu_hash.update_context_history_gpu(
            self.state.context_history_gpu_limbs,
            repeated_flags_gpu,
            self.state.context_history_gpu_index,
        )
        self.state.context_history_gpu_index = (
            self.state.context_history_gpu_index + 1
        ) % self.context_history_size
      if GPU_FUSED_SCORE_UPDATE:
        updated_scores = gpu_hash.update_scores_gpu(
            scores_top_k,
            g_values,
            repeated_flags_gpu,
        )
        return updated_scores, top_k_indices, scores_top_k

      if self._num_leaves == 2:
        if COMPILE_UPDATE_SCORES:
          updated_scores = _compiled_update_scores(scores_top_k, g_values)
        else:
          updated_scores = update_scores(scores_top_k, g_values)
      else:
        if COMPILE_UPDATE_SCORES:
          updated_scores = _compiled_update_scores_distortionary(
              scores_top_k, g_values, self._num_leaves
          )
        else:
          updated_scores = update_scores_distortionary(
              scores_top_k, g_values, self._num_leaves
          )
      is_repeated_context = repeated_flags_gpu.to(dtype=torch.bool)[:, None]
      if not is_repeated_context.any():
        return updated_scores, top_k_indices, scores_top_k
      if is_repeated_context.all():
        return scores_top_k, top_k_indices, scores_top_k
      updated_watermarked_scores = torch.where(
          is_repeated_context,
          input=scores_top_k,
          other=updated_scores,
      )
      return updated_watermarked_scores, top_k_indices, scores_top_k

    if self._num_leaves == 2:
      if use_gpu_score_update:
        if not torch.is_tensor(g_values):
          g_values = torch.as_tensor(g_values, device=device, dtype=torch.uint8)
        updated_scores = gpu_hash.update_scores_gpu(
            scores_top_k,
            g_values,
            self.state.repeated_flags_gpu_zero,
        )
      elif COMPILE_UPDATE_SCORES:
        updated_scores = _compiled_update_scores(scores_top_k, g_values)
      elif CPU_UPDATE_SCORES:
        updated_scores = update_scores_cpu(scores_top_k, g_values)
      else:
        updated_scores = update_scores(scores_top_k, g_values)
    else:
      if COMPILE_UPDATE_SCORES:
        updated_scores = _compiled_update_scores_distortionary(
            scores_top_k, g_values, self._num_leaves
        )
      elif CPU_UPDATE_SCORES:
        updated_scores = update_scores_distortionary_cpu(
            scores_top_k, g_values, self._num_leaves
        )
      else:
        updated_scores = update_scores_distortionary(
            scores_top_k, g_values, self._num_leaves
        )
    # updated scores shape [batch_size, top_k]
    # print("updated_scores shape -> ", updated_scores.shape)

    # 5. Check if the current watermarking context was previously used, if
    # yes skip watermarking.
    if (
        RUST_LIB
        and GPU_HASH_BACKEND
        and torch.is_tensor(hash_result_with_just_context)
    ):
      repeated_flags_gpu = gpu_hash.update_context_history_gpu(
          self.state.context_history_gpu_limbs,
          hash_result_with_just_context,
          self.state.context_history_gpu_index,
      )
      self.state.context_history_gpu_index = (
          self.state.context_history_gpu_index + 1
      ) % self.context_history_size
      is_repeated_context = repeated_flags_gpu.to(dtype=torch.bool)[:, None]
      if not is_repeated_context.any():
        return updated_scores, top_k_indices, scores_top_k
      if is_repeated_context.all():
        return scores_top_k, top_k_indices, scores_top_k
    elif RUST_LIB:
      repeated_flags = []
      for batch, hash_value in enumerate(hash_result_with_just_context):
        hash_int = _context_history_key(hash_value)
        repeated_flags.append(
            self.state.context_history_counts[batch].get(hash_int, 0) > 0
        )
        history = self.state.context_history_python[batch]
        counts = self.state.context_history_counts[batch]
        if len(history) == history.maxlen:
          evicted_hash = history.pop()
          counts[evicted_hash] -= 1
          if counts[evicted_hash] == 0:
            del counts[evicted_hash]
        history.appendleft(hash_int)
        counts[hash_int] += 1
      if not any(repeated_flags):
        return updated_scores, top_k_indices, scores_top_k
      if all(repeated_flags):
        return scores_top_k, top_k_indices, scores_top_k
      is_repeated_context = torch.tensor(
          repeated_flags,
          device=self.device,
          dtype=torch.bool,
      )[:, None]
    else:
      hash_result_with_just_context = hash_result_with_just_context[:, None]
      is_repeated_context = (
          self.state.context_history == hash_result_with_just_context
      ).any(
          dim=1,
          keepdim=True,
      )
      self.state.context_history = torch.concat(
          (hash_result_with_just_context, self.state.context_history),
          dim=1,
      )[:, :-1]

    updated_watermarked_scores = torch.where(
        is_repeated_context,
        input=scores_top_k,
        other=updated_scores,
    )
    return updated_watermarked_scores, top_k_indices, scores_top_k

  def compute_ngram_keys(
      self,
      ngrams: torch.LongTensor,
  ) -> torch.LongTensor:
    """Computes random keys for each ngram and depth.

    Args:
      ngrams: Ngrams (batch_size, num_ngrams, ngram_len).

    Returns:
      ngram keys (batch_size, num_ngrams, depth).
    """
    if len(ngrams.shape) != 3:
      raise ValueError(
          "Ngrams should be of shape (batch_size, num_ngrams, ngram_len), but"
          f" is {ngrams.shape}"
      )
    if ngrams.shape[2] != self.ngram_len:
      raise ValueError(
          "Ngrams should be of shape (batch_size, num_ngrams, ngram_len),"
          f" where ngram_len is {self.ngram_len}, but is {ngrams.shape}"
      )
    
    if RUST_LIB:
      ngrams = list_to_tuple_3d(ngrams.detach().cpu().tolist())
      keys = self._rust_keys
      if IS_LCG:
        hash_result = compute_ngram_keys_use_LCG_from_rustlib(ngrams, keys)
      else:
        hash_result = compute_ngram_keys_use_hash_from_rustlib(ngrams, keys)
      return hash_result
    
    else:
      batch_size, _, _ = ngrams.shape

      hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
      # hash_result shape [batch_size,]
      # ngrams shape [batch_size, num_ngrams, ngram_len]
      hash_result = torch.vmap(
          hashing_function.accumulate_hash, in_dims=(None, 1), out_dims=1
      )(hash_result, ngrams)
      # hash_result shape [batch_size, num_ngrams]

      keys = self.keys[None, None, :, None]
      # hash_result shape [batch_size, num_ngrams]
      # keys shape [1, 1, depth, 1]
      hash_result = torch.vmap(
          hashing_function.accumulate_hash, in_dims=(None, 2), out_dims=2
      )(hash_result, keys)
      # hash_result shape [batch_size, num_ngrams, depth]

      return hash_result

  def _compute_keys(
      self,
      n_minus_1_grams: torch.LongTensor,
      indices: torch.LongTensor,
  ) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Computes random keys for each ngram and depth.

    Args:
      n_minus_1_grams: Ngrams (batch_size, ngram_len - 1).
      indices: indices of the continuations (batch_size, num_indices)

    Returns:
      Ngram keys (batch_size, num_indices, depth).
    """
    
    if RUST_LIB:
      n_minus_1_grams = tuple(tuple(gram) for gram in n_minus_1_grams.detach().cpu().tolist())
      indices = tuple(tuple(index) for index in indices.detach().cpu().tolist())
      keys = self._rust_keys
      if IS_LCG:
        hash_result, hash_result_with_just_context = compute_keys_use_LCG_from_rustlib(n_minus_1_grams, indices, keys)
      else:
        hash_result, hash_result_with_just_context = compute_keys_use_hash_from_rustlib(n_minus_1_grams, indices, keys)
      # return (torch.tensor(hash_result, device=self.device), torch.tensor(hash_result_with_just_context, device=self.device))
      return hash_result, hash_result_with_just_context
    else:
      batch_size, _ = n_minus_1_grams.shape

      hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
      # First hash n_minus_1 gram, for each batch entry we have a single
      # n_minus_1 gram context.
      # hash_result shape [batch_size]
      # n_minus_1_gram shape [batch_size, ngram_len - 1]
      hash_result_with_just_context = hashing_function.accumulate_hash(
          hash_result, n_minus_1_grams
      )
      # hash_result shape [batch_size,]
      # Indices is of shape [batch_size, num_indices], so we make it
      # [batch_size, num_indices, 1] so we can vmap over num_indices dim.
      hash_result = torch.vmap(
          hashing_function.accumulate_hash, in_dims=(None, 1), out_dims=1
      )(hash_result_with_just_context, indices[:, :, None])
      # hash_result shape [batch_size, num_indices]
      # Basically we have a hash for each batch entry and each indices
      # Now we add watermarking keys to this hash.
      # keys are of shape [depth,]
      # We add batch, num_indices and data dimension to this making it
      # [1, 1, depth, 1].
      # So we can vmap over the depth dimension for compute_hash
      keys = self.keys[None, None, :, None]
      hash_result = torch.vmap(
          hashing_function.accumulate_hash, in_dims=(None, 2), out_dims=2
      )(hash_result, keys)
      # hash_result shape should be [batch_size, num_indices, depth]
      return hash_result, hash_result_with_just_context

  def _compute_g_values(
      self,
      n_minus_1_grams: torch.LongTensor,
      indices: torch.LongTensor,
      dtype: Optional[torch.dtype] = None,
  ) -> tuple[torch.LongTensor, torch.LongTensor]:
    """Computes g values and context hashes through the fused Rust helper."""
    if not RUST_LIB:
      ngram_keys, hash_result_with_just_context = self._compute_keys(
          n_minus_1_grams, indices
      )
      return self.sample_g_values(ngram_keys), hash_result_with_just_context

    keys = self._rust_keys
    if IS_LCG:
      n_minus_1_grams = tuple(
          tuple(gram) for gram in n_minus_1_grams.detach().cpu().tolist()
      )
      indices = tuple(tuple(index) for index in indices.detach().cpu().tolist())
      g_values, hash_result_with_just_context = (
          compute_g_values_use_LCG_from_rustlib(
              n_minus_1_grams, indices, keys, self._rust_field_prime
          )
      )
    elif HASH_TYPE == 3:
      g_values, hash_result_with_just_context = (
          compute_g_values_use_poseidon_fast_from_rustlib_buffer(
              n_minus_1_grams, indices, keys
          )
      )
    elif HASH_TYPE == 4:
      g_values, hash_result_with_just_context = (
          compute_g_values_use_poseidon2_fast_from_rustlib_buffer(
              n_minus_1_grams, indices, keys
          )
      )
    elif HASH_TYPE == 5:
      if GPU_HASH_BACKEND and gpu_hash.is_available():
        g_values, hash_result_with_just_context = (
            compute_g_values_use_mimc_gpu_buffer(
                n_minus_1_grams,
                indices,
                keys,
                dtype=dtype,
                return_context_tensor=True,
            )
        )
      else:
        g_values, hash_result_with_just_context = (
            compute_g_values_use_mimc_fast_from_rustlib_buffer(
                n_minus_1_grams, indices, keys
            )
        )
    else:
      n_minus_1_grams = tuple(
          tuple(gram) for gram in n_minus_1_grams.detach().cpu().tolist()
      )
      indices = tuple(tuple(index) for index in indices.detach().cpu().tolist())
      g_values, hash_result_with_just_context = (
          compute_g_values_use_hash_from_rustlib(
              n_minus_1_grams, indices, keys, self._rust_field_prime
          )
      )
    if isinstance(g_values, (bytes, bytearray, memoryview)):
      g_values = np.frombuffer(g_values, dtype=np.uint8).reshape(
          len(n_minus_1_grams), len(indices[0]), len(keys)
      )
    elif (
        isinstance(g_values, tuple)
        and len(g_values) == 5
        and isinstance(g_values[0], (bytes, bytearray, memoryview))
    ):
      g_values, hash_result_with_just_context, batch_size, candidate_size, depth = (
          g_values
      )
      g_values = np.frombuffer(g_values, dtype=np.uint8).reshape(
          batch_size, candidate_size, depth
      )
    return (
        g_values if torch.is_tensor(g_values) or dtype is None
        else torch.as_tensor(g_values, device=self.device, dtype=dtype),
        hash_result_with_just_context,
    )

  # def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
  def sample_g_values(self, ngram_keys) -> torch.LongTensor:
    """Samples g values from Bernoulli distribution.

    It is not possible to pass random keys in a vectorized way in torch. Instead
    we pre-compute a random sampling table, and use apply modulo table size to
    map from ngram keys (int64) to g values.

    Args:
      ngram_keys: Random keys (batch_size, num_ngrams, depth).

    Returns:
      G values (batch_size, num_ngrams, depth).
    """
    
    if RUST_LIB:
      if hasattr(ngram_keys, "detach"):
        ngram_keys = list_to_tuple_3d(ngram_keys.detach().cpu().tolist())
      else:
        ngram_keys = list_to_tuple_3d(ngram_keys)
      if IS_LCG:
        g_value = invoke_sample_g_values_use_LCG_from_rustlib(
            ngram_keys, self._rust_field_prime
        )
      else:
        g_value = invoke_sample_g_values_use_hash_from_rustlib(
            ngram_keys, self._rust_field_prime
        )
      return torch.tensor(g_value, device=self.device)
    
    else:
      (sampling_table_size,) = self.sampling_table.shape
      ## [table_size=2^16]
      sampling_table = self.sampling_table.reshape((1, 1, sampling_table_size))
      ## [0, int64 - 1]
      ngram_keys = ngram_keys % sampling_table_size
      ## [0, table_size - 1]
      return torch.take_along_dim(sampling_table, indices=ngram_keys, dim=2)

  def _check_input_ids_shape(self, input_ids: torch.LongTensor):
    """Checks the shape of input ids."""
    if len(input_ids.shape) != 2:
      raise ValueError(
          "Input ids should be of shape (batch_size, input_len), but is"
          f" {input_ids.shape}"
      )

  def compute_g_values(
      self,
      input_ids: torch.LongTensor,
  ) -> torch.LongTensor:
    """Computes g values for each ngram from the given sequence of tokens.

    Args:
      input_ids: Input token ids (batch_size, input_len).

    Returns:
      G values (batch_size, input_len - (ngram_len - 1), depth).
    """
    self._check_input_ids_shape(input_ids)
    if DEBUG_PRINT:
      print("shape of input_ids -> ", input_ids.shape)
    if RUST_LIB and RUST_FUSED_DETECT_G_VALUES and not IS_LCG:
      keys = self._rust_keys
      if HASH_TYPE == 3:
        g_values = compute_detect_g_values_use_poseidon_fast_from_rustlib_buffer(
            input_ids, keys, self.ngram_len
        )
      elif HASH_TYPE == 4:
        g_values = compute_detect_g_values_use_poseidon2_fast_from_rustlib_buffer(
            input_ids, keys, self.ngram_len
        )
      elif HASH_TYPE == 5:
        g_values = compute_detect_g_values_use_mimc_fast_from_rustlib_buffer(
            input_ids, keys, self.ngram_len
        )
      else:
        g_values = None
      if g_values is not None:
        return torch.as_tensor(g_values, device=self.device)
    ngrams = input_ids.unfold(dimension=1, size=self.ngram_len, step=1)
    if DEBUG_PRINT and not RUST_LIB:
      print("shape of ngrams -> ", ngrams.shape)
    ngram_keys = self.compute_ngram_keys(ngrams)
    if DEBUG_PRINT and not RUST_LIB:
      print("shape of ngram_keys -> ", ngram_keys.shape)
    return self.sample_g_values(ngram_keys)

  def compute_context_repetition_mask(
      self,
      input_ids: torch.LongTensor,
  ) -> torch.LongTensor:
    """Computes repetition mask.

    0 and 1 stand for repeated and not repeated context n-1 grams respectively.

    Args:
      input_ids: Input token ids (batch_size, input_len).

    Returns:
      Repetitions mask (batch_size, input_len - (ngram_len - 1)).
    """
    self._check_input_ids_shape(input_ids)
    batch_size, _ = input_ids.shape
    if RUST_LIB and RUST_FAST_CONTEXT_MASK:
      context_len = self.ngram_len - 1
      mask = compute_context_repetition_mask_lcg_from_rustlib_buffer(
          input_ids,
          context_len,
          self.context_history_size,
      )
      return torch.as_tensor(mask, device=self.device, dtype=torch.bool)

    state = SynthIDState(
        batch_size=batch_size,
        ngram_len=self.ngram_len,
        context_history_size=self.context_history_size,
        device=self.device,
    )
    contexts = input_ids[:, :-1].unfold(
        dimension=1,
        size=self.ngram_len - 1,
        step=1,
    )
    if DETECT_DEBUG_PRINT:
      print("contexts shape -> ", contexts.shape)
    _, num_contexts, _ = contexts.shape

    are_repeated_contexts = []
    context_hash = None
    contexts_cpu = contexts.detach().cpu().tolist() if RUST_LIB else None
    for i in range(num_contexts):
      context = contexts[:, i, :]
      if DETECT_DEBUG_PRINT and i == 0:
        print("context shape -> ", context.shape)
      if RUST_LIB:
        context = tuple(tuple(row[i]) for row in contexts_cpu)
        context_hash = hash_rustlib.compute_LCG_random_use_rust(
            self._rust_context_seed, context
        )
        if DETECT_DEBUG_PRINT and i == 0:
          print("len of context_hash -> ", len(context_hash))
        context_hash = torch.tensor(
            [int(hash_value) for hash_value in context_hash],
            device=self.device,
            dtype=torch.float64,
        )[:, None]
        if DETECT_DEBUG_PRINT and i == 0:
          print("context_hash shape -> ", context_hash.shape)

        
      else:
        hash_result = torch.ones(batch_size, device=self.device, dtype=torch.long)
        context_hash = hashing_function.accumulate_hash(hash_result, context)[
            :, None
        ]
      if DETECT_DEBUG_PRINT and i == 0:
        print("context_hash shape -> ", context_hash.shape)
      is_repeated_context = (state.context_history == context_hash).any(
          dim=1,
          keepdim=True,
      )
      are_repeated_contexts.append(is_repeated_context)
      state.context_history = torch.concat(
          (context_hash, state.context_history),
          dim=1,
      )[:, :-1]
    are_repeated_contexts = torch.concat(are_repeated_contexts, dim=1)

    return torch.logical_not(are_repeated_contexts)

  def compute_eos_token_mask(
      self,
      input_ids: torch.LongTensor,
      eos_token_id: int,
  ) -> torch.LongTensor:
    """Computes repetitions mask.

    1 stands for ngrams that don't contain EOS tokens and vice versa.

    Args:
      input_ids: Input token ids (batch_size, input_len).
      eos_token_id: EOS token ID.

    Returns:
      EOS token mask (batch_size, input_len).
    """
    self._check_input_ids_shape(input_ids)
    all_eos_equated = input_ids == eos_token_id
    _, seq_len = input_ids.shape
    has_eos = all_eos_equated.any(dim=1)
    first_eos = all_eos_equated.long().argmax(dim=1)
    first_eos = torch.where(
        has_eos,
        first_eos,
        torch.full_like(first_eos, seq_len),
    )
    positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    return (positions < first_eos.unsqueeze(1)).to(dtype=input_ids.dtype)
