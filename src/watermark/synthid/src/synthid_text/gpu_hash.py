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

"""Experimental CUDA kernels for hash-based SynthID.

The kernels in this module are optional and currently target the MiMC BN254
variant used by the Rust backend. Field elements are represented as eight
little-endian 32-bit limbs and arithmetic is done in Montgomery form on GPU.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

try:
  from numba import cuda
except Exception:  # pragma: no cover - optional dependency.
  cuda = None


FIELD_PRIME = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
HALF_FIELD_PRIME = FIELD_PRIME // 2
LIMBS = 8
WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
MAX_GPU_CANDIDATES = 256
MIMC_ROUNDS = 91
MONT_INV32 = (-pow(FIELD_PRIME, -1, 1 << WORD_BITS)) % (1 << WORD_BITS)
MONT_R = pow(1 << (WORD_BITS * LIMBS), 1, FIELD_PRIME)
MONT_R2 = (MONT_R * MONT_R) % FIELD_PRIME

FIELD_PRIME_LIMBS = np.asarray(
    [(FIELD_PRIME >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)],
    dtype=np.uint32,
)
HALF_FIELD_PRIME_LIMBS = np.asarray(
    [(HALF_FIELD_PRIME >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)],
    dtype=np.uint32,
)
MONT_R2_LIMBS = np.asarray(
    [(MONT_R2 >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)],
    dtype=np.uint32,
)
ONE_LIMBS = np.asarray([1] + [0] * (LIMBS - 1), dtype=np.uint32)


def _limbs_to_int(limbs: Sequence[int]) -> int:
  value = 0
  for index, limb in enumerate(limbs):
    value |= int(np.uint32(limb)) << (WORD_BITS * index)
  return value


def _int_to_limbs(value: int) -> np.ndarray:
  return np.asarray(
      [(int(value) >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)],
      dtype=np.uint32,
  )


def _montgomery_mul_host(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  """Host copy of the GPU Montgomery multiply, used for constants."""
  tmp = [0] * (LIMBS + 1)
  for i in range(LIMBS):
    carry = 0
    right_limb = int(right[i])
    for j in range(LIMBS):
      uv = tmp[j] + int(left[j]) * right_limb + carry
      tmp[j] = uv & WORD_MASK
      carry = uv >> WORD_BITS
    tmp[LIMBS] = carry

    factor = (tmp[0] * MONT_INV32) & WORD_MASK
    carry = 0
    for j in range(LIMBS):
      uv = tmp[j] + factor * int(FIELD_PRIME_LIMBS[j]) + carry
      low = uv & WORD_MASK
      carry = uv >> WORD_BITS
      if j > 0:
        tmp[j - 1] = low
    uv = tmp[LIMBS] + carry
    tmp[LIMBS - 1] = uv & WORD_MASK
    tmp[LIMBS] = uv >> WORD_BITS

  result = np.asarray(tmp[:LIMBS], dtype=np.uint32)
  if _limbs_to_int(result) >= FIELD_PRIME:
    result = _int_to_limbs(_limbs_to_int(result) - FIELD_PRIME)
  return result


def _to_montgomery_host(value: int) -> np.ndarray:
  return _montgomery_mul_host(_int_to_limbs(value % FIELD_PRIME), MONT_R2_LIMBS)


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _mimc_round_keys_decimal() -> tuple[int, ...]:
  constants_path = (
      _repo_root()
      / "artifact"
      / "third_party"
      / "hash_function"
      / "arkworks-mimc"
      / "src"
      / "params"
      / "mimc_7_91_bn254.rs"
  )
  text = constants_path.read_text(encoding="utf-8")
  match = re.search(r"MIMC_7_91_BN254_ROUND_KEYS:[^\[]+\[(.*?)\];", text, re.S)
  if match is None:
    raise RuntimeError(f"Could not parse MiMC round keys from {constants_path}")
  keys = tuple(int(value) for value in re.findall(r'"([0-9]+)"', match.group(1)))
  if len(keys) != MIMC_ROUNDS:
    raise RuntimeError(f"Expected {MIMC_ROUNDS} MiMC round keys, got {len(keys)}")
  return keys


@functools.lru_cache(maxsize=32)
def _host_constant_arrays(keys_tuple: tuple[int, ...]) -> tuple[np.ndarray, ...]:
  round_keys_mont = np.stack(
      [_to_montgomery_host(value) for value in _mimc_round_keys_decimal()],
      axis=0,
  ).astype(np.uint32)
  keys_mont = np.stack(
      [_to_montgomery_host(int(value)) for value in keys_tuple],
      axis=0,
  ).astype(np.uint32)
  key_index_mont = np.stack(
      [_to_montgomery_host(index) for index in range(len(keys_tuple))],
      axis=0,
  ).astype(np.uint32)
  return (
      FIELD_PRIME_LIMBS.copy(),
      HALF_FIELD_PRIME_LIMBS.copy(),
      MONT_R2_LIMBS.copy(),
      ONE_LIMBS.copy(),
      round_keys_mont,
      keys_mont,
      key_index_mont,
  )


_DEVICE_CONSTANTS: dict[tuple[int, tuple[int, ...]], tuple[object, ...]] = {}


def _device_constant_arrays(device_index: int, keys: Sequence[int]) -> tuple[object, ...]:
  if cuda is None:
    raise RuntimeError("Numba CUDA is not available")
  keys_tuple = tuple(int(value) for value in keys)
  cache_key = (int(device_index), keys_tuple)
  if cache_key not in _DEVICE_CONSTANTS:
    with cuda.gpus[int(device_index)]:
      host_arrays = _host_constant_arrays(keys_tuple)
      _DEVICE_CONSTANTS[cache_key] = tuple(cuda.to_device(array) for array in host_arrays)
  return _DEVICE_CONSTANTS[cache_key]


if cuda is not None:

  @cuda.jit(device=True, inline=True)
  def _copy8(dst, src):
    for i in range(LIMBS):
      dst[i] = src[i]


  @cuda.jit(device=True, inline=True)
  def _copy8_from_2d(dst, src, row):
    for i in range(LIMBS):
      dst[i] = src[row, i]


  @cuda.jit(device=True, inline=True)
  def _copy8_to_2d(dst, row, src):
    for i in range(LIMBS):
      dst[row, i] = src[i]


  @cuda.jit(device=True, inline=True)
  def _copy8_to_shared_2d(dst, row, src):
    for i in range(LIMBS):
      dst[row, i] = src[i]


  @cuda.jit(device=True, inline=True)
  def _copy8_from_shared_2d(dst, src, row):
    for i in range(LIMBS):
      dst[i] = src[row, i]


  @cuda.jit(device=True, inline=True)
  def _set_u64_limbs(out, value):
    out[0] = np.uint32(value & np.uint64(WORD_MASK))
    out[1] = np.uint32((value >> np.uint64(WORD_BITS)) & np.uint64(WORD_MASK))
    for i in range(2, LIMBS):
      out[i] = np.uint32(0)


  @cuda.jit(device=True, inline=True)
  def _ge8(left, right):
    for offset in range(LIMBS):
      index = LIMBS - 1 - offset
      if left[index] > right[index]:
        return True
      if left[index] < right[index]:
        return False
    return True


  @cuda.jit(device=True, inline=True)
  def _add_mod(out, left, right, prime):
    carry = np.uint64(0)
    for i in range(LIMBS):
      total = np.uint64(left[i]) + np.uint64(right[i]) + carry
      out[i] = np.uint32(total & np.uint64(WORD_MASK))
      carry = total >> np.uint64(WORD_BITS)
    if carry != 0 or _ge8(out, prime):
      borrow = np.uint64(0)
      for i in range(LIMBS):
        subtrahend = np.uint64(prime[i]) + borrow
        current = np.uint64(out[i])
        out[i] = np.uint32((current - subtrahend) & np.uint64(WORD_MASK))
        borrow = np.uint64(1) if current < subtrahend else np.uint64(0)


  @cuda.jit(device=True, inline=True)
  def _montgomery_mul(out, left, right, prime):
    tmp = cuda.local.array(9, dtype=np.uint32)
    for i in range(LIMBS + 1):
      tmp[i] = np.uint32(0)

    for i in range(LIMBS):
      carry = np.uint64(0)
      right_limb = np.uint64(right[i])
      for j in range(LIMBS):
        uv = np.uint64(tmp[j]) + np.uint64(left[j]) * right_limb + carry
        tmp[j] = np.uint32(uv & np.uint64(WORD_MASK))
        carry = uv >> np.uint64(WORD_BITS)
      tmp[LIMBS] = np.uint32(carry)

      factor = np.uint32(
          (np.uint64(tmp[0]) * np.uint64(MONT_INV32)) & np.uint64(WORD_MASK)
      )
      carry = np.uint64(0)
      for j in range(LIMBS):
        uv = np.uint64(tmp[j]) + np.uint64(factor) * np.uint64(prime[j]) + carry
        low = np.uint32(uv & np.uint64(WORD_MASK))
        carry = uv >> np.uint64(WORD_BITS)
        if j > 0:
          tmp[j - 1] = low
      uv = np.uint64(tmp[LIMBS]) + carry
      tmp[LIMBS - 1] = np.uint32(uv & np.uint64(WORD_MASK))
      tmp[LIMBS] = np.uint32(uv >> np.uint64(WORD_BITS))

    for i in range(LIMBS):
      out[i] = tmp[i]
    if _ge8(out, prime):
      borrow = np.uint64(0)
      for i in range(LIMBS):
        subtrahend = np.uint64(prime[i]) + borrow
        current = np.uint64(out[i])
        out[i] = np.uint32((current - subtrahend) & np.uint64(WORD_MASK))
        borrow = np.uint64(1) if current < subtrahend else np.uint64(0)


  @cuda.jit(device=True, inline=True)
  def _to_montgomery_u64(out, value, r2, prime):
    canonical = cuda.local.array(8, dtype=np.uint32)
    _set_u64_limbs(canonical, np.uint64(value))
    _montgomery_mul(out, canonical, r2, prime)


  @cuda.jit(device=True, inline=True)
  def _from_montgomery(out, value, one, prime):
    _montgomery_mul(out, value, one, prime)


  @cuda.jit(device=True, inline=True)
  def _pow7(out, value, prime):
    square = cuda.local.array(8, dtype=np.uint32)
    fourth = cuda.local.array(8, dtype=np.uint32)
    sixth = cuda.local.array(8, dtype=np.uint32)
    _montgomery_mul(square, value, value, prime)
    _montgomery_mul(fourth, square, square, prime)
    _montgomery_mul(sixth, fourth, square, prime)
    _montgomery_mul(out, sixth, value, prime)


  @cuda.jit(device=True)
  def _mimc_non_feistel(out, x, k, round_keys, prime):
    r = cuda.local.array(8, dtype=np.uint32)
    tmp = cuda.local.array(8, dtype=np.uint32)
    tmp2 = cuda.local.array(8, dtype=np.uint32)
    round_key = cuda.local.array(8, dtype=np.uint32)
    for limb in range(LIMBS):
      r[limb] = np.uint32(0)

    _add_mod(tmp, k, x, prime)
    _pow7(r, tmp, prime)

    for round_index in range(1, MIMC_ROUNDS):
      _copy8_from_2d(round_key, round_keys, round_index)
      _add_mod(tmp, k, r, prime)
      _add_mod(tmp2, tmp, round_key, prime)
      _pow7(r, tmp2, prime)

    _add_mod(out, r, k, prime)


  @cuda.jit(device=True)
  def _mimc_hash_pair(out, input1, input2, round_keys, prime):
    r = cuda.local.array(8, dtype=np.uint32)
    enc = cuda.local.array(8, dtype=np.uint32)
    tmp = cuda.local.array(8, dtype=np.uint32)
    for limb in range(LIMBS):
      r[limb] = np.uint32(0)

    _mimc_non_feistel(enc, input1, r, round_keys, prime)
    _add_mod(tmp, input1, enc, prime)
    _add_mod(r, r, tmp, prime)

    _mimc_non_feistel(enc, input2, r, round_keys, prime)
    _add_mod(tmp, input2, enc, prime)
    _add_mod(out, r, tmp, prime)


  @cuda.jit
  def _mimc_wet_kernel(
      ngrams,
      indices,
      g_values,
      context_out,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      sliding_window_size,
      candidate_token_size,
      num_keys,
  ):
    batch = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    block_threads = cuda.blockDim.x

    context_hash = cuda.shared.array(8, dtype=np.uint32)
    candidate_hashes = cuda.shared.array((256, 8), dtype=np.uint32)

    if tid == 0:
      result = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      next_hash = cuda.local.array(8, dtype=np.uint32)
      _to_montgomery_u64(result, np.uint64(1), r2, prime)
      for offset in range(sliding_window_size):
        token_value = np.uint64(ngrams[batch, offset])
        _to_montgomery_u64(token_mont, token_value, r2, prime)
        _mimc_hash_pair(next_hash, result, token_mont, round_keys, prime)
        _copy8(result, next_hash)
      _copy8(context_hash, result)
      _from_montgomery(next_hash, result, one, prime)
      _copy8_to_2d(context_out, batch, next_hash)

    cuda.syncthreads()

    if tid < candidate_token_size:
      token_mont = cuda.local.array(8, dtype=np.uint32)
      candidate_hash = cuda.local.array(8, dtype=np.uint32)
      token_value = np.uint64(indices[batch, tid])
      _to_montgomery_u64(token_mont, token_value, r2, prime)
      _mimc_hash_pair(candidate_hash, context_hash, token_mont, round_keys, prime)
      _copy8_to_shared_2d(candidate_hashes, tid, candidate_hash)

    cuda.syncthreads()

    total = candidate_token_size * num_keys
    flat = tid
    while flat < total:
      candidate_index = flat // num_keys
      key_index = flat - candidate_index * num_keys
      candidate_hash = cuda.local.array(8, dtype=np.uint32)
      key_hash = cuda.local.array(8, dtype=np.uint32)
      g_hash = cuda.local.array(8, dtype=np.uint32)
      g_canonical = cuda.local.array(8, dtype=np.uint32)
      key_value = cuda.local.array(8, dtype=np.uint32)
      key_index_value = cuda.local.array(8, dtype=np.uint32)
      _copy8_from_shared_2d(candidate_hash, candidate_hashes, candidate_index)
      _copy8_from_2d(key_value, keys_mont, key_index)
      _copy8_from_2d(key_index_value, key_index_mont, key_index)
      _mimc_hash_pair(key_hash, candidate_hash, key_value, round_keys, prime)
      _mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime)
      _from_montgomery(g_canonical, g_hash, one, prime)
      g_values[batch, candidate_index, key_index] = np.uint8(
          1 if _ge8(g_canonical, half_prime) and not _ge8(half_prime, g_canonical) else 0
      )
      flat += block_threads


  @cuda.jit
  def _mimc_wet_candidate_kernel(
      ngrams,
      indices,
      g_values,
      context_out,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      sliding_window_size,
      candidate_token_size,
      num_keys,
  ):
    batch = cuda.blockIdx.x
    candidate_index = cuda.blockIdx.y
    tid = cuda.threadIdx.x

    context_hash = cuda.shared.array(8, dtype=np.uint32)
    candidate_hash_shared = cuda.shared.array(8, dtype=np.uint32)
    candidate_key_prefix_shared = cuda.shared.array(8, dtype=np.uint32)

    if candidate_index < candidate_token_size and tid == 0:
      result = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      next_hash = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _to_montgomery_u64(result, np.uint64(1), r2, prime)
      for offset in range(sliding_window_size):
        token_value = np.uint64(ngrams[batch, offset])
        _to_montgomery_u64(token_mont, token_value, r2, prime)
        _mimc_hash_pair(next_hash, result, token_mont, round_keys, prime)
        _copy8(result, next_hash)
      _copy8(context_hash, result)
      if candidate_index == 0:
        _from_montgomery(next_hash, result, one, prime)
        _copy8_to_2d(context_out, batch, next_hash)

      token_value = np.uint64(indices[batch, candidate_index])
      _to_montgomery_u64(token_mont, token_value, r2, prime)
      _mimc_hash_pair(next_hash, context_hash, token_mont, round_keys, prime)
      _copy8(candidate_hash_shared, next_hash)
      for limb in range(LIMBS):
        candidate_key_prefix_shared[limb] = np.uint32(0)
      _mimc_non_feistel(enc, next_hash, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, next_hash, enc, prime)
      _add_mod(candidate_key_prefix_shared, candidate_key_prefix_shared, tmp, prime)

    cuda.syncthreads()

    if candidate_index < candidate_token_size and tid < num_keys:
      key_hash = cuda.local.array(8, dtype=np.uint32)
      g_hash = cuda.local.array(8, dtype=np.uint32)
      g_canonical = cuda.local.array(8, dtype=np.uint32)
      key_value = cuda.local.array(8, dtype=np.uint32)
      key_index_value = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _copy8_from_2d(key_value, keys_mont, tid)
      _copy8_from_2d(key_index_value, key_index_mont, tid)
      _mimc_non_feistel(enc, key_value, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, key_value, enc, prime)
      _add_mod(key_hash, candidate_key_prefix_shared, tmp, prime)
      _mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime)
      _from_montgomery(g_canonical, g_hash, one, prime)
      g_values[batch, candidate_index, tid] = np.uint8(
          1 if _ge8(g_canonical, half_prime) and not _ge8(half_prime, g_canonical) else 0
      )


  @cuda.jit
  def _mimc_wet_candidate_history_kernel(
      ngrams,
      indices,
      g_values,
      context_out,
      repeated_flags,
      context_history,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      write_index,
      context_history_size,
      sliding_window_size,
      candidate_token_size,
      num_keys,
  ):
    batch = cuda.blockIdx.x
    candidate_index = cuda.blockIdx.y
    tid = cuda.threadIdx.x

    context_hash = cuda.shared.array(8, dtype=np.uint32)
    candidate_hash_shared = cuda.shared.array(8, dtype=np.uint32)
    candidate_key_prefix_shared = cuda.shared.array(8, dtype=np.uint32)

    if candidate_index < candidate_token_size and tid == 0:
      result = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      next_hash = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _to_montgomery_u64(result, np.uint64(1), r2, prime)
      for offset in range(sliding_window_size):
        token_value = np.uint64(ngrams[batch, offset])
        _to_montgomery_u64(token_mont, token_value, r2, prime)
        _mimc_hash_pair(next_hash, result, token_mont, round_keys, prime)
        _copy8(result, next_hash)
      _copy8(context_hash, result)
      if candidate_index == 0:
        _from_montgomery(next_hash, result, one, prime)
        _copy8_to_2d(context_out, batch, next_hash)

        repeated = False
        for history_index in range(context_history_size):
          matches = True
          for limb in range(LIMBS):
            if context_history[batch, history_index, limb] != next_hash[limb]:
              matches = False
              break
          if matches:
            repeated = True
            break
        repeated_flags[batch] = np.uint8(1 if repeated else 0)
        for limb in range(LIMBS):
          context_history[batch, write_index, limb] = next_hash[limb]

      token_value = np.uint64(indices[batch, candidate_index])
      _to_montgomery_u64(token_mont, token_value, r2, prime)
      _mimc_hash_pair(next_hash, context_hash, token_mont, round_keys, prime)
      _copy8(candidate_hash_shared, next_hash)
      for limb in range(LIMBS):
        candidate_key_prefix_shared[limb] = np.uint32(0)
      _mimc_non_feistel(enc, next_hash, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, next_hash, enc, prime)
      _add_mod(candidate_key_prefix_shared, candidate_key_prefix_shared, tmp, prime)

    cuda.syncthreads()

    if candidate_index < candidate_token_size and tid < num_keys:
      key_hash = cuda.local.array(8, dtype=np.uint32)
      g_hash = cuda.local.array(8, dtype=np.uint32)
      g_canonical = cuda.local.array(8, dtype=np.uint32)
      key_value = cuda.local.array(8, dtype=np.uint32)
      key_index_value = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _copy8_from_2d(key_value, keys_mont, tid)
      _copy8_from_2d(key_index_value, key_index_mont, tid)
      _mimc_non_feistel(enc, key_value, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, key_value, enc, prime)
      _add_mod(key_hash, candidate_key_prefix_shared, tmp, prime)
      _mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime)
      _from_montgomery(g_canonical, g_hash, one, prime)
      g_values[batch, candidate_index, tid] = np.uint8(
          1 if _ge8(g_canonical, half_prime) and not _ge8(half_prime, g_canonical) else 0
      )


  @cuda.jit
  def _mimc_context_kernel(
      ngrams,
      context_out,
      prime,
      r2,
      one,
      round_keys,
      sliding_window_size,
  ):
    batch = cuda.blockIdx.x
    if cuda.threadIdx.x != 0:
      return

    result = cuda.local.array(8, dtype=np.uint32)
    token_mont = cuda.local.array(8, dtype=np.uint32)
    next_hash = cuda.local.array(8, dtype=np.uint32)
    _to_montgomery_u64(result, np.uint64(1), r2, prime)
    for offset in range(sliding_window_size):
      token_value = np.uint64(ngrams[batch, offset])
      _to_montgomery_u64(token_mont, token_value, r2, prime)
      _mimc_hash_pair(next_hash, result, token_mont, round_keys, prime)
      _copy8(result, next_hash)
    _from_montgomery(next_hash, result, one, prime)
    _copy8_to_2d(context_out, batch, next_hash)


  @cuda.jit
  def _mimc_wet_candidate_from_context_kernel(
      context_in,
      indices,
      g_values,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      candidate_token_size,
      num_keys,
  ):
    batch = cuda.blockIdx.x
    candidate_index = cuda.blockIdx.y
    tid = cuda.threadIdx.x

    candidate_hash_shared = cuda.shared.array(8, dtype=np.uint32)
    candidate_key_prefix_shared = cuda.shared.array(8, dtype=np.uint32)

    if candidate_index < candidate_token_size and tid == 0:
      context_mont = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      candidate_hash = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _copy8_from_2d(context_mont, context_in, batch)
      _montgomery_mul(context_mont, context_mont, r2, prime)
      _to_montgomery_u64(token_mont, np.uint64(indices[batch, candidate_index]), r2, prime)
      _mimc_hash_pair(candidate_hash, context_mont, token_mont, round_keys, prime)
      _copy8(candidate_hash_shared, candidate_hash)
      for limb in range(LIMBS):
        candidate_key_prefix_shared[limb] = np.uint32(0)
      _mimc_non_feistel(enc, candidate_hash, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, candidate_hash, enc, prime)
      _add_mod(candidate_key_prefix_shared, candidate_key_prefix_shared, tmp, prime)

    cuda.syncthreads()

    if candidate_index < candidate_token_size and tid < num_keys:
      key_hash = cuda.local.array(8, dtype=np.uint32)
      g_hash = cuda.local.array(8, dtype=np.uint32)
      g_canonical = cuda.local.array(8, dtype=np.uint32)
      key_value = cuda.local.array(8, dtype=np.uint32)
      key_index_value = cuda.local.array(8, dtype=np.uint32)
      enc = cuda.local.array(8, dtype=np.uint32)
      tmp = cuda.local.array(8, dtype=np.uint32)
      _copy8_from_2d(key_value, keys_mont, tid)
      _copy8_from_2d(key_index_value, key_index_mont, tid)
      _mimc_non_feistel(enc, key_value, candidate_key_prefix_shared, round_keys, prime)
      _add_mod(tmp, key_value, enc, prime)
      _add_mod(key_hash, candidate_key_prefix_shared, tmp, prime)
      _mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime)
      _from_montgomery(g_canonical, g_hash, one, prime)
      g_values[batch, candidate_index, tid] = np.uint8(
          1 if _ge8(g_canonical, half_prime) and not _ge8(half_prime, g_canonical) else 0
      )


  @cuda.jit
  def _context_history_update_kernel(
      context_history,
      context_hash,
      repeated_flags,
      write_index,
      context_history_size,
  ):
    batch = cuda.blockIdx.x
    if cuda.threadIdx.x != 0:
      return

    repeated = False
    for history_index in range(context_history_size):
      matches = True
      for limb in range(LIMBS):
        if context_history[batch, history_index, limb] != context_hash[batch, limb]:
          matches = False
          break
      if matches:
        repeated = True
        break

    repeated_flags[batch] = np.uint8(1 if repeated else 0)
    for limb in range(LIMBS):
      context_history[batch, write_index, limb] = context_hash[batch, limb]


  @cuda.jit
  def _batched_context_repetition_kernel(
      context_hashes,
      repeated_flags,
      num_steps,
      batch_size,
      context_history_size,
  ):
    row = cuda.blockIdx.x
    if cuda.threadIdx.x != 0:
      return

    step = row // batch_size
    batch = row - step * batch_size

    repeated = False
    current_is_zero = True
    for limb in range(LIMBS):
      if context_hashes[row, limb] != 0:
        current_is_zero = False
        break
    if current_is_zero and step < context_history_size:
      repeated = True

    if not repeated:
      first_step = step - context_history_size
      if first_step < 0:
        first_step = 0
      for previous_step in range(first_step, step):
        previous_row = previous_step * batch_size + batch
        matches = True
        for limb in range(LIMBS):
          if context_hashes[previous_row, limb] != context_hashes[row, limb]:
            matches = False
            break
        if matches:
          repeated = True
          break

    repeated_flags[row] = np.uint8(1 if repeated else 0)


  @cuda.jit
  def _update_scores_kernel(scores, g_values, repeated_flags, output, candidate_size, num_keys):
    batch = cuda.blockIdx.x
    if cuda.threadIdx.x != 0:
      return

    probs = cuda.local.array(256, dtype=np.float32)
    if repeated_flags[batch] != 0:
      for candidate in range(candidate_size):
        output[batch, candidate] = scores[batch, candidate]
      return

    max_score = np.float32(scores[batch, 0])
    for candidate in range(1, candidate_size):
      score = np.float32(scores[batch, candidate])
      if score > max_score:
        max_score = score

    normalizer = np.float32(0.0)
    for candidate in range(candidate_size):
      value = np.float32(math.exp(np.float32(scores[batch, candidate]) - max_score))
      probs[candidate] = value
      normalizer += value
    for candidate in range(candidate_size):
      probs[candidate] = probs[candidate] / normalizer

    for key_index in range(num_keys):
      g_mass = np.float32(0.0)
      for candidate in range(candidate_size):
        if g_values[batch, candidate, key_index] != 0:
          g_mass += probs[candidate]
      for candidate in range(candidate_size):
        g_value = np.float32(1.0 if g_values[batch, candidate, key_index] != 0 else 0.0)
        probs[candidate] *= np.float32(1.0) + g_value - g_mass

    for candidate in range(candidate_size):
      prob = probs[candidate]
      output[batch, candidate] = (
          np.float32(math.log(prob)) if prob > np.float32(0.0) else np.float32(-1.0e12)
      )


  @cuda.jit
  def _mimc_wet_fused_scores_kernel(
      ngrams,
      indices,
      scores,
      output,
      context_history,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      write_index,
      sliding_window_size,
      context_history_size,
      candidate_token_size,
      num_keys,
  ):
    batch = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    context_hash = cuda.shared.array(8, dtype=np.uint32)
    g_values_shared = cuda.shared.array((256, 64), dtype=np.uint8)
    probs = cuda.shared.array(256, dtype=np.float32)
    repeated_shared = cuda.shared.array(1, dtype=np.uint8)
    g_mass_shared = cuda.shared.array(1, dtype=np.float32)

    if tid == 0:
      result = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      next_hash = cuda.local.array(8, dtype=np.uint32)
      _to_montgomery_u64(result, np.uint64(1), r2, prime)
      for offset in range(sliding_window_size):
        token_value = np.uint64(ngrams[batch, offset])
        _to_montgomery_u64(token_mont, token_value, r2, prime)
        _mimc_hash_pair(next_hash, result, token_mont, round_keys, prime)
        _copy8(result, next_hash)
      _from_montgomery(next_hash, result, one, prime)
      _copy8(context_hash, next_hash)

      repeated = False
      for history_index in range(context_history_size):
        matches = True
        for limb in range(LIMBS):
          if context_history[batch, history_index, limb] != context_hash[limb]:
            matches = False
            break
        if matches:
          repeated = True
          break
      repeated_shared[0] = np.uint8(1 if repeated else 0)
      for limb in range(LIMBS):
        context_history[batch, write_index, limb] = context_hash[limb]

    cuda.syncthreads()

    flat = tid
    total = candidate_token_size * num_keys
    while flat < total:
      candidate_index = flat // num_keys
      key_index = flat - candidate_index * num_keys
      context_mont = cuda.local.array(8, dtype=np.uint32)
      token_mont = cuda.local.array(8, dtype=np.uint32)
      candidate_hash = cuda.local.array(8, dtype=np.uint32)
      key_hash = cuda.local.array(8, dtype=np.uint32)
      g_hash = cuda.local.array(8, dtype=np.uint32)
      g_canonical = cuda.local.array(8, dtype=np.uint32)
      key_value = cuda.local.array(8, dtype=np.uint32)
      key_index_value = cuda.local.array(8, dtype=np.uint32)
      _montgomery_mul(context_mont, context_hash, r2, prime)
      _to_montgomery_u64(token_mont, np.uint64(indices[batch, candidate_index]), r2, prime)
      _copy8_from_2d(key_value, keys_mont, key_index)
      _copy8_from_2d(key_index_value, key_index_mont, key_index)
      _mimc_hash_pair(candidate_hash, context_mont, token_mont, round_keys, prime)
      _mimc_hash_pair(key_hash, candidate_hash, key_value, round_keys, prime)
      _mimc_hash_pair(g_hash, key_hash, key_index_value, round_keys, prime)
      _from_montgomery(g_canonical, g_hash, one, prime)
      g_values_shared[candidate_index, key_index] = np.uint8(
          1 if _ge8(g_canonical, half_prime) and not _ge8(half_prime, g_canonical) else 0
      )
      flat += cuda.blockDim.x

    cuda.syncthreads()

    if tid == 0:
      if repeated_shared[0] != 0:
        for candidate in range(candidate_token_size):
          output[batch, candidate] = scores[batch, candidate]
        return

      max_score = np.float32(scores[batch, 0])
      for candidate in range(1, candidate_token_size):
        score = np.float32(scores[batch, candidate])
        if score > max_score:
          max_score = score

      normalizer = np.float32(0.0)
      for candidate in range(candidate_token_size):
        value = np.float32(math.exp(np.float32(scores[batch, candidate]) - max_score))
        probs[candidate] = value
        normalizer += value
      for candidate in range(candidate_token_size):
        probs[candidate] = probs[candidate] / normalizer

    cuda.syncthreads()

    for key_index in range(num_keys):
      if tid == 0:
        g_mass = np.float32(0.0)
        for candidate in range(candidate_token_size):
          if g_values_shared[candidate, key_index] != 0:
            g_mass += probs[candidate]
        g_mass_shared[0] = g_mass

      cuda.syncthreads()

      candidate = tid
      while candidate < candidate_token_size:
        g_value = np.float32(
            1.0 if g_values_shared[candidate, key_index] != 0 else 0.0
        )
        probs[candidate] *= np.float32(1.0) + g_value - g_mass_shared[0]
        candidate += cuda.blockDim.x

      cuda.syncthreads()

    candidate = tid
    while candidate < candidate_token_size:
      prob = probs[candidate]
      output[batch, candidate] = (
          np.float32(math.log(prob)) if prob > np.float32(0.0) else np.float32(-1.0e12)
      )
      candidate += cuda.blockDim.x


def is_available() -> bool:
  return cuda is not None and cuda.is_available() and torch.cuda.is_available()


def compute_g_values_use_mimc_gpu(
    n_minus_1_grams: torch.Tensor,
    indices: torch.Tensor,
    keys: Sequence[int],
    dtype: torch.dtype | None = None,
    return_context_tensor: bool = False,
) -> tuple[torch.Tensor, list[str]]:
  """Compute MiMC SynthID WET g-values on GPU.

  Args:
    n_minus_1_grams: CUDA tensor of shape [batch, ngram_len - 1].
    indices: CUDA tensor of top-k token ids, shape [batch, candidates].
    keys: SynthID depth keys.
    dtype: Optional dtype for the returned tensor.

  Returns:
    A CUDA tensor of g-values and decimal context hashes for repetition masking.
  """
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if not n_minus_1_grams.is_cuda or not indices.is_cuda:
    raise ValueError("MiMC GPU backend requires CUDA tensors")
  if n_minus_1_grams.device != indices.device:
    raise ValueError("n_minus_1_grams and indices must be on the same CUDA device")

  batch_size, sliding_window_size = n_minus_1_grams.shape
  _, candidate_token_size = indices.shape
  num_keys = len(keys)
  if candidate_token_size > MAX_GPU_CANDIDATES:
    raise ValueError(
        f"MiMC GPU backend supports at most {MAX_GPU_CANDIDATES} candidates, "
        f"got {candidate_token_size}"
    )

  device_index = n_minus_1_grams.device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  constants = _device_constant_arrays(device_index, keys)
  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = constants

  ngrams_cuda = cuda.as_cuda_array(n_minus_1_grams.contiguous())
  indices_cuda = cuda.as_cuda_array(indices.contiguous())
  g_values = torch.empty(
      (batch_size, candidate_token_size, num_keys),
      device=n_minus_1_grams.device,
      dtype=torch.uint8,
  )
  context_limbs = torch.empty(
      (batch_size, LIMBS),
      device=n_minus_1_grams.device,
      dtype=torch.int32,
  )
  g_values_cuda = cuda.as_cuda_array(g_values)
  context_cuda = cuda.as_cuda_array(context_limbs)

  threads = 64
  _mimc_wet_candidate_kernel[(batch_size, candidate_token_size), threads](
      ngrams_cuda,
      indices_cuda,
      g_values_cuda,
      context_cuda,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(sliding_window_size),
      int(candidate_token_size),
      int(num_keys),
  )

  if dtype is not None and dtype != torch.uint8:
    g_values = g_values.to(dtype=dtype)
  if return_context_tensor:
    return g_values, context_limbs
  context_np = context_limbs.detach().cpu().numpy().view(np.uint32)
  context_hashes = [str(_limbs_to_int(row)) for row in context_np]
  return g_values, context_hashes


def compute_g_values_use_mimc_gpu_split_context(
    n_minus_1_grams: torch.Tensor,
    indices: torch.Tensor,
    keys: Sequence[int],
    dtype: torch.dtype | None = None,
    return_context_tensor: bool = False,
) -> tuple[torch.Tensor, list[str]]:
  """Compute MiMC WET g-values after computing each context hash once."""
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if not n_minus_1_grams.is_cuda or not indices.is_cuda:
    raise ValueError("MiMC GPU backend requires CUDA tensors")
  if n_minus_1_grams.device != indices.device:
    raise ValueError("n_minus_1_grams and indices must be on the same CUDA device")

  batch_size, sliding_window_size = n_minus_1_grams.shape
  _, candidate_token_size = indices.shape
  num_keys = len(keys)
  if candidate_token_size > MAX_GPU_CANDIDATES:
    raise ValueError(
        f"MiMC GPU backend supports at most {MAX_GPU_CANDIDATES} candidates, "
        f"got {candidate_token_size}"
    )

  device_index = n_minus_1_grams.device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  constants = _device_constant_arrays(device_index, keys)
  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = constants

  context_limbs = torch.empty(
      (batch_size, LIMBS),
      device=n_minus_1_grams.device,
      dtype=torch.int32,
  )
  g_values = torch.empty(
      (batch_size, candidate_token_size, num_keys),
      device=n_minus_1_grams.device,
      dtype=torch.uint8,
  )

  _mimc_context_kernel[batch_size, 1](
      cuda.as_cuda_array(n_minus_1_grams.contiguous()),
      cuda.as_cuda_array(context_limbs),
      prime,
      r2,
      one,
      round_keys,
      int(sliding_window_size),
  )
  _mimc_wet_candidate_from_context_kernel[(batch_size, candidate_token_size), 32](
      cuda.as_cuda_array(context_limbs),
      cuda.as_cuda_array(indices.contiguous()),
      cuda.as_cuda_array(g_values),
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(candidate_token_size),
      int(num_keys),
  )

  if dtype is not None and dtype != torch.uint8:
    g_values = g_values.to(dtype=dtype)
  if return_context_tensor:
    return g_values, context_limbs
  context_np = context_limbs.detach().cpu().numpy().view(np.uint32)
  context_hashes = [str(_limbs_to_int(row)) for row in context_np]
  return g_values, context_hashes


def compute_g_values_and_repetition_use_mimc_gpu(
    n_minus_1_grams: torch.Tensor,
    indices: torch.Tensor,
    keys: Sequence[int],
    context_history: torch.Tensor,
    write_index: int,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute MiMC g-values and update context history in one GPU launch."""
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if not n_minus_1_grams.is_cuda or not indices.is_cuda:
    raise ValueError("MiMC GPU backend requires CUDA tensors")
  if n_minus_1_grams.device != indices.device:
    raise ValueError("n_minus_1_grams and indices must be on the same CUDA device")

  batch_size, sliding_window_size = n_minus_1_grams.shape
  _, candidate_token_size = indices.shape
  _, context_history_size, _ = context_history.shape
  num_keys = len(keys)
  if candidate_token_size > MAX_GPU_CANDIDATES:
    raise ValueError(
        f"MiMC GPU backend supports at most {MAX_GPU_CANDIDATES} candidates, "
        f"got {candidate_token_size}"
    )

  device_index = n_minus_1_grams.device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  constants = _device_constant_arrays(device_index, keys)
  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = constants

  g_values = torch.empty(
      (batch_size, candidate_token_size, num_keys),
      device=n_minus_1_grams.device,
      dtype=torch.uint8,
  )
  context_limbs = torch.empty(
      (batch_size, LIMBS),
      device=n_minus_1_grams.device,
      dtype=torch.int32,
  )
  repeated_flags = torch.empty(
      (batch_size,),
      device=n_minus_1_grams.device,
      dtype=torch.uint8,
  )

  _mimc_wet_candidate_history_kernel[(batch_size, candidate_token_size), 64](
      cuda.as_cuda_array(n_minus_1_grams.contiguous()),
      cuda.as_cuda_array(indices.contiguous()),
      cuda.as_cuda_array(g_values),
      cuda.as_cuda_array(context_limbs),
      cuda.as_cuda_array(repeated_flags),
      cuda.as_cuda_array(context_history),
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(write_index),
      int(context_history_size),
      int(sliding_window_size),
      int(candidate_token_size),
      int(num_keys),
  )

  if dtype is not None and dtype != torch.uint8:
    g_values = g_values.to(dtype=dtype)
  return g_values, repeated_flags


def update_context_history_gpu(
    context_history: torch.Tensor,
    context_hash: torch.Tensor,
    write_index: int,
) -> torch.Tensor:
  """Check and update context history on GPU.

  Args:
    context_history: CUDA int32 tensor [batch, history, limbs].
    context_hash: CUDA int32 tensor [batch, limbs].
    write_index: Circular slot to overwrite after the repetition check.

  Returns:
    CUDA uint8 tensor [batch], with 1 meaning repeated context.
  """
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  batch_size, context_history_size, _ = context_history.shape
  repeated_flags = torch.empty(
      (batch_size,),
      device=context_history.device,
      dtype=torch.uint8,
  )
  _context_history_update_kernel[batch_size, 1](
      cuda.as_cuda_array(context_history),
      cuda.as_cuda_array(context_hash.contiguous()),
      cuda.as_cuda_array(repeated_flags),
      int(write_index),
      int(context_history_size),
  )
  return repeated_flags


def compute_batched_repetition_flags_gpu(
    context_hashes: torch.Tensor,
    num_steps: int,
    batch_size: int,
    context_history_size: int,
) -> torch.Tensor:
  """Compute replay repetition flags for flattened [step, batch] contexts.

  The online state starts with a circular history filled with zero hashes. For
  replay, a context is repeated if it is zero or if the same batch item saw the
  same context in the previous context_history_size steps.
  """
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if not context_hashes.is_cuda:
    raise ValueError("Batched repetition flags require CUDA context hashes")
  expected_rows = int(num_steps) * int(batch_size)
  if context_hashes.shape != (expected_rows, LIMBS):
    raise ValueError(
        "context_hashes must have shape "
        f"({expected_rows}, {LIMBS}), got {tuple(context_hashes.shape)}"
    )

  repeated_flags = torch.empty(
      (expected_rows,),
      device=context_hashes.device,
      dtype=torch.uint8,
  )
  _batched_context_repetition_kernel[expected_rows, 1](
      cuda.as_cuda_array(context_hashes.contiguous()),
      cuda.as_cuda_array(repeated_flags),
      int(num_steps),
      int(batch_size),
      int(context_history_size),
  )
  return repeated_flags


def update_scores_gpu(
    scores: torch.Tensor,
    g_values: torch.Tensor,
    repeated_flags: torch.Tensor,
) -> torch.Tensor:
  """Fused GPU implementation of the binary SynthID score recurrence."""
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if scores.dtype != torch.float32:
    raise ValueError("GPU score update currently supports float32 scores only")
  batch_size, candidate_size = scores.shape
  num_keys = g_values.shape[-1]
  if candidate_size > MAX_GPU_CANDIDATES:
    raise ValueError(
        f"GPU score update supports at most {MAX_GPU_CANDIDATES} candidates, "
        f"got {candidate_size}"
    )
  output = torch.empty_like(scores)
  _update_scores_kernel[batch_size, 1](
      cuda.as_cuda_array(scores.contiguous()),
      cuda.as_cuda_array(g_values.contiguous()),
      cuda.as_cuda_array(repeated_flags.contiguous()),
      cuda.as_cuda_array(output),
      int(candidate_size),
      int(num_keys),
  )
  return output


def compute_updated_scores_use_mimc_gpu(
    n_minus_1_grams: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    context_history: torch.Tensor,
    write_index: int,
    keys: Sequence[int],
) -> torch.Tensor:
  """Compute MiMC g-values, repetition masking, and score update in one kernel."""
  if not is_available():
    raise RuntimeError("Numba CUDA backend is not available")
  if scores.dtype != torch.float32:
    raise ValueError("Fused MiMC GPU score update supports float32 scores only")
  if not n_minus_1_grams.is_cuda or not indices.is_cuda or not scores.is_cuda:
    raise ValueError("Fused MiMC GPU score update requires CUDA tensors")
  if n_minus_1_grams.device != indices.device or scores.device != indices.device:
    raise ValueError("All fused MiMC GPU tensors must be on the same CUDA device")

  batch_size, sliding_window_size = n_minus_1_grams.shape
  _, candidate_token_size = indices.shape
  _, context_history_size, _ = context_history.shape
  num_keys = len(keys)
  if candidate_token_size > MAX_GPU_CANDIDATES:
    raise ValueError(
        f"Fused MiMC GPU WET supports at most {MAX_GPU_CANDIDATES} candidates, "
        f"got {candidate_token_size}"
    )
  if num_keys > 64:
    raise ValueError(f"Fused MiMC GPU WET supports at most 64 keys, got {num_keys}")

  device_index = n_minus_1_grams.device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  constants = _device_constant_arrays(device_index, keys)
  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = constants
  output = torch.empty_like(scores)
  _mimc_wet_fused_scores_kernel[batch_size, 256](
      cuda.as_cuda_array(n_minus_1_grams.contiguous()),
      cuda.as_cuda_array(indices.contiguous()),
      cuda.as_cuda_array(scores.contiguous()),
      cuda.as_cuda_array(output),
      cuda.as_cuda_array(context_history),
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(write_index),
      int(sliding_window_size),
      int(context_history_size),
      int(candidate_token_size),
      int(num_keys),
  )
  return output
