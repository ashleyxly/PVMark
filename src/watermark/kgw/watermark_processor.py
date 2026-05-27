# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt

import scipy.stats

import numpy as np
import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup

import functools
import os
import hash_rustlib
from enum import Enum

HASH_BIG_PRIME_HEX = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
_POSEIDON2_GPU_DISABLED = False


class HashMethod(Enum):
    TwoToOneFixed = 2
    ThreeToOneFixed = 4


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_compute_hash(left_input, right_input, hash_type):
    hash_result = hash_rustlib.single_two_inputs_hash_computation(str(left_input), str(right_input), int(hash_type))
    return hash_result

@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_id_use_two_to_one_hash_and_fixed_threshold(seed, vocab_size, greenlist_size, gamma, big_prime, hash_type):
    (greenlist_id, this_round_threshold) = hash_rustlib.rayon_get_greenlist_id_and_fixed_threshold_use_multi_two_inputs_hash(seed, vocab_size, greenlist_size, gamma, big_prime, hash_type)
    return (greenlist_id, this_round_threshold)

@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_id_use_three_to_one_hash_and_fixed_threshold(secret_key, pre_token_index, vocab_size, greenlist_size, gamma, big_prime, hash_type):
    (greenlist_id, this_round_threshold) = hash_rustlib.rayon_get_greenlist_id_and_fixed_threshold_use_multi_three_inputs_hash(int(secret_key), int(pre_token_index), vocab_size, greenlist_size, gamma, str(big_prime), int(hash_type))
    return (greenlist_id, this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_mask_use_two_to_one_hash_and_fixed_threshold(seed, vocab_size, gamma, big_prime, hash_type):
    (greenlist_mask, this_round_threshold) = hash_rustlib.rayon_get_greenlist_mask_fixed_threshold_use_multi_two_inputs_hash(str(seed), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_mask), this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_u32_use_two_to_one_hash_and_fixed_threshold(seed, vocab_size, gamma, big_prime, hash_type):
    (greenlist_ids, this_round_threshold) = hash_rustlib.rayon_get_greenlist_u32_fixed_threshold_use_multi_two_inputs_hash(str(seed), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_ids), this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_mask_use_two_to_one_hash_and_fixed_threshold_fused_seed(hash_key, pre_token_index, vocab_size, gamma, big_prime, hash_type):
    (greenlist_mask, this_round_threshold) = hash_rustlib.rayon_get_greenlist_mask_fixed_threshold_use_multi_two_inputs_hash_fused_seed(int(hash_key), int(pre_token_index), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_mask), this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_u32_use_two_to_one_hash_and_fixed_threshold_fused_seed(hash_key, pre_token_index, vocab_size, gamma, big_prime, hash_type):
    (greenlist_ids, this_round_threshold) = hash_rustlib.rayon_get_greenlist_u32_fixed_threshold_use_multi_two_inputs_hash_fused_seed(int(hash_key), int(pre_token_index), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_ids), this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_mask_use_three_to_one_hash_and_fixed_threshold(secret_key, pre_token_index, vocab_size, gamma, big_prime, hash_type):
    (greenlist_mask, this_round_threshold) = hash_rustlib.rayon_get_greenlist_mask_fixed_threshold_use_multi_three_inputs_hash(int(secret_key), int(pre_token_index), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_mask), this_round_threshold)


@functools.lru_cache(maxsize=1000)
def invoke_rustlib_get_greenlist_u32_use_three_to_one_hash_and_fixed_threshold(secret_key, pre_token_index, vocab_size, gamma, big_prime, hash_type):
    (greenlist_ids, this_round_threshold) = hash_rustlib.rayon_get_greenlist_u32_fixed_threshold_use_multi_three_inputs_hash(int(secret_key), int(pre_token_index), int(vocab_size), float(gamma), str(big_prime), int(hash_type))
    return (bytes(greenlist_ids), this_round_threshold)


def poseidon2_gpu_enabled() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU", "").strip().lower() in {"1", "true", "yes", "on", "native"}


def poseidon2_gpu_fused_enabled() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU_FUSED", "").strip().lower() in {"1", "true", "yes", "on"}


def poseidon2_gpu_mask_cache_enabled() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU_MASK_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def poseidon2_gpu_native_fused_enabled() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU_FUSED", "").strip().lower() == "native"


def poseidon2_gpu_native_greenlist_enabled() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU", "").strip().lower() == "native"


def rust_fixed_mask_enabled() -> bool:
    return os.environ.get("HASH_KGW_RUST_MASK", "").strip().lower() in {"1", "true", "yes", "on"}


def rust_fixed_u32_enabled() -> bool:
    return os.environ.get("HASH_KGW_RUST_U32", "1").strip().lower() not in {"0", "false", "no", "off"}


def _poseidon2_gpu_device_from_input(input_ids: torch.LongTensor):
    if not isinstance(input_ids, torch.Tensor):
        return None
    if input_ids.device.type != "cuda":
        return None
    return f"cuda:{input_ids.device.index or 0}"


def maybe_poseidon2_gpu_two_to_one_fixed(random_seed, vocab_size, gamma, big_prime, input_ids):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled():
        return None
    device = _poseidon2_gpu_device_from_input(input_ids)
    if device is None:
        return None
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return None
        greenlist_fn = (
            hash_kgw_poseidon2_gpu.get_greenlist_ids_two_to_one_fixed_native
            if poseidon2_gpu_native_greenlist_enabled()
            else hash_kgw_poseidon2_gpu.get_greenlist_ids_two_to_one_fixed_torch
        )
        return greenlist_fn(
            int(random_seed),
            int(vocab_size),
            float(gamma),
            str(big_prime),
            device,
        )
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU backend disabled after error: {exc}", flush=True)
        return None


def maybe_poseidon2_gpu_three_to_one_fixed(hash_key, prev_token, vocab_size, gamma, big_prime, input_ids):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled():
        return None
    device = _poseidon2_gpu_device_from_input(input_ids)
    if device is None:
        return None
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return None
        greenlist_fn = (
            hash_kgw_poseidon2_gpu.get_greenlist_ids_three_to_one_fixed_native
            if poseidon2_gpu_native_greenlist_enabled()
            else hash_kgw_poseidon2_gpu.get_greenlist_ids_three_to_one_fixed_torch
        )
        return greenlist_fn(
            int(hash_key),
            int(prev_token),
            int(vocab_size),
            float(gamma),
            str(big_prime),
            device,
        )
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU backend disabled after error: {exc}", flush=True)
        return None


def maybe_poseidon2_gpu_mask_two_to_one_fixed(random_seed, vocab_size, gamma, big_prime, input_ids):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled() or not poseidon2_gpu_mask_cache_enabled():
        return None
    device = _poseidon2_gpu_device_from_input(input_ids)
    if device is None:
        return None
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return None
        return hash_kgw_poseidon2_gpu.get_mask_two_to_one_fixed_cached(
            int(random_seed),
            int(vocab_size),
            float(gamma),
            str(big_prime),
            device,
        )
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU mask backend disabled after error: {exc}", flush=True)
        return None


def maybe_poseidon2_gpu_mask_three_to_one_fixed(hash_key, prev_token, vocab_size, gamma, big_prime, input_ids):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled() or not poseidon2_gpu_mask_cache_enabled():
        return None
    device = _poseidon2_gpu_device_from_input(input_ids)
    if device is None:
        return None
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return None
        return hash_kgw_poseidon2_gpu.get_mask_three_to_one_fixed_cached(
            int(hash_key),
            int(prev_token),
            int(vocab_size),
            float(gamma),
            str(big_prime),
            device,
        )
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU mask backend disabled after error: {exc}", flush=True)
        return None


def maybe_poseidon2_gpu_bias_logits_with_mask(mask, scores, delta):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled() or not poseidon2_gpu_mask_cache_enabled():
        return False
    if not isinstance(scores, torch.Tensor) or not scores.is_cuda:
        return False
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return False
        hash_kgw_poseidon2_gpu.bias_logits_with_mask(
            mask,
            scores,
            float(delta),
            f"cuda:{scores.device.index or 0}",
        )
        return True
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU mask backend disabled after error: {exc}", flush=True)
        return False


def maybe_poseidon2_gpu_bias_logits_two_to_one_fixed(random_seed, scores, delta, gamma, big_prime):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled():
        return False
    if not isinstance(scores, torch.Tensor) or not scores.is_cuda:
        return False
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return False
        bias_fn = (
            hash_kgw_poseidon2_gpu.bias_logits_two_to_one_fixed_native
            if poseidon2_gpu_native_fused_enabled()
            else hash_kgw_poseidon2_gpu.bias_logits_two_to_one_fixed_torch
        )
        bias_fn(
            int(random_seed),
            scores,
            float(delta),
            float(gamma),
            str(big_prime),
            f"cuda:{scores.device.index or 0}",
        )
        return True
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU backend disabled after error: {exc}", flush=True)
        return False


def maybe_poseidon2_gpu_bias_logits_three_to_one_fixed(hash_key, prev_token, scores, delta, gamma, big_prime):
    global _POSEIDON2_GPU_DISABLED
    if _POSEIDON2_GPU_DISABLED or not poseidon2_gpu_enabled():
        return False
    if not isinstance(scores, torch.Tensor) or not scores.is_cuda:
        return False
    try:
        from baseline_eval import hash_kgw_poseidon2_gpu

        if not hash_kgw_poseidon2_gpu.is_available():
            _POSEIDON2_GPU_DISABLED = True
            return False
        bias_fn = (
            hash_kgw_poseidon2_gpu.bias_logits_three_to_one_fixed_native
            if poseidon2_gpu_native_fused_enabled()
            else hash_kgw_poseidon2_gpu.bias_logits_three_to_one_fixed_torch
        )
        bias_fn(
            int(hash_key),
            int(prev_token),
            scores,
            float(delta),
            float(gamma),
            str(big_prime),
            f"cuda:{scores.device.index or 0}",
        )
        return True
    except Exception as exc:
        _POSEIDON2_GPU_DISABLED = True
        print(f"Poseidon2 GPU backend disabled after error: {exc}", flush=True)
        return False


@functools.lru_cache(maxsize=1000)
def fixed_threshold_int(gamma, big_prime=HASH_BIG_PRIME_HEX):
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    inverse_gamma = 1.0 / gamma
    if inverse_gamma <= 0.0:
        raise ValueError("invalid gamma")
    return int(str(big_prime), 16) // int(inverse_gamma)


@functools.lru_cache(maxsize=100000)
def is_green_token_two_to_one_fixed(hash_key, prev_token, curr_token, gamma, big_prime, hash_type):
    random_seed_hex = invoke_rustlib_compute_hash(int(hash_key), int(prev_token), int(hash_type))
    random_seed = str(int(random_seed_hex, 16))
    token_hash_hex = invoke_rustlib_compute_hash(random_seed, int(curr_token), int(hash_type))
    return int(token_hash_hex, 16) < fixed_threshold_int(float(gamma), str(big_prime))


@functools.lru_cache(maxsize=100000)
def is_green_token_three_to_one_fixed(hash_key, prev_token, curr_token, gamma, big_prime, hash_type):
    return bool(
        hash_rustlib.is_green_token_fixed_threshold_use_multi_three_inputs_hash(
            int(hash_key),
            int(prev_token),
            int(curr_token),
            float(gamma),
            str(big_prime),
            int(hash_type),
        )
    )


@functools.lru_cache(maxsize=10000)
def score_sequence_two_to_one_fixed(hash_key, token_ids, gamma, big_prime, hash_type):
    green_count, green_mask = hash_rustlib.score_sequence_fixed_threshold_use_multi_two_inputs_hash(
        int(hash_key),
        list(token_ids),
        float(gamma),
        str(big_prime),
        int(hash_type),
    )
    return int(green_count), list(green_mask)


@functools.lru_cache(maxsize=10000)
def score_sequence_three_to_one_fixed(hash_key, token_ids, gamma, big_prime, hash_type):
    green_count, green_mask = hash_rustlib.score_sequence_fixed_threshold_use_multi_three_inputs_hash(
        int(hash_key),
        list(token_ids),
        float(gamma),
        str(big_prime),
        int(hash_type),
    )
    return int(green_count), list(green_mask)



class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        hash_key: int = 2023,
        select_green_tokens: bool = True,
        hash_type: int = 3,
        hash_method: int = 2,
        greenlist_cache_size: int = 2048,
    ):

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens

        self.hash_type = hash_type
        self.hash_method = hash_method
        self.greenlist_cache_size = greenlist_cache_size
        self._greenlist_tensor_cache = collections.OrderedDict()
        if HashMethod(self.hash_method) not in (HashMethod.TwoToOneFixed, HashMethod.ThreeToOneFixed):
            raise ValueError("hash_method must be 2 (TwoToOneFixed) or 4 (ThreeToOneFixed).")

    def _token_to_int(self, token) -> int:
        if isinstance(token, torch.Tensor):
            return int(token.detach().item())
        return int(token)

    def _greenlist_tensor_from_list(self, greenlist_ids_list, input_ids: torch.LongTensor, cache_key) -> torch.LongTensor:
        device_key = str(input_ids.device)
        full_key = (*cache_key, device_key)
        cached = self._greenlist_tensor_cache.get(full_key)
        if cached is not None:
            self._greenlist_tensor_cache.move_to_end(full_key)
            return cached

        if isinstance(greenlist_ids_list, torch.Tensor):
            greenlist_ids = greenlist_ids_list.to(device=input_ids.device, dtype=torch.long)
        else:
            greenlist_ids = torch.tensor(greenlist_ids_list, dtype=torch.long, device=input_ids.device)
        if self.greenlist_cache_size > 0:
            self._greenlist_tensor_cache[full_key] = greenlist_ids
            self._greenlist_tensor_cache.move_to_end(full_key)
            while len(self._greenlist_tensor_cache) > self.greenlist_cache_size:
                self._greenlist_tensor_cache.popitem(last=False)
        return greenlist_ids

    def _cached_greenlist_tensor(self, input_ids: torch.LongTensor, cache_key):
        device_key = str(input_ids.device)
        full_key = (*cache_key, device_key)
        cached = self._greenlist_tensor_cache.get(full_key)
        if cached is not None:
            self._greenlist_tensor_cache.move_to_end(full_key)
        return cached

    def clear_greenlist_tensor_cache(self) -> None:
        self._greenlist_tensor_cache.clear()

    def _greenlist_mask_tensor_from_bytes(self, mask_bytes, input_ids: torch.LongTensor, cache_key) -> torch.BoolTensor:
        device_key = str(input_ids.device)
        full_key = (*cache_key, device_key)
        cached = self._greenlist_tensor_cache.get(full_key)
        if cached is not None:
            self._greenlist_tensor_cache.move_to_end(full_key)
            return cached

        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8, count=self.vocab_size).copy()
        greenlist_mask = torch.as_tensor(mask_np, device=input_ids.device).bool()
        if self.greenlist_cache_size > 0:
            self._greenlist_tensor_cache[full_key] = greenlist_mask
            self._greenlist_tensor_cache.move_to_end(full_key)
            while len(self._greenlist_tensor_cache) > self.greenlist_cache_size:
                self._greenlist_tensor_cache.popitem(last=False)
        return greenlist_mask

    def _cached_greenlist_mask_tensor(self, input_ids: torch.LongTensor, cache_key):
        return self._cached_greenlist_tensor(input_ids, cache_key)

    def _greenlist_tensor_from_u32_bytes(self, ids_bytes, input_ids: torch.LongTensor, cache_key) -> torch.LongTensor:
        device_key = str(input_ids.device)
        full_key = (*cache_key, device_key)
        cached = self._greenlist_tensor_cache.get(full_key)
        if cached is not None:
            self._greenlist_tensor_cache.move_to_end(full_key)
            return cached

        ids_np = np.frombuffer(ids_bytes, dtype=np.uint32).copy()
        greenlist_ids = torch.as_tensor(ids_np.astype(np.int64, copy=False), device=input_ids.device)
        if self.greenlist_cache_size > 0:
            self._greenlist_tensor_cache[full_key] = greenlist_ids
            self._greenlist_tensor_cache.move_to_end(full_key)
            while len(self._greenlist_tensor_cache) > self.greenlist_cache_size:
                self._greenlist_tensor_cache.popitem(last=False)
        return greenlist_ids

    def _is_green_token(self, prefix_ids: torch.LongTensor, curr_token) -> bool:
        hash_method = HashMethod(self.hash_method)
        curr_token_int = self._token_to_int(curr_token)
        if hash_method == HashMethod.TwoToOneFixed:
            prev_token = self._token_to_int(prefix_ids[-1])
            return bool(
                is_green_token_two_to_one_fixed(
                    int(self.hash_key),
                    prev_token,
                    curr_token_int,
                    float(self.gamma),
                    HASH_BIG_PRIME_HEX,
                    int(self.hash_type),
                )
            )
        if hash_method == HashMethod.ThreeToOneFixed:
            prev_token = self._token_to_int(prefix_ids[-1])
            return bool(
                is_green_token_three_to_one_fixed(
                    int(self.hash_key),
                    prev_token,
                    curr_token_int,
                    float(self.gamma),
                    HASH_BIG_PRIME_HEX,
                    int(self.hash_type),
                )
            )

        raise ValueError("hash_method must be 2 (TwoToOneFixed) or 4 (ThreeToOneFixed).")

    def _score_fixed_sequence_batch(self, input_ids: torch.LongTensor) -> tuple[int, list[bool]]:
        token_ids = tuple(int(token) for token in input_ids.detach().cpu().tolist())
        hash_method = HashMethod(self.hash_method)
        if hash_method == HashMethod.TwoToOneFixed:
            return score_sequence_two_to_one_fixed(
                int(self.hash_key),
                token_ids,
                float(self.gamma),
                HASH_BIG_PRIME_HEX,
                int(self.hash_type),
            )
        if hash_method == HashMethod.ThreeToOneFixed:
            return score_sequence_three_to_one_fixed(
                int(self.hash_key),
                token_ids,
                float(self.gamma),
                HASH_BIG_PRIME_HEX,
                int(self.hash_type),
            )
        raise NotImplementedError(f"Batch fixed scoring is not implemented for hash_method={hash_method}")

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return
    
    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        hash_type = self.hash_type
        hash_method = HashMethod(self.hash_method)

        if hash_method == HashMethod.TwoToOneFixed:
            return self._get_greenlist_ids_two_to_one_and_fixed_threshold(input_ids, int(hash_type))
        if hash_method == HashMethod.ThreeToOneFixed:
            return self._get_greenlist_ids_three_to_one_and_fixed_threshold(input_ids, int(hash_type))
        raise ValueError("hash_method must be 2 (TwoToOneFixed) or 4 (ThreeToOneFixed).")

    def _get_greenlist_mask_fixed_threshold(self, input_ids: torch.LongTensor) -> torch.BoolTensor | None:
        if not rust_fixed_mask_enabled():
            return None
        hash_type = int(self.hash_type)
        hash_method = HashMethod(self.hash_method)
        prev_token = self._token_to_int(input_ids[-1])
        BigPrime = HASH_BIG_PRIME_HEX

        if hash_method == HashMethod.TwoToOneFixed:
            cache_key = ("two_to_one_fixed", "rust_mask", hash_type, int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
            cached = self._cached_greenlist_mask_tensor(input_ids, cache_key)
            if cached is not None:
                return cached
            mask_bytes, _ = invoke_rustlib_get_greenlist_mask_use_two_to_one_hash_and_fixed_threshold_fused_seed(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                BigPrime,
                hash_type,
            )
            return self._greenlist_mask_tensor_from_bytes(mask_bytes, input_ids, cache_key)

        if hash_method == HashMethod.ThreeToOneFixed:
            cache_key = ("three_to_one_fixed", "rust_mask", hash_type, int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
            cached = self._cached_greenlist_mask_tensor(input_ids, cache_key)
            if cached is not None:
                return cached
            mask_bytes, _ = invoke_rustlib_get_greenlist_mask_use_three_to_one_hash_and_fixed_threshold(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                BigPrime,
                hash_type,
            )
            return self._greenlist_mask_tensor_from_bytes(mask_bytes, input_ids, cache_key)

        return None

    def _get_greenlist_ids_fixed_threshold_u32(self, input_ids: torch.LongTensor) -> torch.LongTensor | None:
        if not rust_fixed_u32_enabled():
            return None
        hash_type = int(self.hash_type)
        hash_method = HashMethod(self.hash_method)
        prev_token = self._token_to_int(input_ids[-1])
        BigPrime = HASH_BIG_PRIME_HEX

        if hash_method == HashMethod.TwoToOneFixed:
            cache_key = ("two_to_one_fixed", "rust_u32", hash_type, int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
            cached = self._cached_greenlist_tensor(input_ids, cache_key)
            if cached is not None:
                return cached
            ids_bytes, _ = invoke_rustlib_get_greenlist_u32_use_two_to_one_hash_and_fixed_threshold_fused_seed(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                BigPrime,
                hash_type,
            )
            return self._greenlist_tensor_from_u32_bytes(ids_bytes, input_ids, cache_key)

        if hash_method == HashMethod.ThreeToOneFixed:
            cache_key = ("three_to_one_fixed", "rust_u32", hash_type, int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
            cached = self._cached_greenlist_tensor(input_ids, cache_key)
            if cached is not None:
                return cached
            ids_bytes, _ = invoke_rustlib_get_greenlist_u32_use_three_to_one_hash_and_fixed_threshold(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                BigPrime,
                hash_type,
            )
            return self._greenlist_tensor_from_u32_bytes(ids_bytes, input_ids, cache_key)

        return None
    def _get_greenlist_ids_org(self, input_ids: torch.LongTensor) -> list[int]:
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if self.select_green_tokens:
            greenlist_ids = vocab_permutation[:greenlist_size]
        else:
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size) :]
        return greenlist_ids

    def _get_greenlist_ids_two_to_one_and_fixed_threshold(self, input_ids: torch.LongTensor, hash_type) -> list[int]:
        prev_token = self._token_to_int(input_ids[-1])
        greenlist_size = int(self.vocab_size * self.gamma)
        BigPrime = HASH_BIG_PRIME_HEX
        greenlist_ids_list = None
        cache_backend = "cpu"
        cpu_cache_key = ("two_to_one_fixed", "cpu", int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
        gpu_cache_key = ("two_to_one_fixed", "poseidon2_gpu", int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
        if int(hash_type) == 4:
            cached = self._cached_greenlist_tensor(input_ids, gpu_cache_key)
            if cached is not None:
                return cached
        cached = self._cached_greenlist_tensor(input_ids, cpu_cache_key)
        if cached is not None:
            return cached
        random_seed_hex = invoke_rustlib_compute_hash(self.hash_key, prev_token, int(hash_type))
        random_seed = str(int(random_seed_hex, 16))
        if int(hash_type) == 4:
            greenlist_ids_list = maybe_poseidon2_gpu_two_to_one_fixed(
                random_seed,
                self.vocab_size,
                self.gamma,
                BigPrime,
                input_ids,
            )
            if greenlist_ids_list is not None:
                cache_backend = "poseidon2_gpu"
        if greenlist_ids_list is None:
            (greenlist_ids_list, this_round_threshold) = invoke_rustlib_get_greenlist_id_use_two_to_one_hash_and_fixed_threshold(str(random_seed), self.vocab_size, greenlist_size, self.gamma, str(BigPrime), int(hash_type))
        greenlist_ids = self._greenlist_tensor_from_list(
            greenlist_ids_list,
            input_ids,
            ("two_to_one_fixed", cache_backend, int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token),
        )
        return greenlist_ids

    def _get_greenlist_ids_three_to_one_and_fixed_threshold(self, input_ids: torch.LongTensor, hash_type) -> list[int]:
        prev_token = self._token_to_int(input_ids[-1])
        greenlist_size = int(self.vocab_size * self.gamma)
        BigPrime = HASH_BIG_PRIME_HEX
        greenlist_ids_list = None
        cache_backend = "cpu"
        cpu_cache_key = ("three_to_one_fixed", "cpu", int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
        gpu_cache_key = ("three_to_one_fixed", "poseidon2_gpu", int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token)
        if int(hash_type) == 4:
            cached = self._cached_greenlist_tensor(input_ids, gpu_cache_key)
            if cached is not None:
                return cached
            greenlist_ids_list = maybe_poseidon2_gpu_three_to_one_fixed(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                BigPrime,
                input_ids,
            )
            if greenlist_ids_list is not None:
                cache_backend = "poseidon2_gpu"
        if greenlist_ids_list is None:
            cached = self._cached_greenlist_tensor(input_ids, cpu_cache_key)
            if cached is not None:
                return cached
            (greenlist_ids_list, this_round_threshold) = invoke_rustlib_get_greenlist_id_use_three_to_one_hash_and_fixed_threshold(int(self.hash_key), prev_token, self.vocab_size, greenlist_size, self.gamma, str(BigPrime), int(hash_type))
        greenlist_ids = self._greenlist_tensor_from_list(
            greenlist_ids_list,
            input_ids,
            ("three_to_one_fixed", cache_backend, int(hash_type), int(self.hash_key), float(self.gamma), self.vocab_size, prev_token),
        )
        return greenlist_ids


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _bias_greenlist_logits_inplace(self, scores: torch.Tensor, greenlist_token_ids, greenlist_bias: float) -> torch.Tensor:
        for b_idx, greenlist_ids in enumerate(greenlist_token_ids):
            scores[b_idx, greenlist_ids] = scores[b_idx, greenlist_ids] + greenlist_bias
        return scores

    def _try_rust_mask_bias_logits(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if not rust_fixed_mask_enabled():
            return False
        hash_method = HashMethod(self.hash_method)
        if hash_method not in (HashMethod.TwoToOneFixed, HashMethod.ThreeToOneFixed):
            return False
        if scores.shape[0] != input_ids.shape[0]:
            return False

        masks = []
        for b_idx in range(input_ids.shape[0]):
            mask = self._get_greenlist_mask_fixed_threshold(input_ids[b_idx])
            if mask is None:
                return False
            masks.append(mask)

        if scores.shape[0] == 1:
            scores[0, masks[0]] = scores[0, masks[0]] + self.delta
        else:
            greenlist_mask = torch.stack(masks, dim=0)
            scores[greenlist_mask] = scores[greenlist_mask] + self.delta
        return True

    def _try_rust_u32_bias_logits(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if not rust_fixed_u32_enabled():
            return False
        hash_method = HashMethod(self.hash_method)
        if hash_method not in (HashMethod.TwoToOneFixed, HashMethod.ThreeToOneFixed):
            return False
        if scores.shape[0] != input_ids.shape[0]:
            return False

        greenlist_ids = []
        for b_idx in range(input_ids.shape[0]):
            ids = self._get_greenlist_ids_fixed_threshold_u32(input_ids[b_idx])
            if ids is None:
                return False
            greenlist_ids.append(ids)
        self._bias_greenlist_logits_inplace(scores, greenlist_ids, self.delta)
        return True

    def _try_poseidon2_gpu_bias_logits(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if int(self.hash_type) != 4:
            return False
        if not (poseidon2_gpu_fused_enabled() or poseidon2_gpu_native_fused_enabled()):
            return False
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            return False
        hash_method = HashMethod(self.hash_method)
        prev_token = self._token_to_int(input_ids[0, -1])
        score_row = scores[0]
        if hash_method == HashMethod.TwoToOneFixed:
            random_seed_hex = invoke_rustlib_compute_hash(self.hash_key, prev_token, 4)
            random_seed = int(random_seed_hex, 16)
            return maybe_poseidon2_gpu_bias_logits_two_to_one_fixed(
                random_seed,
                score_row,
                self.delta,
                self.gamma,
                HASH_BIG_PRIME_HEX,
            )
        if hash_method == HashMethod.ThreeToOneFixed:
            return maybe_poseidon2_gpu_bias_logits_three_to_one_fixed(
                int(self.hash_key),
                prev_token,
                score_row,
                self.delta,
                self.gamma,
                HASH_BIG_PRIME_HEX,
            )
        return False

    def _try_poseidon2_gpu_mask_cache_bias_logits(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        if int(self.hash_type) != 4:
            return False
        if not poseidon2_gpu_mask_cache_enabled():
            return False
        if input_ids.shape[0] != 1 or scores.shape[0] != 1:
            return False
        hash_method = HashMethod(self.hash_method)
        prev_token = self._token_to_int(input_ids[0, -1])
        score_row = scores[0]
        if hash_method == HashMethod.TwoToOneFixed:
            random_seed_hex = invoke_rustlib_compute_hash(self.hash_key, prev_token, 4)
            random_seed = int(random_seed_hex, 16)
            mask = maybe_poseidon2_gpu_mask_two_to_one_fixed(
                random_seed,
                self.vocab_size,
                self.gamma,
                HASH_BIG_PRIME_HEX,
                input_ids,
            )
            if mask is None:
                return False
            return maybe_poseidon2_gpu_bias_logits_with_mask(mask, score_row, self.delta)
        if hash_method == HashMethod.ThreeToOneFixed:
            mask = maybe_poseidon2_gpu_mask_three_to_one_fixed(
                int(self.hash_key),
                prev_token,
                self.vocab_size,
                self.gamma,
                HASH_BIG_PRIME_HEX,
                input_ids,
            )
            if mask is None:
                return False
            return maybe_poseidon2_gpu_bias_logits_with_mask(mask, score_row, self.delta)
        return False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        if self._try_poseidon2_gpu_mask_cache_bias_logits(input_ids, scores):
            return scores

        if self._try_poseidon2_gpu_bias_logits(input_ids, scores):
            return scores

        if self._try_rust_u32_bias_logits(input_ids, scores):
            return scores

        if self._try_rust_mask_bias_logits(input_ids, scores):
            return scores

        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            greenlist_ids = self._get_greenlist_ids(input_ids[b_idx])
            batched_greenlist_ids[b_idx] = greenlist_ids

        scores = self._bias_greenlist_logits_inplace(
            scores=scores,
            greenlist_token_ids=batched_greenlist_ids,
            greenlist_bias=self.delta,
        )
        return scores


class WatermarkDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
        ignore_repeated_bigrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1":
            self.min_prefix_len = 1
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        try:
            if self.ignore_repeated_bigrams:
                assert return_green_token_mask is False, "Can't return the green/red mask when ignoring repeats."
                bigram_table = {}
                token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
                freq = collections.Counter(token_bigram_generator)
                num_tokens_scored = len(freq.keys())
                for idx, bigram in enumerate(freq.keys()):
                    prefix = torch.tensor([bigram[0]], device=self.device)  # expects a 1-d prefix tensor on the randperm device
                    bigram_table[bigram] = self._is_green_token(prefix, bigram[1])
                green_token_count = sum(bigram_table.values())
            else:
                num_tokens_scored = len(input_ids) - self.min_prefix_len
                if num_tokens_scored < 1:
                    raise ValueError(
                        (
                            f"Must have at least {1} token to score after "
                            f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."
                        )
                    )
                if HashMethod(self.hash_method) in (HashMethod.TwoToOneFixed, HashMethod.ThreeToOneFixed):
                    green_token_count, green_token_mask = self._score_fixed_sequence_batch(input_ids)
                else:
                    green_token_count, green_token_mask = 0, []
                    for idx in range(self.min_prefix_len, len(input_ids)):
                        curr_token = input_ids[idx]
                        if self._is_green_token(input_ids[:idx], curr_token):
                            green_token_count += 1
                            green_token_mask.append(True)
                        else:
                            green_token_mask.append(False)
        except ValueError as e:
            print(f"ValueError: {e}")
            return dict(num_tokens_scored=-100, num_green_tokens=-100, green_fraction=-100, z_score=-100, p_value=-100, green_token_mask=[])

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]

        output_dict = {}
        score_dict = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict
