"""Optional CUDA C++ extension for batched MiMC WET replay."""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.cpp_extension import load

from synthid_text import gpu_hash


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _load_extension():
  cuda_home = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
  cuda_bin = str(Path(cuda_home) / "bin")
  path = os.environ.get("PATH", "")
  if cuda_bin not in path.split(os.pathsep):
    os.environ["PATH"] = cuda_bin + os.pathsep + path
  os.environ.setdefault("CUDA_HOME", cuda_home)
  os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

  return load(
      name="synthid_cuda_hash_ext",
      sources=[
          str(_repo_root() / "src" / "synthid_text" / "cuda_hash_ext.cpp"),
          str(_repo_root() / "src" / "synthid_text" / "cuda_hash_ext_kernel.cu"),
      ],
      extra_cflags=["-O3"],
      extra_cuda_cflags=[
          "-O3",
          "-lineinfo",
      ],
      verbose=bool(int(os.environ.get("SYNTHID_CUDA_EXT_VERBOSE", "0"))),
  )


_DEVICE_CONSTANTS: dict[tuple[int, tuple[int, ...]], tuple[torch.Tensor, ...]] = {}


def _torch_constant_arrays(device: torch.device, keys: Sequence[int]) -> tuple[torch.Tensor, ...]:
  keys_tuple = tuple(int(value) for value in keys)
  device_index = device.index
  if device_index is None:
    device_index = torch.cuda.current_device()
  cache_key = (int(device_index), keys_tuple)
  if cache_key not in _DEVICE_CONSTANTS:
    host_arrays = gpu_hash._host_constant_arrays(keys_tuple)  # pylint: disable=protected-access
    _DEVICE_CONSTANTS[cache_key] = tuple(
        torch.as_tensor(array, device=device, dtype=torch.uint32).contiguous()
        for array in host_arrays
    )
  return _DEVICE_CONSTANTS[cache_key]


def is_available() -> bool:
  return torch.cuda.is_available()


def compute_batched_updated_scores_use_mimc_cpp(
    contexts: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    keys: Sequence[int],
    num_steps: int,
    batch_size: int,
    context_history_size: int,
    return_context_tensor: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
  """Compute batched MiMC replay scores with the CUDA C++ extension."""
  if not is_available():
    raise RuntimeError("CUDA is not available")
  if not contexts.is_cuda or not indices.is_cuda or not scores.is_cuda:
    raise ValueError("CUDA C++ batched WET requires CUDA tensors")
  if contexts.device != indices.device or contexts.device != scores.device:
    raise ValueError("contexts, indices, and scores must be on the same CUDA device")
  if scores.dtype != torch.float32:
    raise ValueError("CUDA C++ batched WET currently supports float32 scores")

  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = (
      _torch_constant_arrays(contexts.device, keys)
  )
  extension = _load_extension()
  output, context_hashes = extension.mimc_batched_wet_update(
      contexts.contiguous(),
      indices.contiguous(),
      scores.contiguous(),
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(num_steps),
      int(batch_size),
      int(context_history_size),
  )
  if return_context_tensor:
    return output, context_hashes
  return output


def debug_batched_mimc_cpp(
    contexts: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    keys: Sequence[int],
    num_steps: int,
    batch_size: int,
    context_history_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return context hashes, g-values, and repetition flags from C++ kernels."""
  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = (
      _torch_constant_arrays(contexts.device, keys)
  )
  extension = _load_extension()
  context_hashes, g_values, repeated_flags = extension.mimc_batched_wet_debug(
      contexts.contiguous(),
      indices.contiguous(),
      scores.contiguous(),
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(num_steps),
      int(batch_size),
      int(context_history_size),
  )
  return context_hashes, g_values, repeated_flags


def update_batched_scores_cpp(
    context_hashes: torch.Tensor,
    g_values: torch.Tensor,
    scores: torch.Tensor,
    num_steps: int,
    batch_size: int,
    context_history_size: int,
) -> torch.Tensor:
  """Run batched repetition check and score update with the CUDA C++ extension."""
  extension = _load_extension()
  return extension.batched_score_update(
      context_hashes.contiguous(),
      g_values.contiguous(),
      scores.contiguous(),
      int(num_steps),
      int(batch_size),
      int(context_history_size),
  )


def compute_online_updated_scores_use_mimc_cpp(
    contexts: torch.Tensor,
    indices: torch.Tensor,
    scores: torch.Tensor,
    context_history: torch.Tensor,
    write_index: int,
    keys: Sequence[int],
) -> torch.Tensor:
  """Compute one true-online MiMC SynthID WET step in a fused CUDA C++ kernel."""
  if not is_available():
    raise RuntimeError("CUDA is not available")
  if not (
      contexts.is_cuda
      and indices.is_cuda
      and scores.is_cuda
      and context_history.is_cuda
  ):
    raise ValueError("CUDA C++ online WET requires CUDA tensors")
  if (
      contexts.device != indices.device
      or contexts.device != scores.device
      or contexts.device != context_history.device
  ):
    raise ValueError("contexts, indices, scores, and history must share a CUDA device")
  if scores.dtype != torch.float32:
    raise ValueError("CUDA C++ online WET currently supports float32 scores")
  if context_history.dtype != torch.int32:
    raise ValueError("context_history must be an int32 limb tensor")
  if not context_history.is_contiguous():
    raise ValueError("context_history must be contiguous for in-place CUDA update")

  prime, half_prime, r2, one, round_keys, keys_mont, key_index_mont = (
      _torch_constant_arrays(contexts.device, keys)
  )
  extension = _load_extension()
  return extension.mimc_online_wet_update(
      contexts.contiguous(),
      indices.contiguous(),
      scores.contiguous(),
      context_history,
      prime,
      half_prime,
      r2,
      one,
      round_keys,
      keys_mont,
      key_index_mont,
      int(write_index),
  )
