from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

BN254_PRIME_HEX = "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
BN254_PRIME = int(BN254_PRIME_HEX, 16)
LIMBS = 8
WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
MONT_R = pow(1 << WORD_BITS, LIMBS, BN254_PRIME)
MONT_R2 = (MONT_R * MONT_R) % BN254_PRIME
MONT_N0 = (-pow(BN254_PRIME, -1, 1 << WORD_BITS)) & WORD_MASK

REPO_ROOT = Path(__file__).resolve().parents[1]
POSEIDON2_T2_SOURCE = REPO_ROOT / "artifact/hash_function/poseidon2/plain_implementations/src/poseidon2/poseidon2_instance_bn256_t_2.rs"
POSEIDON2_T3_SOURCE = REPO_ROOT / "artifact/hash_function/poseidon2/plain_implementations/src/poseidon2/poseidon2_instance_bn256.rs"
POSEIDON_FAST_W3_SOURCE = REPO_ROOT / "artifact/hash_function/hash-function/src/poseidon_fast/poseidon_params_width_3.rs"
POSEIDON_FAST_W4_SOURCE = REPO_ROOT / "artifact/hash_function/hash-function/src/poseidon_fast/poseidon_params_width_4.rs"
MIMC_BN254_SOURCE = REPO_ROOT / "artifact/hash_function/arkworks-mimc/src/params/mimc_7_91_bn254.rs"
CUDA_KERNEL_SOURCE = REPO_ROOT / "baseline_eval/hash_kgw_poseidon2_cuda_kernel.cu"
CUDA_KERNEL_SO = Path(os.environ.get("HASH_KGW_POSEIDON2_CUDA_SO", "/tmp/hash_kgw_cuda_build/libhash_kgw_poseidon2.so"))

_CUDA_MODULES: tuple[Any, Any] | None = None
_DEVICE_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_POSEIDON_FAST_DEVICE_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_MIMC_DEVICE_CACHE: dict[int, dict[str, Any]] = {}
_THRESHOLD_CACHE: dict[tuple[int, float, str], Any] = {}
_CUDA_KERNEL_LIB: Any | None = None
_NATIVE_SCORE_POOL: dict[tuple[int, int], Any] = {}
_MASK_CACHE: dict[tuple[Any, ...], Any] = {}
_TOKEN_MONT_CACHE: dict[tuple[int, int], Any] = {}


def _cuda_modules() -> tuple[Any, Any]:
    global _CUDA_MODULES
    if _CUDA_MODULES is None:
        from numba import cuda, uint32, uint64

        _CUDA_MODULES = (cuda, (uint32, uint64))
    return _CUDA_MODULES


def is_available() -> bool:
    try:
        cuda, _ = _cuda_modules()
        return bool(cuda.is_available())
    except Exception:
        return False


def select_device(device: str | int | None = None) -> None:
    if device is None:
        return
    cuda, _ = _cuda_modules()
    if isinstance(device, str):
        if not device.startswith("cuda"):
            return
        _, _, index = device.partition(":")
        if not index:
            return
        device_id = int(index)
    else:
        device_id = int(device)
    cuda.select_device(device_id)


def _device_id(device: str | int | None = None) -> int:
    if device is None:
        cuda, _ = _cuda_modules()
        return int(cuda.get_current_device().id)
    if isinstance(device, str):
        if not device.startswith("cuda"):
            raise ValueError(f"expected a CUDA device, got {device!r}")
        _, _, index = device.partition(":")
        return int(index) if index else 0
    return int(device)


def _select_torch_cuda_device(device: str | int | None = None) -> None:
    import torch

    torch.cuda.set_device(_device_id(device))


def gamma_to_u64(gamma: float) -> int:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    inverse_gamma = 1.0 / gamma
    if inverse_gamma >= float((1 << 64) - 1) or inverse_gamma <= 0.0:
        raise ValueError("invalid gamma")
    return int(inverse_gamma)


def fixed_threshold_limbs(gamma: float, big_prime_hex: str = BN254_PRIME_HEX) -> np.ndarray:
    threshold = int(str(big_prime_hex), 16) // gamma_to_u64(float(gamma))
    return int_to_limbs(threshold)


def int_to_limbs(value: int) -> np.ndarray:
    value %= BN254_PRIME
    return np.array([(value >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)], dtype=np.uint32)


def raw_int_to_limbs(value: int) -> np.ndarray:
    return np.array([(int(value) >> (WORD_BITS * i)) & WORD_MASK for i in range(LIMBS)], dtype=np.uint32)


def int_to_mont_limbs(value: int) -> np.ndarray:
    return int_to_limbs((value % BN254_PRIME) * MONT_R % BN254_PRIME)


def limbs_to_int(limbs: np.ndarray) -> int:
    total = 0
    for index, value in enumerate(limbs.tolist()):
        total += int(value) << (WORD_BITS * index)
    return total


def _extract_round_constants(source: Path, static_name: str, width: int) -> np.ndarray:
    text = source.read_text(encoding="utf-8")
    start = text.index(f"pub static ref {static_name}")
    end = text.index("pub static ref POSEIDON", start)
    constants = [int(item, 16) for item in re.findall(r'from_hex\("0x([0-9a-fA-F]+)"\)', text[start:end])]
    expected = 64 * width
    if len(constants) != expected:
        raise ValueError(f"{source} has {len(constants)} {static_name} constants, expected {expected}")
    mont_constants = np.zeros((64, width, LIMBS), dtype=np.uint32)
    for round_index in range(64):
        for state_index in range(width):
            mont_constants[round_index, state_index] = int_to_mont_limbs(constants[round_index * width + state_index])
    return mont_constants.reshape(-1)


def _from_raw_u64_array_to_int(raw_words: list[str]) -> int:
    if len(raw_words) != 4:
        raise ValueError(f"expected 4 u64 limbs, got {len(raw_words)}")
    value = 0
    for index, item in enumerate(raw_words):
        value += int(item.replace("_", ""), 16) << (64 * index)
    return value


def _extract_fp_from_raw_values(block: str) -> list[int]:
    values: list[int] = []
    pattern = re.compile(r"Fp::from_raw\(\s*\[\s*(.*?)\s*\]\s*\)", re.DOTALL)
    for match in pattern.finditer(block):
        raw_words = re.findall(r"0x[0-9a-fA-F_]+", match.group(1))
        values.append(_from_raw_u64_array_to_int(raw_words))
    return values


def _extract_poseidon_fast_round_constants(source: Path, width: int) -> np.ndarray:
    text = source.read_text(encoding="utf-8")
    start = text.index("pub(crate) const ROUND_CONSTANTS")
    end = text.index("// MDS matrix:", start)
    constants = _extract_fp_from_raw_values(text[start:end])
    expected = 64 * width
    if len(constants) != expected:
        raise ValueError(f"{source} has {len(constants)} Poseidon constants, expected {expected}")
    mont_constants = np.zeros((64, width, LIMBS), dtype=np.uint32)
    for round_index in range(64):
        for state_index in range(width):
            mont_constants[round_index, state_index] = int_to_mont_limbs(constants[round_index * width + state_index])
    return mont_constants.reshape(-1)


def _extract_poseidon_fast_mds(source: Path, width: int) -> np.ndarray:
    text = source.read_text(encoding="utf-8")
    start = text.index("pub(crate) const MDS")
    end = text.index("pub(crate) const MDS_INV", start)
    constants = _extract_fp_from_raw_values(text[start:end])
    expected = width * width
    if len(constants) != expected:
        raise ValueError(f"{source} has {len(constants)} Poseidon MDS constants, expected {expected}")
    mont_constants = np.zeros((width, width, LIMBS), dtype=np.uint32)
    for row in range(width):
        for col in range(width):
            mont_constants[row, col] = int_to_mont_limbs(constants[row * width + col])
    return mont_constants.reshape(-1)


def _extract_mimc_round_keys(source: Path) -> np.ndarray:
    text = source.read_text(encoding="utf-8")
    start = text.index("pub const MIMC_7_91_BN254_ROUND_KEYS")
    end = text.index("];", start)
    constants = [int(item) for item in re.findall(r'"([0-9]+)"', text[start:end])]
    if len(constants) != 91:
        raise ValueError(f"{source} has {len(constants)} MiMC round keys, expected 91")
    mont_constants = np.zeros((91, LIMBS), dtype=np.uint32)
    for index, value in enumerate(constants):
        mont_constants[index] = int_to_mont_limbs(value)
    return mont_constants.reshape(-1)


@lru_cache(maxsize=2)
def _round_constants_flat(width: int) -> np.ndarray:
    if width == 2:
        return _extract_round_constants(POSEIDON2_T2_SOURCE, "RC2", 2)
    if width == 3:
        return _extract_round_constants(POSEIDON2_T3_SOURCE, "RC3", 3)
    raise ValueError(f"unsupported Poseidon2 width: {width}")


@lru_cache(maxsize=2)
def _poseidon_fast_round_constants_flat(width: int) -> np.ndarray:
    if width == 3:
        return _extract_poseidon_fast_round_constants(POSEIDON_FAST_W3_SOURCE, 3)
    if width == 4:
        return _extract_poseidon_fast_round_constants(POSEIDON_FAST_W4_SOURCE, 4)
    raise ValueError(f"unsupported Poseidon fast width: {width}")


@lru_cache(maxsize=2)
def _poseidon_fast_mds_flat(width: int) -> np.ndarray:
    if width == 3:
        return _extract_poseidon_fast_mds(POSEIDON_FAST_W3_SOURCE, 3)
    if width == 4:
        return _extract_poseidon_fast_mds(POSEIDON_FAST_W4_SOURCE, 4)
    raise ValueError(f"unsupported Poseidon fast width: {width}")


@lru_cache(maxsize=1)
def _mimc_round_keys_flat() -> np.ndarray:
    return _extract_mimc_round_keys(MIMC_BN254_SOURCE)


def _device_arrays(width: int, device: str | int | None = None) -> dict[str, Any]:
    select_device(device)
    cuda, _ = _cuda_modules()
    device_id = int(cuda.get_current_device().id)
    key = (device_id, width)
    cached = _DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    arrays = {
        "p": cuda.to_device(raw_int_to_limbs(BN254_PRIME)),
        "r2": cuda.to_device(int_to_limbs(MONT_R2)),
        "one": cuda.to_device(int_to_limbs(1)),
        "rc": cuda.to_device(_round_constants_flat(width)),
    }
    _DEVICE_CACHE[key] = arrays
    return arrays


def _poseidon_fast_device_arrays(width: int, device: str | int | None = None) -> dict[str, Any]:
    select_device(device)
    cuda, _ = _cuda_modules()
    device_id = int(cuda.get_current_device().id)
    key = (device_id, width)
    cached = _POSEIDON_FAST_DEVICE_CACHE.get(key)
    if cached is not None:
        return cached
    if width == 3:
        domain_value = 2 << 64
    elif width == 4:
        domain_value = 3 << 64
    else:
        raise ValueError(f"unsupported Poseidon fast width: {width}")
    arrays = {
        "p": cuda.to_device(raw_int_to_limbs(BN254_PRIME)),
        "r2": cuda.to_device(int_to_limbs(MONT_R2)),
        "one": cuda.to_device(int_to_limbs(1)),
        "domain": cuda.to_device(int_to_mont_limbs(domain_value)),
        "rc": cuda.to_device(_poseidon_fast_round_constants_flat(width)),
        "mds": cuda.to_device(_poseidon_fast_mds_flat(width)),
    }
    _POSEIDON_FAST_DEVICE_CACHE[key] = arrays
    return arrays


def _mimc_device_arrays(device: str | int | None = None) -> dict[str, Any]:
    select_device(device)
    cuda, _ = _cuda_modules()
    device_id = int(cuda.get_current_device().id)
    cached = _MIMC_DEVICE_CACHE.get(device_id)
    if cached is not None:
        return cached
    arrays = {
        "one": cuda.to_device(int_to_limbs(1)),
        "round_keys": cuda.to_device(_mimc_round_keys_flat()),
    }
    _MIMC_DEVICE_CACHE[device_id] = arrays
    return arrays


def _threshold_array(gamma: float, big_prime_hex: str, device: str | int | None = None):
    select_device(device)
    cuda, _ = _cuda_modules()
    device_id = int(cuda.get_current_device().id)
    key = (device_id, float(gamma), str(big_prime_hex))
    cached = _THRESHOLD_CACHE.get(key)
    if cached is not None:
        return cached
    threshold = cuda.to_device(fixed_threshold_limbs(gamma, big_prime_hex))
    _THRESHOLD_CACHE[key] = threshold
    return threshold


def _torch_stream_ptr():
    import torch

    stream = torch.cuda.current_stream()
    return getattr(stream, "cuda_stream", None)


def _cuda_kernel_lib():
    global _CUDA_KERNEL_LIB
    if _CUDA_KERNEL_LIB is not None:
        return _CUDA_KERNEL_LIB

    import ctypes

    so_path = CUDA_KERNEL_SO
    if not so_path.exists():
        so_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_arch = os.environ.get("HASH_KGW_CUDA_ARCH", "86")
    compile_cmd = [
        "/usr/local/cuda/bin/nvcc",
        "-O3",
        "--use_fast_math",
        "-Xptxas",
        "-O3",
        "-gencode",
        f"arch=compute_{cuda_arch},code=sm_{cuda_arch}",
        "-gencode",
        f"arch=compute_{cuda_arch},code=compute_{cuda_arch}",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-o",
        str(so_path),
        str(CUDA_KERNEL_SOURCE),
    ]
    build_info_path = so_path.with_suffix(so_path.suffix + ".cmd")
    build_info = "\n".join(compile_cmd)
    old_build_info = build_info_path.read_text(encoding="utf-8") if build_info_path.exists() else ""
    needs_rebuild = (
        (not so_path.exists())
        or CUDA_KERNEL_SOURCE.stat().st_mtime > so_path.stat().st_mtime
        or old_build_info != build_info
        or os.environ.get("HASH_KGW_FORCE_CUDA_REBUILD") == "1"
    )
    if needs_rebuild:
        subprocess.run(compile_cmd, check=True)
        build_info_path.write_text(build_info, encoding="utf-8")
    lib = ctypes.CDLL(str(so_path))
    c_u32 = ctypes.c_uint32
    c_int = ctypes.c_int
    c_float = ctypes.c_float
    c_void_p = ctypes.c_void_p
    lib.poseidon2_t2_bias.argtypes = [
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_float,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t2_bias.restype = c_int
    lib.poseidon2_t2_bias_precomputed.argtypes = [
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_float,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t2_bias_precomputed.restype = c_int
    lib.poseidon2_t3_bias.argtypes = [
        c_u32,
        c_u32,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_float,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t3_bias.restype = c_int
    lib.poseidon2_t3_bias_precomputed.argtypes = [
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_u32,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_float,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t3_bias_precomputed.restype = c_int
    lib.poseidon2_t2_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t2_masks_precomputed.restype = c_int
    lib.poseidon2_t3_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon2_t3_masks_precomputed.restype = c_int
    lib.poseidon_fast_w3_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon_fast_w3_masks_precomputed.restype = c_int
    lib.poseidon_fast_w4_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.poseidon_fast_w4_masks_precomputed.restype = c_int
    lib.mimc_t2_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.mimc_t2_masks_precomputed.restype = c_int
    lib.mimc_t3_masks_precomputed.argtypes = [
        c_int,
        c_int,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_u32,
        c_void_p,
        c_void_p,
    ]
    lib.mimc_t3_masks_precomputed.restype = c_int
    _CUDA_KERNEL_LIB = lib
    return lib


def native_cuda_available() -> bool:
    try:
        return CUDA_KERNEL_SO.exists() or CUDA_KERNEL_SOURCE.exists()
    except Exception:
        return False


def _native_score_buffer(vocab_size: int, device: str | int | None = None):
    import torch

    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    key = (device_id, int(vocab_size))
    cached = _NATIVE_SCORE_POOL.get(key)
    if cached is not None:
        return cached
    scores = torch.empty(int(vocab_size), dtype=torch.float32, device=f"cuda:{device_id}")
    _NATIVE_SCORE_POOL[key] = scores
    return scores


def _token_mont_table(vocab_size: int, device: str | int | None = None):
    select_device(device)
    cuda, _ = _cuda_modules()
    device_id = int(cuda.get_current_device().id)
    key = (device_id, int(vocab_size))
    cached = _TOKEN_MONT_CACHE.get(key)
    if cached is not None:
        return cached
    table = np.empty((int(vocab_size), LIMBS), dtype=np.uint32)
    for token_id in range(int(vocab_size)):
        table[token_id] = int_to_mont_limbs(token_id)
    token_mont = cuda.to_device(table.reshape(-1))
    _TOKEN_MONT_CACHE[key] = token_mont
    return token_mont


def _limb_table(values: list[int] | tuple[int, ...], device: str | int | None = None):
    select_device(device)
    cuda, _ = _cuda_modules()
    if not values:
        raise ValueError("values must not be empty")
    table = np.empty((len(values), LIMBS), dtype=np.uint32)
    for index, value in enumerate(values):
        table[index] = int_to_mont_limbs(int(value))
    return cuda.to_device(table.reshape(-1))


def _launch_two_to_one(seed: int, vocab_size: int, gamma: float, big_prime_hex: str, device: str | int | None) -> np.ndarray:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    cuda, _ = _cuda_modules()
    arrays = _device_arrays(2, device)
    seed_mont = cuda.to_device(int_to_mont_limbs(seed))
    threshold = _threshold_array(gamma, big_prime_hex, device)
    mask = cuda.device_array(int(vocab_size), dtype=np.uint8)
    threads = 128
    blocks = (int(vocab_size) + threads - 1) // threads
    _poseidon2_t2_fixed_kernel[blocks, threads](
        seed_mont,
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        mask,
    )
    return mask.copy_to_host()


def _launch_two_to_one_torch_mask(seed: int, vocab_size: int, gamma: float, big_prime_hex: str, device: str | int | None):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import torch

    cuda, _ = _cuda_modules()
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(2, device)
    seed_mont = cuda.to_device(int_to_mont_limbs(seed))
    threshold = _threshold_array(gamma, big_prime_hex, device)
    mask = torch.empty(int(vocab_size), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    mask_array = cuda.as_cuda_array(mask)
    threads = 128
    blocks = (int(vocab_size) + threads - 1) // threads
    _poseidon2_t2_fixed_kernel[blocks, threads](
        seed_mont,
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        mask_array,
    )
    return mask


def _launch_three_to_one(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str,
    device: str | int | None,
) -> np.ndarray:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    cuda, _ = _cuda_modules()
    arrays = _device_arrays(3, device)
    secret_mont = cuda.to_device(int_to_mont_limbs(secret_key))
    previous_mont = cuda.to_device(int_to_mont_limbs(previous_token))
    threshold = _threshold_array(gamma, big_prime_hex, device)
    mask = cuda.device_array(int(vocab_size), dtype=np.uint8)
    threads = 128
    blocks = (int(vocab_size) + threads - 1) // threads
    _poseidon2_t3_fixed_kernel[blocks, threads](
        secret_mont,
        previous_mont,
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        mask,
    )
    return mask.copy_to_host()


def _launch_three_to_one_torch_mask(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str,
    device: str | int | None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import torch

    cuda, _ = _cuda_modules()
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(3, device)
    secret_mont = cuda.to_device(int_to_mont_limbs(secret_key))
    previous_mont = cuda.to_device(int_to_mont_limbs(previous_token))
    threshold = _threshold_array(gamma, big_prime_hex, device)
    mask = torch.empty(int(vocab_size), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    mask_array = cuda.as_cuda_array(mask)
    threads = 128
    blocks = (int(vocab_size) + threads - 1) // threads
    _poseidon2_t3_fixed_kernel[blocks, threads](
        secret_mont,
        previous_mont,
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        mask_array,
    )
    return mask


def get_greenlist_ids_two_to_one_fixed(
    seed: str | int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> list[int]:
    mask = _launch_two_to_one(int(seed), int(vocab_size), float(gamma), str(big_prime_hex), device)
    return np.nonzero(mask)[0].astype(np.int64).tolist()


def get_greenlist_ids_two_to_one_fixed_torch(
    seed: str | int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    import torch

    mask = _launch_two_to_one_torch_mask(int(seed), int(vocab_size), float(gamma), str(big_prime_hex), device)
    return torch.nonzero(mask, as_tuple=False).flatten().to(dtype=torch.long)


def get_greenlist_ids_two_to_one_fixed_native(
    seed: str | int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    import torch

    scores = _native_score_buffer(int(vocab_size), device)
    scores.zero_()
    bias_logits_two_to_one_fixed_native(seed, scores, 1.0, gamma, big_prime_hex, device)
    return torch.nonzero(scores, as_tuple=False).flatten().to(dtype=torch.long)


def get_greenlist_ids_three_to_one_fixed(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> list[int]:
    mask = _launch_three_to_one(
        int(secret_key),
        int(previous_token),
        int(vocab_size),
        float(gamma),
        str(big_prime_hex),
        device,
    )
    return np.nonzero(mask)[0].astype(np.int64).tolist()


def get_greenlist_ids_three_to_one_fixed_torch(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    import torch

    mask = _launch_three_to_one_torch_mask(
        int(secret_key),
        int(previous_token),
        int(vocab_size),
        float(gamma),
        str(big_prime_hex),
        device,
    )
    return torch.nonzero(mask, as_tuple=False).flatten().to(dtype=torch.long)


def get_greenlist_ids_three_to_one_fixed_native(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    import torch

    scores = _native_score_buffer(int(vocab_size), device)
    scores.zero_()
    bias_logits_three_to_one_fixed_native(secret_key, previous_token, scores, 1.0, gamma, big_prime_hex, device)
    return torch.nonzero(scores, as_tuple=False).flatten().to(dtype=torch.long)


def clear_mask_cache() -> None:
    _MASK_CACHE.clear()


def get_mask_two_to_one_fixed_cached(
    seed: str | int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    device_id = _device_id(device)
    key = ("t2", device_id, int(vocab_size), float(gamma), str(big_prime_hex), int(seed))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = _launch_two_to_one_torch_mask(int(seed), int(vocab_size), float(gamma), str(big_prime_hex), device)
    _MASK_CACHE[key] = mask
    return mask


def get_mask_three_to_one_fixed_cached(
    secret_key: int,
    previous_token: int,
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    device_id = _device_id(device)
    key = ("t3", device_id, int(vocab_size), float(gamma), str(big_prime_hex), int(secret_key), int(previous_token))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = _launch_three_to_one_torch_mask(
        int(secret_key),
        int(previous_token),
        int(vocab_size),
        float(gamma),
        str(big_prime_hex),
        device,
    )
    _MASK_CACHE[key] = mask
    return mask


def get_masks_two_to_one_fixed_native(
    seeds: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not seeds:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _device_arrays(2, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    seed_table = _limb_table([int(seed) for seed in seeds], device)
    masks = torch.empty((len(seeds), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon2_t2_masks_precomputed(
        ctypes.c_int(len(seeds)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(seed_table.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon2 t=2 batched mask kernel failed with cuda error {err}")
    return masks


def get_masks_three_to_one_fixed_native(
    secret_key: int,
    previous_tokens: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not previous_tokens:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _device_arrays(3, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    secret = _limb_table([int(secret_key)], device)
    previous = _limb_table([int(token) for token in previous_tokens], device)
    masks = torch.empty((len(previous_tokens), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon2_t3_masks_precomputed(
        ctypes.c_int(len(previous_tokens)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(secret.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(previous.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon2 t=3 batched mask kernel failed with cuda error {err}")
    return masks


def get_poseidon_fast_masks_two_to_one_fixed_native(
    seeds: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not seeds:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _poseidon_fast_device_arrays(3, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    seed_table = _limb_table([int(seed) for seed in seeds], device)
    masks = torch.empty((len(seeds), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon_fast_w3_masks_precomputed(
        ctypes.c_int(len(seeds)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(seed_table.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["domain"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["mds"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon fast width=3 batched mask kernel failed with cuda error {err}")
    return masks


def get_poseidon_fast_masks_three_to_one_fixed_native(
    secret_key: int,
    previous_tokens: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not previous_tokens:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _poseidon_fast_device_arrays(4, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    secret = _limb_table([int(secret_key)], device)
    previous = _limb_table([int(token) for token in previous_tokens], device)
    masks = torch.empty((len(previous_tokens), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon_fast_w4_masks_precomputed(
        ctypes.c_int(len(previous_tokens)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(secret.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(previous.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["domain"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["mds"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon fast width=4 batched mask kernel failed with cuda error {err}")
    return masks


def get_mimc_masks_two_to_one_fixed_native(
    seeds: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not seeds:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _mimc_device_arrays(device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    seed_table = _limb_table([int(seed) for seed in seeds], device)
    masks = torch.empty((len(seeds), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.mimc_t2_masks_precomputed(
        ctypes.c_int(len(seeds)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(seed_table.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["round_keys"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA MiMC t=2 batched mask kernel failed with cuda error {err}")
    return masks


def get_mimc_masks_three_to_one_fixed_native(
    secret_key: int,
    previous_tokens: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
):
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not previous_tokens:
        return torch.empty((0, int(vocab_size)), dtype=torch.uint8, device=f"cuda:{_device_id(device)}")
    select_device(device)
    _select_torch_cuda_device(device)
    device_id = _device_id(device)
    arrays = _mimc_device_arrays(device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(vocab_size), device)
    secret = _limb_table([int(secret_key)], device)
    previous = _limb_table([int(token) for token in previous_tokens], device)
    masks = torch.empty((len(previous_tokens), int(vocab_size)), dtype=torch.uint8, device=f"cuda:{device_id}")
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.mimc_t3_masks_precomputed(
        ctypes.c_int(len(previous_tokens)),
        ctypes.c_int(int(vocab_size)),
        ctypes.c_void_p(int(secret.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(previous.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["round_keys"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_void_p(int(masks.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA MiMC t=3 batched mask kernel failed with cuda error {err}")
    return masks


def prefill_mask_cache_two_to_one_fixed_native(
    seed_by_prev_token: dict[int, int],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> dict[str, int]:
    if not seed_by_prev_token:
        return {"prefixes": 0, "vocab_size": int(vocab_size)}
    device_id = _device_id(device)
    items = sorted((int(prev), int(seed)) for prev, seed in seed_by_prev_token.items())
    seeds = [seed for _prev, seed in items]
    masks = get_masks_two_to_one_fixed_native(seeds, vocab_size, gamma, big_prime_hex, device)
    for row_index, (prev_token, seed) in enumerate(items):
        key = ("t2", device_id, int(vocab_size), float(gamma), str(big_prime_hex), int(seed))
        _MASK_CACHE[key] = masks[row_index]
    return {"prefixes": len(items), "vocab_size": int(vocab_size)}


def prefill_mask_cache_three_to_one_fixed_native(
    secret_key: int,
    previous_tokens: list[int] | tuple[int, ...],
    vocab_size: int,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> dict[str, int]:
    unique_previous = sorted({int(token) for token in previous_tokens})
    if not unique_previous:
        return {"prefixes": 0, "vocab_size": int(vocab_size)}
    device_id = _device_id(device)
    masks = get_masks_three_to_one_fixed_native(secret_key, unique_previous, vocab_size, gamma, big_prime_hex, device)
    for row_index, prev_token in enumerate(unique_previous):
        key = ("t3", device_id, int(vocab_size), float(gamma), str(big_prime_hex), int(secret_key), int(prev_token))
        _MASK_CACHE[key] = masks[row_index]
    return {"prefixes": len(unique_previous), "vocab_size": int(vocab_size)}


def bias_logits_with_mask(mask, scores, delta: float, device: str | int | None = None) -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import torch

    if not isinstance(scores, torch.Tensor) or not scores.is_cuda or scores.dtype != torch.float32:
        raise ValueError("scores must be a CUDA float32 torch.Tensor")
    if not isinstance(mask, torch.Tensor) or not mask.is_cuda or mask.dtype != torch.uint8:
        raise ValueError("mask must be a CUDA uint8 torch.Tensor")
    if scores.numel() != mask.numel():
        raise ValueError("scores and mask must have the same number of elements")

    cuda, _ = _cuda_modules()
    if device is None:
        device = f"cuda:{scores.device.index or 0}"
    select_device(device)
    _select_torch_cuda_device(device)
    mask_array = cuda.as_cuda_array(mask)
    scores_array = cuda.as_cuda_array(scores)
    vocab_size = int(scores.numel())
    threads = 256
    blocks = (vocab_size + threads - 1) // threads
    _bias_logits_with_mask_kernel[blocks, threads](mask_array, np.int32(vocab_size), np.float32(delta), scores_array)


def bias_logits_two_to_one_fixed_torch(
    seed: str | int,
    scores,
    delta: float,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import torch

    if not isinstance(scores, torch.Tensor) or not scores.is_cuda:
        raise ValueError("scores must be a CUDA torch.Tensor")

    cuda, _ = _cuda_modules()
    if device is None:
        device = f"cuda:{scores.device.index or 0}"
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(2, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    seed_limbs = int_to_mont_limbs(int(seed))
    scores_array = cuda.as_cuda_array(scores)
    vocab_size = int(scores.numel())
    threads = 128
    blocks = (vocab_size + threads - 1) // threads
    _poseidon2_t2_bias_logits_kernel[blocks, threads](
        np.uint32(seed_limbs[0]),
        np.uint32(seed_limbs[1]),
        np.uint32(seed_limbs[2]),
        np.uint32(seed_limbs[3]),
        np.uint32(seed_limbs[4]),
        np.uint32(seed_limbs[5]),
        np.uint32(seed_limbs[6]),
        np.uint32(seed_limbs[7]),
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        np.float32(delta),
        scores_array,
    )


def bias_logits_two_to_one_fixed_native(
    seed: str | int,
    scores,
    delta: float,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not isinstance(scores, torch.Tensor) or not scores.is_cuda or scores.dtype != torch.float32:
        raise ValueError("scores must be a CUDA float32 torch.Tensor")

    if device is None:
        device = f"cuda:{scores.device.index or 0}"
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(2, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(scores.numel()), device)
    seed_limbs = int_to_mont_limbs(int(seed))
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon2_t2_bias_precomputed(
        ctypes.c_uint32(int(seed_limbs[0])),
        ctypes.c_uint32(int(seed_limbs[1])),
        ctypes.c_uint32(int(seed_limbs[2])),
        ctypes.c_uint32(int(seed_limbs[3])),
        ctypes.c_uint32(int(seed_limbs[4])),
        ctypes.c_uint32(int(seed_limbs[5])),
        ctypes.c_uint32(int(seed_limbs[6])),
        ctypes.c_uint32(int(seed_limbs[7])),
        ctypes.c_int(int(scores.numel())),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_float(float(delta)),
        ctypes.c_void_p(int(scores.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon2 t=2 kernel failed with cuda error {err}")


def bias_logits_three_to_one_fixed_torch(
    secret_key: int,
    previous_token: int,
    scores,
    delta: float,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import torch

    if not isinstance(scores, torch.Tensor) or not scores.is_cuda:
        raise ValueError("scores must be a CUDA torch.Tensor")

    cuda, _ = _cuda_modules()
    if device is None:
        device = f"cuda:{scores.device.index or 0}"
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(3, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    scores_array = cuda.as_cuda_array(scores)
    vocab_size = int(scores.numel())
    threads = 128
    blocks = (vocab_size + threads - 1) // threads
    _poseidon2_t3_bias_logits_kernel[blocks, threads](
        np.uint32(secret_key),
        np.uint32(previous_token),
        np.int32(vocab_size),
        threshold,
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        np.float32(delta),
        scores_array,
    )


def bias_logits_three_to_one_fixed_native(
    secret_key: int,
    previous_token: int,
    scores,
    delta: float,
    gamma: float,
    big_prime_hex: str = BN254_PRIME_HEX,
    device: str | int | None = None,
) -> None:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    import ctypes
    import torch

    if not isinstance(scores, torch.Tensor) or not scores.is_cuda or scores.dtype != torch.float32:
        raise ValueError("scores must be a CUDA float32 torch.Tensor")

    if device is None:
        device = f"cuda:{scores.device.index or 0}"
    select_device(device)
    _select_torch_cuda_device(device)
    arrays = _device_arrays(3, device)
    threshold = _threshold_array(gamma, big_prime_hex, device)
    token_mont = _token_mont_table(int(scores.numel()), device)
    secret_limbs = int_to_mont_limbs(int(secret_key))
    previous_limbs = int_to_mont_limbs(int(previous_token))
    lib = _cuda_kernel_lib()
    stream_ptr = _torch_stream_ptr()
    err = lib.poseidon2_t3_bias_precomputed(
        ctypes.c_uint32(int(secret_limbs[0])),
        ctypes.c_uint32(int(secret_limbs[1])),
        ctypes.c_uint32(int(secret_limbs[2])),
        ctypes.c_uint32(int(secret_limbs[3])),
        ctypes.c_uint32(int(secret_limbs[4])),
        ctypes.c_uint32(int(secret_limbs[5])),
        ctypes.c_uint32(int(secret_limbs[6])),
        ctypes.c_uint32(int(secret_limbs[7])),
        ctypes.c_uint32(int(previous_limbs[0])),
        ctypes.c_uint32(int(previous_limbs[1])),
        ctypes.c_uint32(int(previous_limbs[2])),
        ctypes.c_uint32(int(previous_limbs[3])),
        ctypes.c_uint32(int(previous_limbs[4])),
        ctypes.c_uint32(int(previous_limbs[5])),
        ctypes.c_uint32(int(previous_limbs[6])),
        ctypes.c_uint32(int(previous_limbs[7])),
        ctypes.c_int(int(scores.numel())),
        ctypes.c_void_p(int(threshold.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(token_mont.device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["one"].device_ctypes_pointer.value)),
        ctypes.c_void_p(int(arrays["rc"].device_ctypes_pointer.value)),
        ctypes.c_uint32(MONT_N0),
        ctypes.c_float(float(delta)),
        ctypes.c_void_p(int(scores.data_ptr())),
        ctypes.c_void_p(int(stream_ptr or 0)),
    )
    if err != 0:
        raise RuntimeError(f"native CUDA Poseidon2 t=3 kernel failed with cuda error {err}")


def debug_u32_roundtrip(value: int, device: str | int | None = None) -> int:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    cuda, _ = _cuda_modules()
    arrays = _device_arrays(2, device)
    out = cuda.device_array(LIMBS, dtype=np.uint32)
    _debug_u32_roundtrip_kernel[1, 1](
        np.uint32(value),
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        np.uint32(MONT_N0),
        out,
    )
    return limbs_to_int(out.copy_to_host())


def debug_mont_mul(a: int, b: int, device: str | int | None = None) -> int:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    cuda, _ = _cuda_modules()
    arrays = _device_arrays(2, device)
    a_mont = cuda.to_device(int_to_mont_limbs(a))
    b_mont = cuda.to_device(int_to_mont_limbs(b))
    out = cuda.device_array(LIMBS, dtype=np.uint32)
    _debug_mont_mul_kernel[1, 1](
        a_mont,
        b_mont,
        arrays["p"],
        arrays["one"],
        np.uint32(MONT_N0),
        out,
    )
    return limbs_to_int(out.copy_to_host())


def debug_poseidon2_t2_hash(seed: int, token: int, device: str | int | None = None) -> int:
    if not is_available():
        raise RuntimeError("CUDA is not available")
    cuda, _ = _cuda_modules()
    arrays = _device_arrays(2, device)
    seed_mont = cuda.to_device(int_to_mont_limbs(seed))
    out = cuda.device_array(LIMBS, dtype=np.uint32)
    _debug_poseidon2_t2_hash_kernel[1, 1](
        seed_mont,
        np.uint32(token),
        arrays["p"],
        arrays["r2"],
        arrays["one"],
        arrays["rc"],
        np.uint32(MONT_N0),
        out,
    )
    return limbs_to_int(out.copy_to_host())


def enabled_from_env() -> bool:
    return os.environ.get("HASH_KGW_POSEIDON2_GPU", "").strip().lower() in {"1", "true", "yes", "on", "native"}


def _define_kernels() -> None:
    global _poseidon2_t2_fixed_kernel, _poseidon2_t3_fixed_kernel, _bias_logits_with_mask_kernel
    global _poseidon2_t2_bias_logits_kernel, _poseidon2_t3_bias_logits_kernel
    global _debug_u32_roundtrip_kernel, _debug_mont_mul_kernel, _debug_poseidon2_t2_hash_kernel
    cuda, numba_types = _cuda_modules()
    uint32, uint64 = numba_types

    @cuda.jit(device=True, inline=True)
    def copy8(dst, src):
        for i in range(8):
            dst[i] = src[i]

    @cuda.jit(device=True, inline=True)
    def zero8(dst):
        for i in range(8):
            dst[i] = uint32(0)

    @cuda.jit(device=True, inline=True)
    def init_bn254_prime(dst):
        dst[0] = uint32(4026531841)
        dst[1] = uint32(1138881939)
        dst[2] = uint32(2042196113)
        dst[3] = uint32(674490440)
        dst[4] = uint32(2172737629)
        dst[5] = uint32(3092268470)
        dst[6] = uint32(3778125865)
        dst[7] = uint32(811880050)

    @cuda.jit(device=True, inline=True)
    def geq8(a, b):
        for rev in range(8):
            i = 7 - rev
            if a[i] > b[i]:
                return True
            if a[i] < b[i]:
                return False
        return True

    @cuda.jit(device=True, inline=True)
    def lt8(a, b):
        for rev in range(8):
            i = 7 - rev
            if a[i] < b[i]:
                return True
            if a[i] > b[i]:
                return False
        return False

    @cuda.jit(device=True, inline=True)
    def sub_assign(a, b):
        borrow = uint64(0)
        base = uint64(4294967296)
        for i in range(8):
            ai = uint64(a[i])
            bi = uint64(b[i]) + borrow
            if ai >= bi:
                a[i] = uint32(ai - bi)
                borrow = uint64(0)
            else:
                a[i] = uint32(base + ai - bi)
                borrow = uint64(1)

    @cuda.jit(device=True, inline=True)
    def add_assign_mod(a, b, p):
        carry = uint64(0)
        mask = uint64(4294967295)
        for i in range(8):
            uv = uint64(a[i]) + uint64(b[i]) + carry
            a[i] = uint32(uv & mask)
            carry = uv >> 32
        if carry != 0 or geq8(a, p):
            sub_assign(a, p)

    @cuda.jit(device=True, inline=True)
    def add_const_assign_mod(a, constants, offset, p):
        carry = uint64(0)
        mask = uint64(4294967295)
        for i in range(8):
            uv = uint64(a[i]) + uint64(constants[offset + i]) + carry
            a[i] = uint32(uv & mask)
            carry = uv >> 32
        if carry != 0 or geq8(a, p):
            sub_assign(a, p)

    @cuda.jit(device=True, inline=True)
    def double_assign_mod(a, p):
        carry = uint64(0)
        mask = uint64(4294967295)
        for i in range(8):
            uv = uint64(a[i]) + uint64(a[i]) + carry
            a[i] = uint32(uv & mask)
            carry = uv >> 32
        if carry != 0 or geq8(a, p):
            sub_assign(a, p)

    @cuda.jit(device=True)
    def mont_mul(out, a, b, p, n0):
        tmp = cuda.local.array(18, dtype=uint64)
        mask = uint64(4294967295)
        for i in range(18):
            tmp[i] = uint64(0)

        for i in range(8):
            carry = uint64(0)
            ai = uint64(a[i])
            for j in range(8):
                uv = tmp[i + j] + ai * uint64(b[j]) + carry
                tmp[i + j] = uv & mask
                carry = uv >> 32
            k = i + 8
            while carry != 0:
                uv = tmp[k] + carry
                tmp[k] = uv & mask
                carry = uv >> 32
                k += 1

        for i in range(8):
            m = uint32((tmp[i] * uint64(n0)) & mask)
            carry = uint64(0)
            for j in range(8):
                uv = tmp[i + j] + uint64(m) * uint64(p[j]) + carry
                tmp[i + j] = uv & mask
                carry = uv >> 32
            k = i + 8
            while carry != 0:
                uv = tmp[k] + carry
                tmp[k] = uv & mask
                carry = uv >> 32
                k += 1

        for i in range(8):
            out[i] = uint32(tmp[i + 8] & mask)
        if geq8(out, p):
            sub_assign(out, p)

    @cuda.jit(device=True)
    def u32_to_mont(out, value, r2, p, n0):
        canonical = cuda.local.array(8, dtype=uint32)
        zero8(canonical)
        canonical[0] = uint32(value)
        mont_mul(out, canonical, r2, p, n0)

    @cuda.jit(device=True)
    def pow5_inplace(a, p, n0):
        x2 = cuda.local.array(8, dtype=uint32)
        x4 = cuda.local.array(8, dtype=uint32)
        mont_mul(x2, a, a, p, n0)
        mont_mul(x4, x2, x2, p, n0)
        mont_mul(a, x4, a, p, n0)

    @cuda.jit(device=True)
    def mat_external2(s0, s1, p):
        total = cuda.local.array(8, dtype=uint32)
        copy8(total, s0)
        add_assign_mod(total, s1, p)
        add_assign_mod(s0, total, p)
        add_assign_mod(s1, total, p)

    @cuda.jit(device=True)
    def mat_internal2(s0, s1, p):
        total = cuda.local.array(8, dtype=uint32)
        copy8(total, s0)
        add_assign_mod(total, s1, p)
        add_assign_mod(s0, total, p)
        double_assign_mod(s1, p)
        add_assign_mod(s1, total, p)

    @cuda.jit(device=True)
    def mat_external3(s0, s1, s2, p):
        total = cuda.local.array(8, dtype=uint32)
        copy8(total, s0)
        add_assign_mod(total, s1, p)
        add_assign_mod(total, s2, p)
        add_assign_mod(s0, total, p)
        add_assign_mod(s1, total, p)
        add_assign_mod(s2, total, p)

    @cuda.jit(device=True)
    def mat_internal3(s0, s1, s2, p):
        total = cuda.local.array(8, dtype=uint32)
        copy8(total, s0)
        add_assign_mod(total, s1, p)
        add_assign_mod(total, s2, p)
        add_assign_mod(s0, total, p)
        add_assign_mod(s1, total, p)
        double_assign_mod(s2, p)
        add_assign_mod(s2, total, p)

    @cuda.jit
    def bias_logits_with_mask_kernel(mask, vocab_size, delta, scores):
        idx = cuda.grid(1)
        if idx >= vocab_size:
            return
        if mask[idx] != 0:
            scores[idx] = scores[idx] + delta

    @cuda.jit
    def debug_u32_roundtrip_kernel(value, p, r2, one, n0, out):
        mont = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        u32_to_mont(mont, value, r2, p, n0)
        mont_mul(result, mont, one, p, n0)
        for i in range(8):
            out[i] = result[i]

    @cuda.jit
    def debug_mont_mul_kernel(a, b, p, one, n0, out):
        product = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        mont_mul(product, a, b, p, n0)
        mont_mul(result, product, one, p, n0)
        for i in range(8):
            out[i] = result[i]

    @cuda.jit
    def debug_poseidon2_t2_hash_kernel(seed, token, p, r2, one, rc, n0, out):
        s0 = cuda.local.array(8, dtype=uint32)
        s1 = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        copy8(s0, seed)
        u32_to_mont(s1, token, r2, p, n0)

        mat_external2(s0, s1, p)

        for r in range(4):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            mat_external2(s0, s1, p)

        for r in range(4, 60):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            pow5_inplace(s0, p, n0)
            mat_internal2(s0, s1, p)

        for r in range(60, 64):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            mat_external2(s0, s1, p)

        mont_mul(result, s0, one, p, n0)
        for i in range(8):
            out[i] = result[i]

    @cuda.jit
    def poseidon2_t2_bias_logits_kernel(
        seed0,
        seed1,
        seed2,
        seed3,
        seed4,
        seed5,
        seed6,
        seed7,
        vocab_size,
        threshold,
        p,
        r2,
        one,
        rc,
        n0,
        delta,
        scores,
    ):
        idx = cuda.grid(1)
        if idx >= vocab_size:
            return

        p_local = cuda.local.array(8, dtype=uint32)
        s0 = cuda.local.array(8, dtype=uint32)
        s1 = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        init_bn254_prime(p_local)
        s0[0] = seed0
        s0[1] = seed1
        s0[2] = seed2
        s0[3] = seed3
        s0[4] = seed4
        s0[5] = seed5
        s0[6] = seed6
        s0[7] = seed7
        u32_to_mont(s1, idx, r2, p_local, n0)

        mat_external2(s0, s1, p_local)

        for r in range(4):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p_local)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            pow5_inplace(s1, p_local, n0)
            mat_external2(s0, s1, p_local)

        for r in range(4, 60):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            mat_internal2(s0, s1, p_local)

        for r in range(60, 64):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p_local)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            pow5_inplace(s1, p_local, n0)
            mat_external2(s0, s1, p_local)

        mont_mul(result, s0, one, p_local, n0)
        if lt8(result, threshold):
            scores[idx] = scores[idx] + delta

    @cuda.jit
    def poseidon2_t3_bias_logits_kernel(secret_key, previous_token, vocab_size, threshold, p, r2, one, rc, n0, delta, scores):
        idx = cuda.grid(1)
        if idx >= vocab_size:
            return

        p_local = cuda.local.array(8, dtype=uint32)
        s0 = cuda.local.array(8, dtype=uint32)
        s1 = cuda.local.array(8, dtype=uint32)
        s2 = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        init_bn254_prime(p_local)
        u32_to_mont(s0, secret_key, r2, p_local, n0)
        u32_to_mont(s1, previous_token, r2, p_local, n0)
        u32_to_mont(s2, idx, r2, p_local, n0)

        mat_external3(s0, s1, s2, p_local)

        for r in range(4):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p_local)
            add_const_assign_mod(s1, rc, (r * 3 + 1) * 8, p_local)
            add_const_assign_mod(s2, rc, (r * 3 + 2) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            pow5_inplace(s1, p_local, n0)
            pow5_inplace(s2, p_local, n0)
            mat_external3(s0, s1, s2, p_local)

        for r in range(4, 60):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            mat_internal3(s0, s1, s2, p_local)

        for r in range(60, 64):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p_local)
            add_const_assign_mod(s1, rc, (r * 3 + 1) * 8, p_local)
            add_const_assign_mod(s2, rc, (r * 3 + 2) * 8, p_local)
            pow5_inplace(s0, p_local, n0)
            pow5_inplace(s1, p_local, n0)
            pow5_inplace(s2, p_local, n0)
            mat_external3(s0, s1, s2, p_local)

        mont_mul(result, s0, one, p_local, n0)
        if lt8(result, threshold):
            scores[idx] = scores[idx] + delta

    @cuda.jit
    def poseidon2_t2_fixed_kernel(seed, vocab_size, threshold, p, r2, one, rc, n0, mask_out):
        idx = cuda.grid(1)
        if idx >= vocab_size:
            return

        s0 = cuda.local.array(8, dtype=uint32)
        s1 = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        copy8(s0, seed)
        u32_to_mont(s1, idx, r2, p, n0)

        mat_external2(s0, s1, p)

        for r in range(4):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            mat_external2(s0, s1, p)

        for r in range(4, 60):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            pow5_inplace(s0, p, n0)
            mat_internal2(s0, s1, p)

        for r in range(60, 64):
            add_const_assign_mod(s0, rc, (r * 2) * 8, p)
            add_const_assign_mod(s1, rc, (r * 2 + 1) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            mat_external2(s0, s1, p)

        mont_mul(result, s0, one, p, n0)
        mask_out[idx] = uint32(1) if lt8(result, threshold) else uint32(0)

    @cuda.jit
    def poseidon2_t3_fixed_kernel(secret, previous, vocab_size, threshold, p, r2, one, rc, n0, mask_out):
        idx = cuda.grid(1)
        if idx >= vocab_size:
            return

        s0 = cuda.local.array(8, dtype=uint32)
        s1 = cuda.local.array(8, dtype=uint32)
        s2 = cuda.local.array(8, dtype=uint32)
        result = cuda.local.array(8, dtype=uint32)
        copy8(s0, secret)
        copy8(s1, previous)
        u32_to_mont(s2, idx, r2, p, n0)

        mat_external3(s0, s1, s2, p)

        for r in range(4):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p)
            add_const_assign_mod(s1, rc, (r * 3 + 1) * 8, p)
            add_const_assign_mod(s2, rc, (r * 3 + 2) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            pow5_inplace(s2, p, n0)
            mat_external3(s0, s1, s2, p)

        for r in range(4, 60):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p)
            pow5_inplace(s0, p, n0)
            mat_internal3(s0, s1, s2, p)

        for r in range(60, 64):
            add_const_assign_mod(s0, rc, (r * 3) * 8, p)
            add_const_assign_mod(s1, rc, (r * 3 + 1) * 8, p)
            add_const_assign_mod(s2, rc, (r * 3 + 2) * 8, p)
            pow5_inplace(s0, p, n0)
            pow5_inplace(s1, p, n0)
            pow5_inplace(s2, p, n0)
            mat_external3(s0, s1, s2, p)

        mont_mul(result, s0, one, p, n0)
        mask_out[idx] = uint32(1) if lt8(result, threshold) else uint32(0)

    _poseidon2_t2_fixed_kernel = poseidon2_t2_fixed_kernel
    _poseidon2_t3_fixed_kernel = poseidon2_t3_fixed_kernel
    _bias_logits_with_mask_kernel = bias_logits_with_mask_kernel
    _poseidon2_t2_bias_logits_kernel = poseidon2_t2_bias_logits_kernel
    _poseidon2_t3_bias_logits_kernel = poseidon2_t3_bias_logits_kernel
    _debug_u32_roundtrip_kernel = debug_u32_roundtrip_kernel
    _debug_mont_mul_kernel = debug_mont_mul_kernel
    _debug_poseidon2_t2_hash_kernel = debug_poseidon2_t2_hash_kernel


_define_kernels()
