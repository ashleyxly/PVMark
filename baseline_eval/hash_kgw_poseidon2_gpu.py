# Re-export all symbols from src/native_libraries/kgw/hash_kgw_poseidon2_gpu.py
import sys
from pathlib import Path

_kgw_dir = str(Path(__file__).resolve().parent.parent / "src" / "native_libraries" / "kgw")
if _kgw_dir not in sys.path:
    sys.path.insert(0, _kgw_dir)

try:
    from hash_kgw_poseidon2_gpu import *  # noqa: F401,F403
except (ImportError, SystemError) as e:
    import warnings
    warnings.warn(f"hash_kgw_poseidon2_gpu not available (requires CUDA/numba): {e}")
