import sys
from pathlib import Path
_d = str(Path(__file__).resolve().parent.parent / "src" / "baselines" / "markllm_attacks")
if _d not in sys.path: sys.path.insert(0, _d)
try:
    from prepare_kgw_legacy import *  # noqa
except ImportError:
    print("[WARN] prepare_kgw_legacy module not found - this module may not be needed for main experiments")
