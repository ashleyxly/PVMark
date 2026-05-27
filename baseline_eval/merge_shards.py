import sys
from pathlib import Path
_d = str(Path(__file__).resolve().parent.parent / "src" / "watermark" / "synthid" / "notebooks" / "baseline_compare")
if _d not in sys.path: sys.path.insert(0, _d)
try:
    from merge_shards import *  # noqa
except ImportError:
    print("[WARN] merge_shards module not found - this module may not be needed for main experiments")
