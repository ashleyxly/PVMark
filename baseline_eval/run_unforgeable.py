import sys
from pathlib import Path
_d = str(Path(__file__).resolve().parent.parent / "src" / "baselines" / "upv")
if _d not in sys.path: sys.path.insert(0, _d)
from run_unforgeable import *  # noqa
