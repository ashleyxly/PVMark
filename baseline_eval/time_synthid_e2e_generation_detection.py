import sys
from pathlib import Path
_scripts = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
from time_synthid_e2e_generation_detection import *  # noqa
