from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str) -> str:
    return os.environ.get(name, default)


PVMARK_ROOT = Path(env_path("PVMARK_ROOT", Path(__file__).resolve().parents[2].as_posix()))
DATA_DIR = Path(env_path("DATA_DIR", (PVMARK_ROOT / "experiment_data").as_posix()))
RESULT_DIR = Path(env_path("RESULT_DIR", (PVMARK_ROOT / "reproduction_outputs").as_posix()))

GEN_MODEL = env_path("GEN_MODEL", "facebook/opt-1.3b")
PPL_MODEL = env_path("PPL_MODEL", "facebook/opt-2.7b")
BERT_MODEL = env_path("BERT_MODEL", "bert-base-uncased")
MARKLLM_ROOT = env_path("MARKLLM_ROOT", "")


def c4_prompt_file(input_num: int) -> Path:
    candidates = [
        DATA_DIR / "prompts" / f"num_{input_num}.json",
        PVMARK_ROOT / "data" / "c4_test" / "news" / f"num_{input_num}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def kgw_result_file(input_num: int, hash_type: int, hash_method: int, part: int) -> Path:
    return RESULT_DIR / "c4_dataset_test" / f"num_{input_num}_hash_type_{hash_type}_hash_method_{hash_method}_part_{part}.json"


def detection_result_file(model: str, dataset: str, hash_type: int, hash_method: int) -> Path:
    return RESULT_DIR / "Detection" / f"Detection_model_{model}_dataset_{dataset}_hash_type_{hash_type}_hash_method_{hash_method}.json"


def ppl_result_file(model: str, dataset: str, hash_type: int, hash_method: int) -> Path:
    return RESULT_DIR / "PPL" / f"PPL_model_{model}_dataset_{dataset}_hash_type_{hash_type}_hash_method_{hash_method}.json"
