# Common utilities for UPV baseline experiments
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

PVMARK_ROOT = Path(os.environ.get("PVMARK_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_DATASET_PATH = os.environ.get("DATASET", str(PVMARK_ROOT / "experiment_data" / "prompts" / "num_100.json"))
DEFAULT_GPT2_PATH = os.environ.get("GEN_MODEL", "openai-community/gpt2")
DEFAULT_UPV_ROOT = os.environ.get("UPV_ROOT", "")
DEFAULT_UPV_GENERATOR = os.environ.get("UPV_GENERATOR", "")


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_payload_records(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    records: Dict[str, Any] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = record_key(row)
                records[key] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return records


def record_key(row: Dict[str, Any]) -> str:
    return str(row.get("id", row.get("sample_id", "")))


def completed_keys(records: Dict[str, Any]) -> set:
    return set(records.keys())


def sort_records(records: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(records.values(), key=lambda r: r.get("id", 0))


def summarize_numbers(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def load_eli5_prompts(path: str, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    data = read_json(path)
    records = []
    for item in data:
        prompt = item.get("text_shortened", item.get("text", item.get("question", "")))
        if prompt:
            records.append({"text": prompt})
        if max_samples and len(records) >= max_samples:
            break
    return records
