from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


REPO_ROOT = Path(os.environ.get("PVMark_SYNTHID_ROOT", "."))
DEFAULT_DATASET_PATH = Path(
    os.environ.get("PVMark_ELI5_SELECT_TEST", "experiment_data/prompts/select_test.json")
)
DEFAULT_GPT2_PATH = Path(os.environ.get("PVMark_GPT2_MODEL", "gpt2"))
DEFAULT_OPT27B_PATH = Path(
    os.environ.get("PVMark_OPT_PPL_MODEL", "facebook/opt-2.7b")
)
DEFAULT_BERT_PATH = Path(
    os.environ.get("PVMark_BERT_MODEL", "bert-base-uncased")
)
DEFAULT_PUBLICLY_DETECTABLE_ROOT = Path(
    os.environ.get("PVMark_PDW_ROOT", "external/publicly-detectable-watermark")
)
DEFAULT_UPV_ROOT = Path(os.environ.get("PVMark_UPV_ROOT", "external/unforgeable_watermark"))
DEFAULT_UPV_GENERATOR = (
    DEFAULT_UPV_ROOT / "experiments/main_experiments/generator_model/combine_model.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def completed_keys(path: str | Path, key_fn: Callable[[dict[str, Any]], str]) -> set[str]:
    return {key_fn(record) for record in read_jsonl(path)}


def shard_items(items: list[dict[str, Any]], shard_index: int = 0, num_shards: int = 1) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    return [
        item
        for position, item in enumerate(items)
        if position % num_shards == shard_index
    ]


def record_key(record: dict[str, Any], *extra: Any) -> str:
    parts = [
        str(record.get("sample_id")),
        "wm" if bool(record.get("watermarked")) else "uwm",
    ]
    parts.extend(str(x) for x in extra if x is not None)
    return "::".join(parts)


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda r: (
            int(r.get("sample_id") if r.get("sample_id") is not None else -1),
            1 if bool(r.get("watermarked")) else 2,
            str(r.get("attack") or ""),
        ),
    )


def load_existing_payload_records(
    payload_path: str | Path,
    jsonl_path: str | Path,
    key_fn: Callable[[dict[str, Any]], str],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    payload_path = Path(payload_path)
    if payload_path.exists():
        payload = read_json(payload_path)
        for record in payload.get("records", []):
            merged[key_fn(record)] = record
    for record in read_jsonl(jsonl_path):
        merged[key_fn(record)] = record
    return list(merged.values())


def load_eli5_prompts(dataset_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    data = read_json(dataset_path)
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        title = item.get("title")
        if not title:
            continue
        records.append(
            {
                "sample_id": len(records),
                "source_index": idx,
                "q_id": item.get("q_id"),
                "prompt": title,
                "subreddit": item.get("subreddit"),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_generation_payload(
    *,
    method: str,
    model_name_or_path: str,
    dataset_path: str,
    seed: int,
    records: list[dict[str, Any]],
    generation_config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "created_at": int(time.time()),
            "method": method,
            "model_name_or_path": model_name_or_path,
            "dataset_path": dataset_path,
            "seed": seed,
            "num_records": len(records),
            "generation_config": generation_config,
            **(extra_metadata or {}),
        },
        "records": records,
    }


def cuda_synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __enter__(self) -> "Timer":
        cuda_synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        cuda_synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def summarize_numbers(values: Iterable[float]) -> dict[str, float | int | None]:
    xs = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not xs:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p90": None,
        }
    return {
        "count": len(xs),
        "mean": float(statistics.fmean(xs)),
        "median": float(statistics.median(xs)),
        "min": float(min(xs)),
        "max": float(max(xs)),
        "p90": float(np.percentile(np.asarray(xs), 90)),
    }


def split_records_by_watermark(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wm = [r for r in records if bool(r.get("watermarked"))]
    uwm = [r for r in records if not bool(r.get("watermarked"))]
    return wm, uwm
