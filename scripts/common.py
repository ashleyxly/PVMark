from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch


DEFAULT_PVMARK_ROOT = Path(os.environ.get("PVMARK_ROOT", Path.cwd()))
DEFAULT_DATASET = os.environ.get("DATASET", str(DEFAULT_PVMARK_ROOT / "experiment_data" / "prompts" / "num_100.json"))
DEFAULT_GENERATION_MODEL = os.environ.get("GEN_MODEL", "facebook/opt-1.3b")
DEFAULT_PPL_MODEL = os.environ.get("PPL_MODEL", "facebook/opt-2.7b")
DEFAULT_RESULTS_ROOT = os.environ.get("RESULTS_ROOT", str(DEFAULT_PVMARK_ROOT / "reproduction_outputs" / "baseline_results"))
DEFAULT_MARKLLM_ROOT = os.environ.get("MARKLLM_ROOT", "")
DEFAULT_PDW_ROOT = os.environ.get("PDW_ROOT", "")
DEFAULT_UPV_ROOT = os.environ.get("UPV_ROOT", "")
DEFAULT_BERT_MODEL = os.environ.get("BERT_MODEL", "bert-base-uncased")


def add_repo_to_path(repo_root: str) -> None:
    root = str(Path(repo_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class WallTimer:
    def __enter__(self) -> "WallTimer":
        cuda_sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            cuda_sync()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    tmp_path = path_obj.with_name(f".{path_obj.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path_obj)


def load_c4_records(
    dataset_path: str,
    max_samples: Optional[int] = None,
    shard_index: int = 0,
    num_shards: int = 1,
) -> List[Dict[str, str]]:
    if max_samples is not None and max_samples <= 0:
        return []
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    data = read_json(dataset_path)
    records: List[Dict[str, str]] = []
    for idx, row in enumerate(data):
        if "text_shortened" not in row:
            continue
        records.append(
            {
                "id": idx,
                "input_text": row.get("text_shortened", ""),
                "reference_text_removed": row.get("text_removed", ""),
                "text_full": row.get("text_full", ""),
            }
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    if num_shards > 1:
        records = [row for pos, row in enumerate(records) if pos % num_shards == shard_index]
    return records


def select_shard(items: List[Dict[str, Any]], shard_index: int = 0, num_shards: int = 1) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return items
    return [item for pos, item in enumerate(items) if pos % num_shards == shard_index]


def read_json_dir(path: str | Path, id_field: str) -> Dict[int, Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    for item_path in sorted(path_obj.glob("*.json")):
        row = read_json(item_path)
        if id_field in row:
            rows[int(row[id_field])] = row
    return rows


def checkpoint_path(base_dir: str | Path, sample_id: int) -> Path:
    return Path(base_dir) / f"{int(sample_id):06d}.json"


def write_checkpoint(base_dir: str | Path, sample_id: int, row: Dict[str, Any]) -> None:
    write_json(checkpoint_path(base_dir, sample_id), row)


def make_failure(sample_id: int, stage: str, exc: BaseException, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    failure = {
        "id": int(sample_id),
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        failure["extra"] = extra
    return failure


def remove_checkpoint_if_exists(base_dir: str | Path, sample_id: int) -> None:
    path = checkpoint_path(base_dir, sample_id)
    if path.exists():
        path.unlink()


def get_device(use_gpu: bool = True) -> str:
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"


def load_causal_lm_and_tokenizer(
    model_name_or_path: str,
    use_gpu: bool = True,
    load_fp16: bool = False,
    device: Optional[str] = None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {}
    if load_fp16:
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    device = device or get_device(use_gpu)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")
    if not load_fp16:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    return model, tokenizer, device


def prepare_prompt(
    tokenizer: Any,
    prompt: str,
    device: str,
    prompt_max_length: int,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_max_length,
    ).to(device)


def decode_new_tokens(tokenizer: Any, output_ids: torch.Tensor) -> str:
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    return int(tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1])


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(result):
        return result
    return None


def bool_from_score(score: Optional[float], threshold: float) -> bool:
    return bool(score is not None and score > threshold)


@dataclass
class GenerationSample:
    id: int
    input_text: str
    reference_text_removed: str
    output_without_watermark: str
    output_with_watermark: str
    token_count_without_watermark: int
    token_count_with_watermark: int
    char_count_without_watermark: int
    char_count_with_watermark: int
    score_without_watermark: Optional[float]
    score_with_watermark: Optional[float]
    prediction_without_watermark: bool
    prediction_with_watermark: bool
    generation_time_without_watermark_sec: float
    generation_time_with_watermark_sec: float
    detection_time_without_watermark_sec: float
    detection_time_with_watermark_sec: float
    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_generation_payload(
    scheme: str,
    args: argparse.Namespace,
    samples: Iterable[GenerationSample | Dict[str, Any]],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "metadata": {
            "scheme": scheme,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
        },
        "samples": [sample.to_dict() if isinstance(sample, GenerationSample) else sample for sample in samples],
    }
    if extra_metadata:
        payload["metadata"].update(extra_metadata)
    return payload


def summarize_clean_detection(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    wm = [s for s in samples if s.get("output_with_watermark")]
    plain = [s for s in samples if s.get("output_without_watermark")]
    tpr_num = sum(1 for s in wm if s.get("prediction_with_watermark"))
    fpr_num = sum(1 for s in plain if s.get("prediction_without_watermark"))
    return {
        "num_watermarked": len(wm),
        "num_unwatermarked": len(plain),
        "watermarked_detected": tpr_num,
        "unwatermarked_false_positive": fpr_num,
        "tpr": tpr_num / len(wm) if wm else None,
        "fpr": fpr_num / len(plain) if plain else None,
        "avg_generation_time_with_watermark_sec": _avg(
            s.get("generation_time_with_watermark_sec") for s in wm
        ),
        "avg_generation_time_without_watermark_sec": _avg(
            s.get("generation_time_without_watermark_sec") for s in plain
        ),
        "avg_detection_time_with_watermark_sec": _avg(
            s.get("detection_time_with_watermark_sec") for s in wm
        ),
        "avg_detection_time_without_watermark_sec": _avg(
            s.get("detection_time_without_watermark_sec") for s in plain
        ),
        "avg_token_count_with_watermark": _avg(s.get("token_count_with_watermark") for s in wm),
        "avg_token_count_without_watermark": _avg(s.get("token_count_without_watermark") for s in plain),
    }


def _avg(values: Iterable[Any]) -> Optional[float]:
    vals = [finite_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def append_resume_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failures", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)


def append_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--prompt-max-length", type=int, default=1848)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-fp16", action=argparse.BooleanOptionalAction, default=False)
    append_resume_shard_args(parser)
