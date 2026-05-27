from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GPT2_PATH,
    DEFAULT_UPV_GENERATOR,
    DEFAULT_UPV_ROOT,
    Timer,
    append_jsonl,
    completed_keys,
    ensure_dir,
    load_eli5_prompts,
    load_existing_payload_records,
    record_key,
    set_seed,
    summarize_numbers,
    write_json,
)
from upv_network_detector import (
    TransformerClassifier,
    compute_binary_metrics,
    ids_to_feature,
)


DEFAULT_OUTPUT_DIR = Path("tests/baseline_comparison/upv_reverse_training_gpt2_eli5")
DEFAULT_NETWORK_DETECTOR = Path(
    "tests/baseline_comparison/upv_network_detector_gpt2_eli5/detector_z1.pt"
)
DEFAULT_SUBNET = (
    DEFAULT_UPV_ROOT / "experiments/main_experiments/generator_model/sub_net.pt"
)
DEFAULT_UPV_TRAIN_DATA = (
    DEFAULT_UPV_ROOT / "experiments/main_experiments/train_and_test_data/train_data.jsonl"
)


class SubNet(nn.Module):
    """Same token-level subnet used by UPV model_key.py."""

    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BinaryClassifier(nn.Module):
    """Same architecture as UPV model_key.py::BinaryClassifier."""

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        num_layers: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.sub_net = SubNet(input_dim, num_layers, hidden_dim)
        self.window_size = window_size
        self.relu = nn.ReLU()
        self.combine_layer = nn.Linear(window_size * hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, x.shape[-1])
        sub_net_output = self.sub_net(x)
        sub_net_output = sub_net_output.view(batch_size, -1)
        combined_features = self.combine_layer(sub_net_output)
        combined_features = self.relu(combined_features)
        output = self.output_layer(combined_features)
        return self.sigmoid(output)


class QueryDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path, label_mode: str):
        self.rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                target = float(row["label"] if label_mode == "hard" else row["score"])
                self.rows.append(
                    {
                        "query_id": int(row["query_id"]),
                        "token_ids": [int(x) for x in row["token_ids"]],
                        "target": target,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reverse-train an attacker generator against the UPV network detector."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "make-query-data",
            "train-reverse",
            "eval-cracking",
            "eval-forgery",
            "summarize",
            "full",
        ],
        default="full",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--upv-root", default=str(DEFAULT_UPV_ROOT))
    parser.add_argument("--detector-checkpoint", default=str(DEFAULT_NETWORK_DETECTOR))
    parser.add_argument("--true-generator", default=str(DEFAULT_UPV_GENERATOR))
    parser.add_argument("--subnet", default=str(DEFAULT_SUBNET))
    parser.add_argument("--query-data", default=None)
    parser.add_argument("--source-query-data", default=str(DEFAULT_UPV_TRAIN_DATA))
    parser.add_argument("--reverse-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-query", type=int, default=10_000)
    parser.add_argument("--query-min-length", type=int, default=100)
    parser.add_argument("--query-max-length", type=int, default=200)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--synthetic-candidate-batch", type=int, default=128)
    parser.add_argument(
        "--query-source",
        choices=["upv-train-data", "synthetic-green-ratio", "uniform-random"],
        default="upv-train-data",
        help=(
            "Appendix-D-style reverse training needs random token lists from the "
            "same synthetic distribution used by the detector. uniform-random is "
            "kept as a diagnostic and often degenerates to all-negative labels."
        ),
    )
    parser.add_argument(
        "--detector-source",
        choices=["network", "zscore"],
        default="network",
        help="D(x) source: trained network detector or saved/key-based z-score.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["hard", "soft"],
        default="hard",
        help="hard uses 1(score > threshold); soft fits the detector score directly.",
    )
    parser.add_argument(
        "--attacker-mode",
        choices=["fixed-subnet", "finetune-subnet", "random"],
        default="fixed-subnet",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-windows", type=int, default=200_000)
    parser.add_argument(
        "--eval-token-max",
        type=int,
        default=None,
        help="Maximum token id for cracking eval. Default is 2**bit_number - 2.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--generation-limit", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--llm-name", default="gpt2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def format_num(n: int) -> str:
    if n >= 1000 and n % 1000 == 0:
        return f"{n // 1000}k"
    return str(n)


def experiment_tag(args: argparse.Namespace) -> str:
    return (
        f"{args.attacker_mode}_{args.detector_source}_{args.query_source}_{args.label_mode}_"
        f"{format_num(args.num_query)}_seed{args.seed}"
    )


def query_data_path(args: argparse.Namespace) -> Path:
    if args.query_data:
        return Path(args.query_data)
    return (
        Path(args.output_dir)
        / "query_data"
        / (
            f"query_{args.detector_source}_{args.query_source}_"
            f"{args.label_mode}_{format_num(args.num_query)}_seed{args.seed}.jsonl"
        )
    )


def reverse_checkpoint_path(args: argparse.Namespace, latest: bool = False) -> Path:
    if args.reverse_checkpoint and not latest:
        return Path(args.reverse_checkpoint)
    suffix = "_latest" if latest else "_best"
    return Path(args.output_dir) / "checkpoints" / f"reverse_{experiment_tag(args)}{suffix}.pt"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def int_to_bin_list(n: int, number: int) -> list[int]:
    return [int(b) for b in format(int(n), "b").zfill(number)]


def ids_to_fixed_feature(
    token_ids: list[int], bit_number: int, sequence_length: int
) -> torch.Tensor:
    return ids_to_feature(token_ids, bit_number, sequence_length)


def token_windows_feature(
    token_ids: list[int], bit_number: int, window_size: int
) -> torch.Tensor:
    if len(token_ids) < window_size:
        raise ValueError("token list is shorter than window_size")
    windows: list[list[list[int]]] = []
    for start in range(0, len(token_ids) - window_size + 1):
        window = token_ids[start : start + window_size]
        windows.append([int_to_bin_list(token_id, bit_number) for token_id in window])
    return torch.tensor(windows, dtype=torch.float32)


def sample_token_ids(
    *,
    query_id: int,
    seed: int,
    min_length: int,
    max_length: int,
    vocab_size: int,
    eos_token_id: int | None,
) -> list[int]:
    rng = random.Random(seed + query_id * 1_000_003)
    length = rng.randint(min_length, max_length)
    token_ids: list[int] = []
    upper = vocab_size - 1
    while len(token_ids) < length:
        token_id = rng.randint(1, upper)
        if eos_token_id is not None and token_id == eos_token_id:
            continue
        token_ids.append(token_id)
    return token_ids


def load_network_detector(args: argparse.Namespace, device: torch.device) -> TransformerClassifier:
    model = TransformerClassifier(args.bit_number, args.layers, 64, 128)
    ckpt = torch.load(args.detector_checkpoint, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_upv_train_rows(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "source": "upv-train-data",
                    "token_ids": [int(x) for x in item["Input"]],
                    "z_score": float(item["Output"]),
                    "green_tags": [int(x) for x in item.get("Tag", [])],
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def compute_z_score(observed_count: int, token_count: int, gamma: float = 0.5) -> float:
    sigma = 0.01
    numer = observed_count - gamma * token_count
    denom = math.sqrt(token_count * gamma * (1 - gamma) + sigma * sigma * token_count)
    return float(numer / denom)


def synthetic_sequence_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir)
        / "synthetic_sequences"
        / f"synthetic_{format_num(args.num_query)}_seed{args.seed}_w{args.window_size}.jsonl"
    )


def synthetic_sequence_key(record: dict[str, Any]) -> str:
    return str(record.get("query_id"))


def sample_matching_token(
    *,
    prefix: list[int],
    want_green: bool,
    true_model: BinaryClassifier,
    rng: random.Random,
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    token_max = 2**args.bit_number - 2
    context = prefix[-(args.window_size - 1) :] if args.window_size > 1 else []
    while True:
        candidates = [
            rng.randint(1, token_max) for _ in range(args.synthetic_candidate_batch)
        ]
        features = []
        for candidate in candidates:
            window = context + [candidate]
            features.append(
                [int_to_bin_list(token_id, args.bit_number) for token_id in window]
            )
        batch = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            decisions = (true_model(batch).reshape(-1) > args.threshold).detach().cpu().tolist()
        for candidate, is_green in zip(candidates, decisions):
            if bool(is_green) == want_green:
                return int(candidate)


def sample_synthetic_green_ratio_sequence(
    *,
    query_id: int,
    true_model: BinaryClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    rng = random.Random(args.seed + query_id * 1_000_003)
    token_max = 2**args.bit_number - 2
    length = rng.randint(args.query_min_length, args.query_max_length)
    green_ratio = rng.random()
    prefix_len = max(args.window_size - 1, 0)
    token_ids = rng.sample(range(1, token_max + 1), prefix_len) if prefix_len else []
    generated_green: list[int] = []
    while len(token_ids) < length:
        want_green = rng.random() < green_ratio
        token = sample_matching_token(
            prefix=token_ids,
            want_green=want_green,
            true_model=true_model,
            rng=rng,
            args=args,
            device=device,
        )
        token_ids.append(token)
        generated_green.append(1 if want_green else 0)
    z_score = compute_z_score(sum(generated_green), len(generated_green))
    return {
        "source": "synthetic-green-ratio",
        "token_ids": token_ids,
        "z_score": z_score,
        "green_ratio": green_ratio,
        "length": length,
        "green_count": sum(generated_green),
        "scored_token_count": len(generated_green),
    }


def init_synthetic_state(query_id: int, args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed + query_id * 1_000_003)
    token_max = 2**args.bit_number - 2
    length = rng.randint(args.query_min_length, args.query_max_length)
    green_ratio = rng.random()
    prefix_len = max(args.window_size - 1, 0)
    token_ids = rng.sample(range(1, token_max + 1), prefix_len) if prefix_len else []
    return {
        "query_id": query_id,
        "rng": rng,
        "length": length,
        "green_ratio": green_ratio,
        "token_ids": token_ids,
        "generated_green": [],
    }


def finalize_synthetic_state(state: dict[str, Any]) -> dict[str, Any]:
    generated_green = [int(x) for x in state["generated_green"]]
    z_score = compute_z_score(sum(generated_green), len(generated_green))
    return {
        "query_id": int(state["query_id"]),
        "source": "synthetic-green-ratio",
        "token_ids": [int(x) for x in state["token_ids"]],
        "z_score": z_score,
        "green_ratio": float(state["green_ratio"]),
        "length": int(state["length"]),
        "green_count": int(sum(generated_green)),
        "scored_token_count": int(len(generated_green)),
    }


def generate_synthetic_sequences_batch(
    *,
    query_ids: list[int],
    true_model: BinaryClassifier,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    token_max = 2**args.bit_number - 2
    states = [init_synthetic_state(query_id, args) for query_id in query_ids]
    active = [idx for idx, state in enumerate(states) if len(state["token_ids"]) < state["length"]]
    while active:
        feature_rows: list[list[list[int]]] = []
        metadata: list[tuple[int, bool, list[int]]] = []
        for state_idx in active:
            state = states[state_idx]
            rng = state["rng"]
            want_green = rng.random() < state["green_ratio"]
            candidates = [
                rng.randint(1, token_max) for _ in range(args.synthetic_candidate_batch)
            ]
            context = (
                state["token_ids"][-(args.window_size - 1) :]
                if args.window_size > 1
                else []
            )
            for candidate in candidates:
                window = context + [candidate]
                feature_rows.append(
                    [int_to_bin_list(token_id, args.bit_number) for token_id in window]
                )
            metadata.append((state_idx, want_green, candidates))

        batch = torch.tensor(feature_rows, dtype=torch.float32, device=device)
        with torch.no_grad():
            decisions = (
                true_model(batch).reshape(len(metadata), args.synthetic_candidate_batch)
                > args.threshold
            ).detach().cpu().tolist()

        next_active: list[int] = []
        for row_idx, (state_idx, want_green, candidates) in enumerate(metadata):
            state = states[state_idx]
            chosen: int | None = None
            for candidate, is_green in zip(candidates, decisions[row_idx]):
                if bool(is_green) == bool(want_green):
                    chosen = int(candidate)
                    break
            if chosen is None:
                # This is extremely unlikely with candidate_batch=128, but keeps
                # generation robust if a batch happens to miss the target label.
                chosen = sample_matching_token(
                    prefix=state["token_ids"],
                    want_green=bool(want_green),
                    true_model=true_model,
                    rng=state["rng"],
                    args=args,
                    device=device,
                )
            state["token_ids"].append(chosen)
            state["generated_green"].append(1 if want_green else 0)
            if len(state["token_ids"]) < state["length"]:
                next_active.append(state_idx)
        active = next_active
    return [finalize_synthetic_state(state) for state in states]


def ensure_synthetic_sequences(args: argparse.Namespace, device: torch.device) -> Path:
    output_path = synthetic_sequence_path(args)
    if output_path.exists() and not args.force:
        existing = completed_keys(output_path, synthetic_sequence_key)
        if len(existing) >= args.num_query:
            return output_path
    ensure_dir(output_path.parent)
    existing_keys = completed_keys(output_path, synthetic_sequence_key) if args.resume else set()
    true_model = load_binary_classifier(args.true_generator, args, device)
    true_model.eval()
    pending_ids = [idx for idx in range(args.num_query) if str(idx) not in existing_keys]
    for start in tqdm(
        range(0, len(pending_ids), args.query_batch_size),
        desc="generate synthetic query",
    ):
        batch_ids = pending_ids[start : start + args.query_batch_size]
        records = generate_synthetic_sequences_batch(
            query_ids=batch_ids,
            true_model=true_model,
            args=args,
            device=device,
        )
        for record in records:
            append_jsonl(output_path, record)
            existing_keys.add(str(record["query_id"]))
    return output_path


def load_binary_classifier(
    path: str | Path,
    args: argparse.Namespace,
    device: torch.device,
) -> BinaryClassifier:
    model = BinaryClassifier(args.bit_number, args.window_size, args.layers)
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def detector_subnet_state(detector: TransformerClassifier) -> dict[str, torch.Tensor]:
    prefix = "binary_classifier."
    state: dict[str, torch.Tensor] = {}
    for key, value in detector.state_dict().items():
        if key.startswith(prefix):
            state[key[len(prefix) :]] = value.detach().cpu()
    return state


def make_reverse_model(
    args: argparse.Namespace,
    detector: TransformerClassifier,
    device: torch.device,
) -> BinaryClassifier:
    model = BinaryClassifier(args.bit_number, args.window_size, args.layers)
    if args.attacker_mode in {"fixed-subnet", "finetune-subnet"}:
        model.sub_net.load_state_dict(detector_subnet_state(detector), strict=True)
    if args.attacker_mode == "fixed-subnet":
        for param in model.sub_net.parameters():
            param.requires_grad = False
    return model.to(device)


def make_query_data(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    output_path = query_data_path(args)
    summary_path = output_path.with_suffix(".summary.json")
    if output_path.exists() and not args.force:
        existing = completed_keys(output_path, lambda r: str(r.get("query_id")))
        if len(existing) >= args.num_query:
            return output_path
    ensure_dir(output_path.parent)
    device = resolve_device(args.device)
    tokenizer = None
    source_rows: list[dict[str, Any]] | None = None
    synthetic_rows: dict[int, dict[str, Any]] | None = None
    if args.query_source == "uniform-random":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    elif args.query_source == "upv-train-data":
        source_rows = load_upv_train_rows(args.source_query_data)
        if len(source_rows) < args.num_query:
            raise ValueError(
                f"--source-query-data has only {len(source_rows)} rows; need {args.num_query}"
            )
        rng = random.Random(args.seed)
        source_rows = rng.sample(source_rows, args.num_query)
    else:
        synthetic_path = ensure_synthetic_sequences(args, device)
        synthetic_rows = {
            int(row["query_id"]): row
            for row in load_existing_payload_records(
                synthetic_path.with_suffix(".summary.json"),
                synthetic_path,
                synthetic_sequence_key,
            )
        }
    detector = load_network_detector(args, device) if args.detector_source == "network" else None
    existing_keys = completed_keys(output_path, lambda r: str(r.get("query_id"))) if args.resume else set()

    pending_ids = [idx for idx in range(args.num_query) if str(idx) not in existing_keys]
    for start in tqdm(range(0, len(pending_ids), args.query_batch_size), desc="label queries"):
        batch_ids = pending_ids[start : start + args.query_batch_size]
        token_lists: list[list[int]] = []
        z_scores: list[float | None] = []
        for query_id in batch_ids:
            if args.query_source == "uniform-random":
                if tokenizer is None:
                    raise RuntimeError("tokenizer was not initialized")
                token_ids = sample_token_ids(
                    query_id=query_id,
                    seed=args.seed,
                    min_length=args.query_min_length,
                    max_length=args.query_max_length,
                    vocab_size=tokenizer.vocab_size,
                    eos_token_id=tokenizer.eos_token_id,
                )
                z_score = None
            elif args.query_source == "upv-train-data":
                if source_rows is None:
                    raise RuntimeError("source_rows was not initialized")
                row = source_rows[query_id]
                token_ids = row["token_ids"]
                z_score = float(row["z_score"])
            else:
                if synthetic_rows is None:
                    raise RuntimeError("synthetic_rows was not initialized")
                row = synthetic_rows[query_id]
                token_ids = [int(x) for x in row["token_ids"]]
                z_score = float(row["z_score"])
            token_lists.append(token_ids)
            z_scores.append(z_score)

        if args.detector_source == "network":
            if detector is None:
                raise RuntimeError("detector was not initialized")
            features = torch.stack(
                [
                    ids_to_fixed_feature(ids, args.bit_number, args.sequence_length)
                    for ids in token_lists
                ]
            ).to(device)
            with torch.no_grad():
                scores = detector(features.float()).reshape(-1).detach().cpu().tolist()
        else:
            if any(z is None for z in z_scores):
                raise ValueError(
                    "--detector-source zscore requires query records with z_score"
                )
            scores = [float(z) for z in z_scores if z is not None]

        for query_id, token_ids, z_score, score in zip(batch_ids, token_lists, z_scores, scores):
            label = int(float(score) > args.threshold)
            append_jsonl(
                output_path,
                {
                    "query_id": int(query_id),
                    "length": len(token_ids),
                    "token_ids": token_ids,
                    "score": float(score),
                    "z_score": z_score,
                    "label": label,
                    "threshold": args.threshold,
                    "label_mode": args.label_mode,
                    "query_source": args.query_source,
                    "detector_source": args.detector_source,
                    "detector": str(args.detector_checkpoint),
                },
            )

    rows = load_existing_payload_records(
        summary_path,
        output_path,
        lambda r: str(r.get("query_id")),
    )
    rows = sorted(rows, key=lambda r: int(r["query_id"]))[: args.num_query]
    scores = [float(r["score"]) for r in rows]
    positives = sum(int(r["label"]) for r in rows)
    write_json(
        summary_path,
        {
            "metadata": {
                "path": str(output_path),
                "num_query": args.num_query,
                "seed": args.seed,
                "query_min_length": args.query_min_length,
                "query_max_length": args.query_max_length,
                "detector_checkpoint": args.detector_checkpoint,
                "detector_source": args.detector_source,
                "query_source": args.query_source,
                "source_query_data": args.source_query_data,
                "true_generator": args.true_generator,
                "threshold": args.threshold,
            },
            "summary": {
                "count": len(rows),
                "positive_count": positives,
                "positive_rate": positives / len(rows) if rows else None,
                "score": summarize_numbers(scores),
            },
        },
    )
    return output_path


def collate_query_batch(
    batch: list[dict[str, Any]],
    *,
    bit_number: int,
    window_size: int,
) -> dict[str, torch.Tensor]:
    windows = [
        token_windows_feature(row["token_ids"], bit_number, window_size) for row in batch
    ]
    lengths = torch.tensor([w.shape[0] for w in windows], dtype=torch.long)
    max_windows = int(lengths.max().item())
    features = torch.zeros(
        len(batch), max_windows, window_size, bit_number, dtype=torch.float32
    )
    mask = torch.zeros(len(batch), max_windows, dtype=torch.bool)
    for idx, item_windows in enumerate(windows):
        n = item_windows.shape[0]
        features[idx, :n] = item_windows
        mask[idx, :n] = True
    targets = torch.tensor([float(row["target"]) for row in batch], dtype=torch.float32)
    return {"features": features, "mask": mask, "targets": targets, "lengths": lengths}


def sequence_probs_from_windows(
    model: BinaryClassifier,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_features = features[mask]
    valid_probs = model(valid_features).reshape(-1)
    seq_probs = torch.zeros(features.shape[0], device=features.device)
    cursor = 0
    counts = mask.sum(dim=1)
    for idx, count_tensor in enumerate(counts):
        count = int(count_tensor.item())
        seq_probs[idx] = valid_probs[cursor : cursor + count].mean()
        cursor += count
    return seq_probs


def train_reverse(args: argparse.Namespace) -> Path:
    query_path = make_query_data(args) if not query_data_path(args).exists() else query_data_path(args)
    device = resolve_device(args.device)
    set_seed(args.seed)
    detector = load_network_detector(args, device)
    model = make_reverse_model(args, detector, device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.BCELoss()
    dataset = QueryDataset(query_path, args.label_mode)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_query_batch(
            b, bit_number=args.bit_number, window_size=args.window_size
        ),
    )

    latest_ckpt = reverse_checkpoint_path(args, latest=True)
    best_ckpt = reverse_checkpoint_path(args, latest=False)
    ensure_dir(latest_ckpt.parent)
    start_epoch = 0
    best_loss = math.inf
    if args.resume and latest_ckpt.exists() and not args.force:
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))

    log_path = Path(args.output_dir) / "train_logs" / f"train_{experiment_tag(args)}.jsonl"
    ensure_dir(log_path.parent)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses: list[float] = []
        labels: list[int] = []
        scores: list[float] = []
        start_time = time.time()
        for batch in tqdm(loader, desc=f"train epoch {epoch}"):
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            seq_probs = sequence_probs_from_windows(model, features, mask)
            loss = loss_fn(seq_probs, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.extend(float(v) for v in seq_probs.detach().cpu().tolist())
            labels.extend(1 if float(v) > args.threshold else 0 for v in targets.detach().cpu().tolist())

        epoch_loss = float(np.mean(losses)) if losses else None
        metrics = compute_binary_metrics(labels, scores, args.threshold)
        metrics.update(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "epoch_time_sec": time.time() - start_time,
                "tag": experiment_tag(args),
                "query_data": str(query_path),
                "attacker_mode": args.attacker_mode,
                "label_mode": args.label_mode,
            }
        )
        append_jsonl(log_path, metrics)
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "best_loss": best_loss,
        }
        torch.save(payload, latest_ckpt)
        if epoch_loss is not None and epoch_loss < best_loss:
            best_loss = epoch_loss
            payload["best_loss"] = best_loss
            torch.save(payload, best_ckpt)

    if not best_ckpt.exists():
        torch.save(
            {
                "epoch": args.epochs - 1,
                "model_state": model.state_dict(),
                "args": vars(args),
                "best_loss": best_loss,
            },
            best_ckpt,
        )
    write_json(
        Path(args.output_dir) / "checkpoints" / f"reverse_{experiment_tag(args)}_summary.json",
        {
            "checkpoint": str(best_ckpt),
            "latest_checkpoint": str(latest_ckpt),
            "query_data": str(query_path),
            "train_log": str(log_path),
            "num_query": args.num_query,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "attacker_mode": args.attacker_mode,
            "label_mode": args.label_mode,
            "seed": args.seed,
        },
    )
    return best_ckpt


def sample_eval_windows(
    args: argparse.Namespace,
    *,
    start_index: int,
    count: int,
) -> torch.Tensor:
    rng = random.Random(args.seed + 91_337)
    token_max = args.eval_token_max if args.eval_token_max is not None else (2**args.bit_number - 2)
    windows = []
    for _ in range(start_index * args.window_size):
        rng.randint(1, token_max)
    for _ in range(count):
        token_ids = [rng.randint(1, token_max) for _ in range(args.window_size)]
        windows.append([int_to_bin_list(token_id, args.bit_number) for token_id in token_ids])
    return torch.tensor(windows, dtype=torch.float32)


def eval_cracking(args: argparse.Namespace) -> Path:
    ckpt_path = reverse_checkpoint_path(args, latest=False)
    if not ckpt_path.exists():
        ckpt_path = train_reverse(args)
    device = resolve_device(args.device)
    reverse_model = load_binary_classifier(ckpt_path, args, device)
    true_model = load_binary_classifier(args.true_generator, args, device)
    labels: list[int] = []
    scores: list[float] = []
    positives = 0
    with torch.no_grad():
        remaining = args.eval_windows
        offset = 0
        pbar = tqdm(total=args.eval_windows, desc="eval cracking")
        while remaining > 0:
            batch_count = min(args.eval_batch_size, remaining)
            features = sample_eval_windows(
                args, start_index=offset, count=batch_count
            ).to(device)
            true_scores = true_model(features).reshape(-1)
            reverse_scores = reverse_model(features).reshape(-1)
            batch_labels = (true_scores > args.threshold).int().detach().cpu().tolist()
            labels.extend(int(v) for v in batch_labels)
            scores.extend(float(v) for v in reverse_scores.detach().cpu().tolist())
            positives += sum(int(v) for v in batch_labels)
            remaining -= batch_count
            offset += batch_count
            pbar.update(batch_count)
        pbar.close()

    metrics = compute_binary_metrics(labels, scores, args.threshold)
    metrics["true_positive_rate_in_windows"] = positives / len(labels) if labels else None
    metrics["score"] = summarize_numbers(scores)
    out_path = (
        Path(args.output_dir)
        / "cracking_eval"
        / f"cracking_{experiment_tag(args)}.json"
    )
    write_json(
        out_path,
        {
            "metadata": {
                "reverse_checkpoint": str(ckpt_path),
                "true_generator": args.true_generator,
                "eval_windows": args.eval_windows,
                "threshold": args.threshold,
                "seed": args.seed,
                "attacker_mode": args.attacker_mode,
                "label_mode": args.label_mode,
                "detector_source": args.detector_source,
                "query_source": args.query_source,
                "num_query": args.num_query,
            },
            "metrics": metrics,
        },
    )
    return out_path


def import_upv(root: str | Path) -> tuple[Any, Any]:
    root = Path(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from watermark_model import CustomLogitsProcessor, WatermarkLogitsProcessor  # type: ignore

    return WatermarkLogitsProcessor, CustomLogitsProcessor


def load_gpt2(model_name_or_path: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, tokenizer


def predict_texts(
    detector: TransformerClassifier,
    tokenizer: Any,
    texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    detector.eval()
    with torch.no_grad():
        for start in range(0, len(texts), 256):
            batch_texts = texts[start : start + 256]
            features = []
            for text in batch_texts:
                token_ids = tokenizer(text or "", return_tensors=None, add_special_tokens=True)[
                    "input_ids"
                ]
                features.append(ids_to_fixed_feature(token_ids, args.bit_number, args.sequence_length))
            batch = torch.stack(features).to(device).float()
            scores.extend(float(v) for v in detector(batch).reshape(-1).detach().cpu().tolist())
    return scores


def eval_forgery(args: argparse.Namespace) -> Path:
    ckpt_path = reverse_checkpoint_path(args, latest=False)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing reverse checkpoint: {ckpt_path}")
    device = resolve_device(args.device)
    set_seed(args.seed)
    WatermarkLogitsProcessor, CustomLogitsProcessor = import_upv(args.upv_root)
    model, tokenizer = load_gpt2(args.model_name_or_path)
    reverse_model = load_binary_classifier(ckpt_path, args, torch.device("cpu"))
    reverse_model.eval()
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        delta=args.delta,
        model=reverse_model,
        window_size=args.window_size,
        cache={},
        bit_number=args.bit_number,
        beam_size=args.beam_size,
        llm_name=args.llm_name,
    )
    custom_processor = CustomLogitsProcessor(llm_name=args.llm_name)
    generation_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
    if args.beam_size > 0:
        generation_kwargs["num_beams"] = args.beam_size
    else:
        generation_kwargs.update(
            {"do_sample": True, "temperature": args.temperature, "top_k": args.top_k}
        )

    prompts = load_eli5_prompts(args.dataset_path, args.generation_limit)
    output_path = (
        Path(args.output_dir)
        / "forged_generation"
        / f"generations_{experiment_tag(args)}.json"
    )
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()
    for item in tqdm(prompts, desc="generate forged"):
        if (
            record_key({**item, "watermarked": True}) in existing_keys
            and record_key({**item, "watermarked": False}) in existing_keys
        ):
            continue
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        for watermarked, processor in [
            (True, watermark_processor),
            (False, custom_processor),
        ]:
            key = record_key({**item, "watermarked": watermarked})
            if key in existing_keys:
                continue
            with Timer() as timer:
                try:
                    output = model.generate(
                        **inputs,
                        logits_processor=LogitsProcessorList([processor]),
                        **generation_kwargs,
                    )
                    completion_ids = output[:, prompt_len:]
                    completion_text = tokenizer.batch_decode(
                        completion_ids, skip_special_tokens=True
                    )[0]
                    full_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                    error = None
                except Exception as exc:
                    completion_text = ""
                    full_text = item["prompt"]
                    error = repr(exc)
            append_jsonl(
                records_jsonl,
                {
                    **item,
                    "method": "upv_reverse_forgery",
                    "watermarked": watermarked,
                    "prompt_template": "raw_title",
                    "completion_text": completion_text,
                    "full_text": full_text,
                    "prompt_token_count": int(prompt_len),
                    "completion_token_count": len(
                        tokenizer.encode(completion_text, add_special_tokens=False)
                    ),
                    "generation_time_sec": timer.elapsed,
                    "method_metadata": {
                        "reverse_checkpoint": str(ckpt_path),
                        "attacker_mode": args.attacker_mode,
                        "label_mode": args.label_mode,
                        "generation_error": error,
                    },
                },
            )
            existing_keys.add(key)

    records = sorted(
        load_existing_payload_records(output_path, records_jsonl, lambda r: record_key(r)),
        key=lambda r: (int(r["sample_id"]), 0 if r["watermarked"] else 1),
    )
    write_json(
        output_path,
        {
            "metadata": {
                "method": "upv_reverse_forgery",
                "reverse_checkpoint": str(ckpt_path),
                "model_name_or_path": args.model_name_or_path,
                "dataset_path": args.dataset_path,
                "generation_config": generation_kwargs,
                "checkpoint_jsonl": str(records_jsonl),
            },
            "records": records,
        },
    )

    detector_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if detector_tokenizer.pad_token is None:
        detector_tokenizer.pad_token = detector_tokenizer.eos_token
    detector = load_network_detector(args, device)
    detection_path = (
        Path(args.output_dir)
        / "forged_detection"
        / f"detection_{experiment_tag(args)}.json"
    )
    detection_jsonl = detection_path.with_suffix(detection_path.suffix + ".records.jsonl")
    detection_keys = completed_keys(detection_jsonl, lambda r: record_key(r)) if args.resume else set()
    pending = [r for r in records if record_key(r) not in detection_keys]
    scores = predict_texts(
        detector,
        detector_tokenizer,
        [r.get("completion_text", "") for r in pending],
        args,
        device,
    )
    for record, score in zip(pending, scores):
        append_jsonl(
            detection_jsonl,
            {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": "upv_reverse_forgery",
                "watermarked": bool(record.get("watermarked")),
                "detected": bool(score > args.threshold),
                "score": float(score),
                "threshold": args.threshold,
                "detector_checkpoint": args.detector_checkpoint,
                "error": None,
            },
        )
    detection_records = sorted(
        load_existing_payload_records(detection_path, detection_jsonl, lambda r: record_key(r)),
        key=lambda r: (int(r["sample_id"]), 0 if r["watermarked"] else 1),
    )
    labels = [1 if bool(r["watermarked"]) else 0 for r in detection_records]
    scores = [float(r["score"]) for r in detection_records]
    write_json(
        detection_path,
        {
            "metadata": {
                "generation": str(output_path),
                "detector_checkpoint": args.detector_checkpoint,
                "threshold": args.threshold,
            },
            "summary": compute_binary_metrics(labels, scores, args.threshold),
            "records": detection_records,
        },
    )
    return detection_path


def summarize(args: argparse.Namespace) -> Path:
    root = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "cracking_eval").glob("cracking_*.json")):
        obj = json.load(open(path, "r", encoding="utf-8"))
        meta = obj.get("metadata", {})
        metrics = obj.get("metrics", {})
        rows.append(
            {
                "file": str(path),
                "detector_source": meta.get("detector_source"),
                "query_source": meta.get("query_source"),
                "attacker_mode": meta.get("attacker_mode"),
                "label_mode": meta.get("label_mode"),
                "seed": meta.get("seed"),
                "eval_windows": meta.get("eval_windows"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("tp", 0)
                / (metrics.get("tp", 0) + metrics.get("fp", 0))
                if (metrics.get("tp", 0) + metrics.get("fp", 0))
                else None,
                "recall": metrics.get("tpr"),
                "f1": metrics.get("f1"),
                "tpr": metrics.get("tpr"),
                "fpr": metrics.get("fpr"),
                "true_window_positive_rate": metrics.get("true_positive_rate_in_windows"),
            }
        )
    out_json = root / "summary" / "reverse_training_summary.json"
    write_json(out_json, {"rows": rows})
    out_md = root / "summary" / "reverse_training_summary.md"
    ensure_dir(out_md.parent)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# UPV Reverse Training Summary\n\n")
        f.write("| detector | query | attacker | label | seed | windows | accuracy | F1 | TPR | FPR |\n")
        f.write("|---|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {detector_source} | {query_source} | {attacker_mode} | {label_mode} | {seed} | {eval_windows} | "
                "{accuracy:.4f} | {f1:.4f} | {tpr:.4f} | {fpr:.4f} |\n".format(
                    **{
                        **row,
                        "detector_source": row["detector_source"] or "",
                        "query_source": row["query_source"] or "",
                        "accuracy": row["accuracy"] or 0.0,
                        "f1": row["f1"] or 0.0,
                        "tpr": row["tpr"] or 0.0,
                        "fpr": row["fpr"] or 0.0,
                    }
                )
            )
    return out_json


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.mode == "make-query-data":
        make_query_data(args)
    elif args.mode == "train-reverse":
        train_reverse(args)
    elif args.mode == "eval-cracking":
        eval_cracking(args)
    elif args.mode == "eval-forgery":
        eval_forgery(args)
    elif args.mode == "summarize":
        summarize(args)
    elif args.mode == "full":
        make_query_data(args)
        train_reverse(args)
        eval_cracking(args)
        summarize(args)


if __name__ == "__main__":
    main()
