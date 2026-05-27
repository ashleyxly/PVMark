from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from common import (
    DEFAULT_GPT2_PATH,
    DEFAULT_UPV_ROOT,
    Timer,
    append_jsonl,
    completed_keys,
    ensure_dir,
    load_existing_payload_records,
    record_key,
    sort_records,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("tests/baseline_comparison/upv_network_detector_gpt2_eli5")
DEFAULT_UPV_MAIN = DEFAULT_UPV_ROOT / "experiments/main_experiments"
DEFAULT_TRAIN_DATA = DEFAULT_UPV_MAIN / "train_and_test_data/train_data.jsonl"
DEFAULT_SUBNET = DEFAULT_UPV_MAIN / "generator_model/sub_net.pt"
DEFAULT_GENERATIONS = Path("tests/baseline_comparison/upv_gpt2/generations.json")
DEFAULT_ATTACKS = Path("tests/baseline_comparison/upv_gpt2/attacks.json")


class SubNet(nn.Module):
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


class TransformerClassifier(nn.Module):
    """Same architecture as UPV detector.py:TransformerClassifier."""

    def __init__(
        self,
        bit_number: int,
        b_layers: int,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.binary_classifier = SubNet(bit_number, b_layers)
        self.classifier = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x1 = x.view(batch_size * seq_len, -1)
        features = self.binary_classifier(x1)
        features = features.view(batch_size, seq_len, -1)
        output, _ = self.classifier(features)
        output = self.fc_hidden(output[:, -1, :])
        output = self.sigmoid(output)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the UPV network-based detector for GPT-2."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "paper-eval", "eli5-eval", "attack-eval"],
        default="full",
    )
    parser.add_argument("--upv-root", default=str(DEFAULT_UPV_ROOT))
    parser.add_argument("--train-data", default=str(DEFAULT_TRAIN_DATA))
    parser.add_argument("--subnet", default=str(DEFAULT_SUBNET))
    parser.add_argument("--model-name-or-path", default=str(DEFAULT_GPT2_PATH))
    parser.add_argument("--generations", default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--attacks", default=str(DEFAULT_ATTACKS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--z-value", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def int_to_bin_list(n: int, number: int) -> list[int]:
    return [int(b) for b in format(int(n), "b").zfill(number)]


def pad_or_truncate(features: list[list[int]], length: int, bit_number: int) -> torch.Tensor:
    if len(features) >= length:
        return torch.tensor(features[:length], dtype=torch.float32)
    padded = features + [[0] * bit_number for _ in range(length - len(features))]
    return torch.tensor(padded, dtype=torch.float32)


def ids_to_feature(token_ids: list[int], bit_number: int, length: int) -> torch.Tensor:
    features = [int_to_bin_list(int(token_id), bit_number) for token_id in token_ids]
    return pad_or_truncate(features, length, bit_number)


class SyntheticZScoreDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, bit_number: int, length: int, z_value: float):
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                token_ids = [int(v) for v in item["Input"]]
                label = 1.0 if float(item["Output"]) > z_value else 0.0
                self.rows.append(
                    (
                        ids_to_feature(token_ids, bit_number, length),
                        torch.tensor(label, dtype=torch.float32),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rows[idx]


class PaperTestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        bit_number: int,
        length: int,
    ):
        self.rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                token_ids = tokenizer(
                    item["Input"], return_tensors=None, add_special_tokens=True
                )["input_ids"]
                self.rows.append(
                    (
                        ids_to_feature(token_ids, bit_number, length),
                        torch.tensor(float(item["Tag"]), dtype=torch.float32),
                        torch.tensor(float(item["Z-score"]), dtype=torch.float32),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rows[idx]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def make_model(args: argparse.Namespace, device: torch.device) -> TransformerClassifier:
    model = TransformerClassifier(args.bit_number, args.layers, 64, 128)
    subnet_state = torch.load(args.subnet, map_location="cpu")
    model.binary_classifier.load_state_dict(subnet_state, strict=True)
    for param in model.binary_classifier.parameters():
        param.requires_grad = False
    return model.to(device)


def checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    return Path(args.output_dir) / f"detector_z{format_z(args.z_value)}.pt"


def latest_checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / f"detector_z{format_z(args.z_value)}_latest.pt"


def format_z(z_value: float) -> str:
    if float(z_value).is_integer():
        return str(int(z_value))
    return str(z_value).replace(".", "p")


def compute_binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        pred = 1 if score > threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    total = tp + fp + tn + fn
    return {
        "count": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / total if total else None,
        "tpr": tp / (tp + fn) if (tp + fn) else None,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "tnr": tn / (fp + tn) if (fp + tn) else None,
        "fnr": fn / (tp + fn) if (tp + fn) else None,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None,
    }


def evaluate_dataset(
    model: TransformerClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    z_scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            features, targets, batch_z = batch
            probs = model(features.to(device).float()).reshape(-1).detach().cpu().tolist()
            scores.extend(float(v) for v in probs)
            labels.extend(int(v) for v in targets.tolist())
            z_scores.extend(float(v) for v in batch_z.tolist())
    metrics = compute_binary_metrics(labels, scores, threshold)
    metrics["mean_score"] = float(np.mean(scores)) if scores else None
    metrics["mean_z_score"] = float(np.mean(z_scores)) if z_scores else None
    return metrics


def train(args: argparse.Namespace) -> Path:
    output_dir = ensure_dir(args.output_dir)
    final_ckpt = checkpoint_path(args)
    latest_ckpt = latest_checkpoint_path(args)
    if final_ckpt.exists() and not args.force_train:
        return final_ckpt

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = SyntheticZScoreDataset(
        args.train_data, args.bit_number, args.sequence_length, args.z_value
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    model = make_model(args, device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.BCELoss()

    start_epoch = 0
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt["epoch"]) + 1

    train_log_jsonl = output_dir / f"train_log_z{format_z(args.z_value)}.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses: list[float] = []
        labels: list[int] = []
        scores: list[float] = []
        epoch_start = time.time()
        for features, targets in loader:
            features = features.to(device).float()
            targets = targets.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features).reshape(-1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            scores.extend(float(v) for v in outputs.detach().cpu().tolist())
            labels.extend(int(v) for v in targets.detach().cpu().tolist())

        train_metrics = compute_binary_metrics(labels, scores, args.threshold)
        train_metrics["loss"] = float(np.mean(losses)) if losses else None
        train_metrics["epoch"] = epoch
        train_metrics["epoch_time_sec"] = time.time() - epoch_start
        append_jsonl(train_log_jsonl, train_metrics)

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": vars(args),
                "train_metrics": train_metrics,
            },
            latest_ckpt,
        )

    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state": model.state_dict(),
            "args": vars(args),
            "source": "UPV detector.py architecture, trained with z_value labels",
        },
        final_ckpt,
    )
    write_json(
        output_dir / f"train_summary_z{format_z(args.z_value)}.json",
        {
            "checkpoint": str(final_ckpt),
            "latest_checkpoint": str(latest_ckpt),
            "train_data": args.train_data,
            "subnet": args.subnet,
            "z_value": args.z_value,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "seed": args.seed,
        },
    )
    return final_ckpt


def load_detector(args: argparse.Namespace, device: torch.device) -> TransformerClassifier:
    model = make_model(args, device)
    ckpt = torch.load(checkpoint_path(args), map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_tokenizer(path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def paper_test_paths(root: Path) -> dict[str, Path]:
    base = root / "train_and_test_data/gpt2"
    return {
        "c4_topk": base / "c4_topk/test_data.jsonl",
        "c4_8beams": base / "c4_8beams/test_data.jsonl",
        "dbpedia_topk": base / "dbpedia_topk/test_data.jsonl",
        "dbpedia_8beams": base / "dbpedia_8beams/test_data.jsonl",
    }


def evaluate_paper_tests(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_detector(args, device)
    output: dict[str, Any] = {
        "metadata": {
            "checkpoint": str(checkpoint_path(args)),
            "model_name_or_path": args.model_name_or_path,
            "z_value": args.z_value,
            "threshold": args.threshold,
        },
        "results": {},
    }
    for name, path in paper_test_paths(DEFAULT_UPV_MAIN).items():
        dataset = PaperTestDataset(path, tokenizer, args.bit_number, args.sequence_length)
        loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        output["results"][name] = evaluate_dataset(model, loader, device, args.threshold)
        output["results"][name]["test_data"] = str(path)
    out_path = Path(args.output_dir) / f"paper_gpt2_eval_z{format_z(args.z_value)}.json"
    write_json(out_path, output)
    return out_path


def predict_texts(
    model: TransformerClassifier,
    tokenizer: Any,
    texts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), args.eval_batch_size):
            batch_texts = texts[start : start + args.eval_batch_size]
            features = []
            for text in batch_texts:
                token_ids = tokenizer(text or "", return_tensors=None, add_special_tokens=True)[
                    "input_ids"
                ]
                features.append(ids_to_feature(token_ids, args.bit_number, args.sequence_length))
            batch = torch.stack(features).to(device).float()
            probs = model(batch).reshape(-1).detach().cpu().tolist()
            scores.extend(float(v) for v in probs)
    return scores


def detection_rate(records: list[dict[str, Any]], watermarked: bool) -> dict[str, Any]:
    subset = [r for r in records if bool(r.get("watermarked")) == watermarked]
    valid = [r for r in subset if r.get("detected") is not None]
    positives = [r for r in valid if bool(r.get("detected"))]
    return {
        "count": len(subset),
        "valid_count": len(valid),
        "error_count": len(subset) - len(valid),
        "positive_count": len(positives),
        "positive_rate": len(positives) / len(valid) if valid else None,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if bool(r.get("watermarked")) else 0 for r in records if r.get("detected") is not None]
    scores = [float(r.get("score")) for r in records if r.get("detected") is not None]
    return {
        "wm_detection": detection_rate(records, True),
        "uwm_false_positive": detection_rate(records, False),
        "binary_metrics": compute_binary_metrics(labels, scores, 0.5),
    }


def evaluate_eli5_generations(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_detector(args, device)
    with open(args.generations, "r", encoding="utf-8") as f:
        payload = json.load(f)

    output_path = Path(args.output_dir) / f"detection_network_z{format_z(args.z_value)}.json"
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = completed_keys(records_jsonl, lambda r: record_key(r)) if args.resume else set()

    pending = [r for r in payload["records"] if record_key(r) not in existing_keys]
    texts = [r.get("completion_text", "") for r in pending]
    scores = predict_texts(model, tokenizer, texts, args, device) if pending else []
    for record, score in zip(pending, scores):
        with Timer() as t:
            detected = bool(score > args.threshold)
        append_jsonl(
            records_jsonl,
            {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": "upv_network",
                "source_method": record.get("method"),
                "watermarked": bool(record.get("watermarked")),
                "detected": detected,
                "score": score,
                "threshold": args.threshold,
                "training_z_value": args.z_value,
                "detection_time_sec": t.elapsed,
                "error": None,
            },
        )

    records = sort_records(
        load_existing_payload_records(output_path, records_jsonl, lambda r: record_key(r))
    )
    write_json(
        output_path,
        {
            "metadata": {
                "source": args.generations,
                "method": "upv_network",
                "detector": "TransformerClassifier",
                "checkpoint": str(checkpoint_path(args)),
                "threshold": args.threshold,
                "training_z_value": args.z_value,
                "model_name_or_path": args.model_name_or_path,
            },
            "records": records,
        },
    )
    write_json(
        Path(args.output_dir) / f"summary_network_z{format_z(args.z_value)}.json",
        {
            "metadata": {"detection": str(output_path), "source": args.generations},
            "summary": summarize_records(records),
        },
    )
    return output_path


def evaluate_attacks(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.model_name_or_path)
    model = load_detector(args, device)
    with open(args.attacks, "r", encoding="utf-8") as f:
        payload = json.load(f)

    output_path = Path(args.output_dir) / f"attack_detection_network_z{format_z(args.z_value)}.json"
    records_jsonl = output_path.with_suffix(output_path.suffix + ".records.jsonl")
    existing_keys = (
        completed_keys(records_jsonl, lambda r: record_key(r, r.get("attack")))
        if args.resume
        else set()
    )

    pending: list[tuple[dict[str, Any], str, str]] = []
    for record in payload["records"]:
        for attack_name, text in record.get("attacks", {}).items():
            key = record_key(record, attack_name)
            if key not in existing_keys:
                pending.append((record, attack_name, text))

    scores = predict_texts(model, tokenizer, [p[2] for p in pending], args, device) if pending else []
    for (record, attack_name, _text), score in zip(pending, scores):
        with Timer() as t:
            detected = bool(score > args.threshold)
        append_jsonl(
            records_jsonl,
            {
                "sample_id": record.get("sample_id"),
                "q_id": record.get("q_id"),
                "method": "upv_network",
                "source_method": record.get("method"),
                "attack": attack_name,
                "watermarked": bool(record.get("watermarked")),
                "detected": detected,
                "score": score,
                "threshold": args.threshold,
                "training_z_value": args.z_value,
                "detection_time_sec": t.elapsed,
                "error": None,
            },
        )

    records = sort_records(
        load_existing_payload_records(output_path, records_jsonl, lambda r: record_key(r, r.get("attack")))
    )
    write_json(
        output_path,
        {
            "metadata": {
                "source": args.attacks,
                "method": "upv_network",
                "detector": "TransformerClassifier",
                "checkpoint": str(checkpoint_path(args)),
                "threshold": args.threshold,
                "training_z_value": args.z_value,
                "model_name_or_path": args.model_name_or_path,
            },
            "records": records,
        },
    )

    summary: dict[str, Any] = {}
    for attack_name in sorted({r.get("attack") for r in records}):
        attack_records = [r for r in records if r.get("attack") == attack_name]
        summary[str(attack_name)] = summarize_records(attack_records)
    write_json(
        Path(args.output_dir) / f"robustness_summary_network_z{format_z(args.z_value)}.json",
        {
            "metadata": {"detection": str(output_path), "source": args.attacks},
            "summary": summary,
        },
    )
    return output_path


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.mode in {"full", "train"}:
        train(args)
    if args.mode in {"full", "paper-eval"}:
        evaluate_paper_tests(args)
    if args.mode in {"full", "eli5-eval"}:
        evaluate_eli5_generations(args)
    if args.mode in {"full", "attack-eval"}:
        evaluate_attacks(args)


if __name__ == "__main__":
    main()
