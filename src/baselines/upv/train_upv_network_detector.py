from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baseline_eval.common import DEFAULT_GENERATION_MODEL, DEFAULT_UPV_ROOT, add_repo_to_path, ensure_dir, set_seed, write_json
from baseline_eval.upv_network import token_ids_to_bit_features


class UpvTrainDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str, bit_number: int, fixed_length: int, z_value: float) -> None:
        self.rows: list[tuple[torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = torch.tensor(row["Input"], dtype=torch.long)
                label = 1.0 if float(row["Output"]) > z_value else 0.0
                self.rows.append((token_ids_to_bit_features(ids, bit_number, fixed_length), torch.tensor(label)))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rows[index]


class UpvTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str,
        tokenizer_path: str,
        bit_number: int,
        fixed_length: int,
    ) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                encoded = tokenizer(row["Input"], return_tensors="pt", add_special_tokens=True)
                ids = encoded["input_ids"].squeeze(0)
                self.rows.append(
                    (
                        token_ids_to_bit_features(ids, bit_number, fixed_length),
                        torch.tensor(float(row["Tag"])),
                        torch.tensor(float(row.get("Z-score", 0.0))),
                    )
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UPV network-based detector and save the full detector.")
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", default=None)
    parser.add_argument("--tokenizer", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--sub-net", required=True, help="Generator sub_net.pt used to initialize the detector embedding.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--fixed-length", type=int, default=200)
    parser.add_argument("--z-value", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_i = (pred > 0.5).int()
    target_i = target.int()
    total = max(1, int(target_i.numel()))
    tp = int(((pred_i == 1) & (target_i == 1)).sum().item())
    fp = int(((pred_i == 1) & (target_i == 0)).sum().item())
    fn = int(((pred_i == 0) & (target_i == 1)).sum().item())
    tn = int(((pred_i == 0) & (target_i == 0)).sum().item())
    return {
        "accuracy": (tp + tn) / total,
        "tpr": tp / (tp + fn) if (tp + fn) else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "tnr": tn / (fp + tn) if (fp + tn) else 0.0,
        "fnr": fn / (tp + fn) if (tp + fn) else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def evaluate(model: torch.nn.Module, loader: DataLoader[Any], device: str, loss_fn: torch.nn.Module) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 2:
                inputs, target = batch
            else:
                inputs, target, _ = batch
            inputs = inputs.float().to(device)
            target = target.float().to(device)
            output = model(inputs).reshape(-1)
            loss = loss_fn(output, target)
            losses.append(float(loss.detach().cpu().item()))
            preds.append(output.detach().cpu())
            targets.append(target.detach().cpu())
    pred = torch.cat(preds) if preds else torch.empty(0)
    target = torch.cat(targets) if targets else torch.empty(0)
    return {
        "loss": sum(losses) / len(losses) if losses else None,
        **compute_metrics(pred, target),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    add_repo_to_path(args.upv_root)
    from detector import TransformerClassifier

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = ensure_dir(args.output_dir)
    train_dataset = UpvTrainDataset(args.train_data, args.bit_number, args.fixed_length, args.z_value)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = None
    if args.test_data:
        test_dataset = UpvTextDataset(args.test_data, args.tokenizer, args.bit_number, args.fixed_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TransformerClassifier(args.bit_number, args.layers, 64, 128).to(device)
    sub_state = torch.load(args.sub_net, map_location=device)
    model.binary_classifier.load_state_dict(sub_state, strict=True)
    for param in model.binary_classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()

    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for inputs, target in tqdm(train_loader, desc=f"detector train epoch {epoch + 1}/{args.epochs}"):
            inputs = inputs.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs).reshape(-1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses) if losses else None,
        }
        if test_loader is not None:
            row["test"] = evaluate(model, test_loader, device, loss_fn)
        history.append(row)
        write_json(out_dir / "training_history.json", history)

    final_train = evaluate(model, DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False), device, loss_fn)
    final_test = evaluate(model, test_loader, device, loss_fn) if test_loader is not None else None
    detector_path = out_dir / "network_detector.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "bit_number": args.bit_number,
                "layers": args.layers,
                "fixed_length": args.fixed_length,
                "z_value": args.z_value,
                "sub_net": args.sub_net,
                "train_data": args.train_data,
                "test_data": args.test_data,
                "tokenizer": args.tokenizer,
                "epochs": args.epochs,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
        detector_path,
    )
    summary = {
        "detector_model": str(detector_path),
        "elapsed_sec": time.time() - started,
        "num_train": len(train_dataset),
        "final_train": final_train,
        "final_test": final_test,
        "last_5_test_avg": None,
    }
    last5 = [row["test"] for row in history[-5:] if "test" in row]
    if last5:
        keys = ["accuracy", "tpr", "fpr", "tnr", "fnr", "f1"]
        summary["last_5_test_avg"] = {key: sum(float(row[key]) for row in last5) / len(last5) for key in keys}
    write_json(out_dir / "detector_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

