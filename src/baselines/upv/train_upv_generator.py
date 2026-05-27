from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baseline_eval.common import DEFAULT_UPV_ROOT, add_repo_to_path, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the UPV generator/key network without hard-coded CUDA device.")
    parser.add_argument("--upv-root", default=DEFAULT_UPV_ROOT)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bit-number", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20242024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    add_repo_to_path(args.upv_root)
    from model_key import get_model, load_data

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = get_model(args.bit_number, args.window_size, None, args.layers).to(device)
    features, labels = load_data(args.data_dir)
    train_data = TensorDataset(torch.from_numpy(np.array(features)), torch.from_numpy(np.array(labels)))
    train_loader: DataLoader[Any] = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        last_loss = None
        for inputs, targets in train_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {last_loss:.4f}")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "combine_model.pt"))
    torch.save(model.sub_net.state_dict(), os.path.join(args.model_dir, "sub_net.pt"))


if __name__ == "__main__":
    main()

