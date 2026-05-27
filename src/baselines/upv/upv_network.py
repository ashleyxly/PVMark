from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from baseline_eval.common import add_repo_to_path


def token_ids_to_bit_features(
    token_ids: torch.Tensor,
    bit_number: int,
    fixed_length: int = 200,
) -> torch.Tensor:
    ids = token_ids.detach().flatten().to(torch.long).cpu()
    if ids.numel() > fixed_length:
        ids = ids[:fixed_length]
    elif ids.numel() < fixed_length:
        ids = torch.nn.functional.pad(ids, (0, fixed_length - ids.numel()), value=0)
    shifts = torch.arange(bit_number - 1, -1, -1, dtype=torch.long)
    return ((ids.unsqueeze(-1) >> shifts) & 1).float()


class UpvNetworkDetector:
    def __init__(
        self,
        *,
        upv_root: str,
        detector_model: str,
        tokenizer_path: str,
        bit_number: int = 16,
        layers: int = 5,
        fixed_length: int = 200,
        threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        add_repo_to_path(upv_root)
        from detector import TransformerClassifier
        from transformers import AutoTokenizer

        self.bit_number = bit_number
        self.fixed_length = fixed_length
        self.threshold = threshold
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.model = TransformerClassifier(bit_number, layers, 64, 128).to(device)

        checkpoint = torch.load(detector_model, map_location=device)
        state_dict: dict[str, Any]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    def score_ids(self, token_ids: torch.Tensor) -> tuple[float, bool]:
        features = token_ids_to_bit_features(token_ids, self.bit_number, self.fixed_length)
        with torch.inference_mode():
            score = self.model(features.unsqueeze(0).to(self.device)).reshape(-1)[0]
        value = float(score.detach().cpu().item())
        return value, value > self.threshold

    def score_text(self, text: str) -> tuple[float, bool]:
        if not text:
            return 0.0, False
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"].squeeze(0)
        return self.score_ids(ids)

