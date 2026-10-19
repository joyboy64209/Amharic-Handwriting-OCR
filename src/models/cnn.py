"""CNN baseline for isolated Amharic character recognition.

A re-implementation of the upstream AHCNN using standard ``nn.Conv2d`` /
``nn.BatchNorm2d`` modules (instead of raw ``nn.Parameter`` + ``F.conv2d``).
Architecture: 4 conv blocks -> global average pooling -> linear classifier.

Input:  (1, 1, 28, 28) grayscale.
Output: (1, num_classes) logits.

Checkpoint compatibility
------------------------
The supplied checkpoint ``models/amharic_ocr_v2.pth`` was saved from the
upstream ``AHCNN`` whose state-dict keys look like::

    block1.weight, block1.bias, block1.bn.weight, block1.bn.bias,
    block1.bn.running_mean, block1.bn.running_var, ...
    classifier.weight, classifier.bias

This module's keys are::

    block1.conv.weight, block1.conv.bias, block1.bn.weight, ...
    classifier.weight, classifier.bias

``load_checkpoint`` remaps ``blockN.weight`` -> ``blockN.conv.weight`` and
``blockN.bias`` -> ``blockN.conv.bias`` so the upstream checkpoint loads
verbatim into this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class AHCNN(nn.Module):
    def __init__(self, num_classes: int = 237):
        super().__init__()
        self.block1 = ConvBlock(1, 32, 0.1)    # 32x14x14
        self.block2 = ConvBlock(32, 64, 0.15)  # 64x7x7
        self.block3 = ConvBlock(64, 128, 0.2)  # 128x3x3
        self.block4 = ConvBlock(128, 256, 0.25)  # 256x3x3 (no pool)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    """Map upstream AHCNN parameter names to this module's names."""
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for block in ("block1", "block2", "block3", "block4"):
            if key == f"{block}.weight":
                new_key = f"{block}.conv.weight"
            elif key == f"{block}.bias":
                new_key = f"{block}.conv.bias"
        remapped[new_key] = value
    return remapped


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> dict:
    """Load a checkpoint into ``model`` and return the metadata dict.

    Handles both the upstream checkpoint format (with ``class_to_idx``) and a
    plain state-dict. Returns whatever metadata keys the checkpoint carried
    (e.g. ``class_to_idx``, ``input_shape``, ``final_accuracy``).
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = _remap_checkpoint_keys(checkpoint["model_state_dict"])
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        state_dict = _remap_checkpoint_keys(checkpoint if isinstance(checkpoint, dict) else {})
        metadata = {}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[cnn] Warning: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"[cnn] Warning: unexpected keys in checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return metadata


def build_model(num_classes: int, checkpoint_path: str | None = None,
                device: str = "cpu") -> tuple[nn.Module, dict]:
    """Build an AHCNN and optionally load a checkpoint.

    Returns (model, metadata). If no checkpoint is given or it cannot be
    loaded, the model is returned with random weights and metadata is empty;
    callers should check metadata before serving predictions.
    """
    model = AHCNN(num_classes=num_classes)
    metadata: dict = {}
    if checkpoint_path:
        try:
            metadata = load_checkpoint(model, checkpoint_path, device=device)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[cnn] Could not load checkpoint {checkpoint_path}: {exc}")
    return model, metadata


__all__ = ["AHCNN", "ConvBlock", "load_checkpoint", "build_model"]