from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def _build_model_from_state_dict(state_dict: dict[str, torch.Tensor], num_classes: int) -> torch.nn.Module:
    # EfficientNet-style checkpoints expose a top-level "features.*" backbone.
    if any(key.startswith("features.") for key in state_dict):
        model = models.efficientnet_b3(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(1536, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
        return model

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    resolved_path = Path(model_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model weights not found at {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location=device)
    class_names = ("hemorrhage", "normal")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("class_to_idx"), dict):
        idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}
        class_names = tuple(idx_to_class[idx] for idx in sorted(idx_to_class))

    model = _build_model_from_state_dict(state_dict, num_classes=len(class_names))

    # Strip DataParallel prefix if present.
    if isinstance(state_dict, dict) and any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.class_names = class_names
    model.to(device)
    model.eval()
    return model
