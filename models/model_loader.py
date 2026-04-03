from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    resolved_path = Path(model_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model weights not found at {resolved_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(resolved_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
