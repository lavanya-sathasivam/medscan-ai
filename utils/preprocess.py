from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ]
)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    pil_image = Image.fromarray(image).convert("RGB")
    return transform(pil_image).unsqueeze(0)
