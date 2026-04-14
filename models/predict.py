from __future__ import annotations

import numpy as np
import torch

CLASS_NAMES = ("hemorrhage", "normal")


def predict_image(model: torch.nn.Module, input_tensor: torch.Tensor) -> tuple[str, float, np.ndarray]:
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    class_names = tuple(getattr(model, "class_names", CLASS_NAMES))
    predicted_index = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_index].item())
    return class_names[predicted_index], confidence, probabilities[0].cpu().numpy()
