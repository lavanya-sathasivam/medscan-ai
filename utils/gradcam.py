from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Generate Grad-CAM heatmaps for convolutional image classifiers."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        output[:, class_idx].sum().backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        weights = gradients.mean(axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for index, weight in enumerate(weights):
            cam += weight * activations[index]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam /= np.max(cam)

        return cv2.resize(cam, (224, 224))

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()
