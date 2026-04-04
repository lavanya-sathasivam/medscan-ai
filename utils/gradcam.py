from __future__ import annotations

import cv2
import numpy as np
import torch


class GradCAM:
    """Improved Grad-CAM with better localization and stability."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        output[:, class_idx].sum().backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed: no gradients/activations")

        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)

        if cam.max() != 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (224, 224))
        cam = cv2.GaussianBlur(cam, (5,5), 0)

        return cam

    def overlay(self, image: np.ndarray, cam: np.ndarray) -> np.ndarray:
        """Overlay heatmap on original image"""
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = heatmap * 0.4 + image * 0.6
        return np.uint8(overlay)

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()