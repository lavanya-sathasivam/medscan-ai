import torch
import numpy as np
import cv2
from importlib import import_module

GradCAM = None
show_cam_on_image = None
try:
    GradCAM = import_module("pytorch_grad_cam").GradCAM
    show_cam_on_image = import_module("pytorch_grad_cam.utils.image").show_cam_on_image
except Exception:  # pragma: no cover - optional runtime dependency
    GradCAM = None
    show_cam_on_image = None


class GradCAMPlusPlus:
    """
    Stable Grad-CAM++ for medical imaging
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        target = output[:, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed")

        gradients = self.gradients[0]        # [C,H,W]
        activations = self.activations[0]    # [C,H,W]

        # Grad-CAM++
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)

        eps = 1e-8
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + eps)

        positive_gradients = torch.relu(gradients)
        weights = torch.sum(alpha * positive_gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        # ✅ Normalize safely
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam.cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# 🔥 MAIN FUNCTION
def _select_target_layer(model):
    # Match your Colab setup for ResNet-style models first.
    if hasattr(model, "layer3"):
        return model.layer3[-1]
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Unsupported model architecture for Grad-CAM")


def _normalize_heatmap(heatmap: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0)
    if float(heatmap.max()) > 0:
        heatmap = heatmap / float(heatmap.max())
    heatmap = np.power(heatmap, 2.2)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap = cv2.resize(heatmap, (image_shape[1], image_shape[0]))
    heatmap[heatmap < 0.25] = 0
    return heatmap


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def generate_gradcam_overlay(model, input_tensor, image):
    """Return Grad-CAM visualization using the same flow as the Colab code."""

    if hasattr(model, "layer3"):
        target_layer = model.layer3[-1]
    else:
        target_layer = _select_target_layer(model)

    if GradCAM is not None:
        with GradCAM(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]
    else:
        gradcam = GradCAMPlusPlus(model, target_layer)
        try:
            grayscale_cam = gradcam.generate(input_tensor)
        finally:
            gradcam.close()

    rgb_image = _to_uint8_rgb(image)
    rgb_224 = cv2.resize(rgb_image, (224, 224))
    rgb_float = np.float32(rgb_224) / 255.0

    if show_cam_on_image is not None:
        visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    else:
        heatmap = _normalize_heatmap(grayscale_cam, rgb_224.shape)
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        visualization = cv2.addWeighted(rgb_224, 0.5, heatmap_rgb, 0.5, 0)

    return cv2.resize(visualization, (rgb_image.shape[1], rgb_image.shape[0]))