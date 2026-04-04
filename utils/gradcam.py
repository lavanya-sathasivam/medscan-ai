import torch
import numpy as np
import cv2


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for better localization (especially medical images)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generates Grad-CAM++ heatmap
        """

        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        target = output[:, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed: gradients or activations not found")

        gradients = self.gradients[0]        # [C, H, W]
        activations = self.activations[0]    # [C, H, W]

        # Grad-CAM++ calculations
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)

        eps = 1e-8
        alpha = grad_2 / (2 * grad_2 + sum_activations * grad_3 + eps)

        positive_gradients = torch.relu(gradients)
        weights = torch.sum(alpha * positive_gradients, dim=(1, 2))

        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)

        # Normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam = cam.cpu().numpy()

        return cam

    def overlay(self, image, cam):
        """
        Overlay heatmap on image
        image: RGB image (numpy array, 224x224)
        cam: heatmap (0–1)
        """

        # Resize CAM to image size
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Ensure image is in correct range
        if image.max() <= 1.0:
            image = image * 255

        image = image.astype(np.uint8)

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend image and heatmap
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.5, 0)

        return overlay

    def close(self):
        """
        Remove hooks (important for avoiding memory leaks)
        """
        self.forward_handle.remove()
        self.backward_handle.remove()