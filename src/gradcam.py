"""FCVG101 Coursework - Task 2.3: Explainable AI via Grad-CAM (Gradient-weighted Class Activation Mapping).
Computes visual attention heatmaps indicating which spatial regions most influence CNN classification decisions.
"""
from typing import Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class GradCAM:
    """Computes Grad-CAM heatmaps for a specified convolutional layer in PyTorch models."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks with handles for safe removal
        self.fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Safely removes registered forward and backward hooks to prevent hook leaks."""
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
            self.fwd_handle = None
        if self.bwd_handle is not None:
            self.bwd_handle.remove()
            self.bwd_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generates 2D Grad-CAM heatmap normalized to [0, 1]."""
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        # Global average pooling on gradients: alpha_k = (1 / Z) * sum(grad_k)
        # self.gradients: [batch, channels, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Linear combination of activations weighted by alpha
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to focus only on features with positive influence
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_gradcam_on_image(
    original_rgb: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[np.ndarray, np.ndarray]:
    """Resizes Grad-CAM heatmap to original image dimensions and blends with RGB image."""
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam_heatmap, (w, h))

    # Convert normalized heatmap to uint8 [0, 255]
    heatmap_uint8 = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(heatmap_colored_rgb, alpha, original_rgb, 1.0 - alpha, 0)
    return blended, heatmap_colored_rgb
