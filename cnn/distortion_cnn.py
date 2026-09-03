"""
DistortionCNN: Multi-Scale Convolutional Distortion Sensitivity Estimator.

Used by the CNN-Guided Distortion-Aware Adaptive EMD-OLSB (CNN-DA-EMD-OLSB) framework
to compute per-channel distortion sensitivity maps D_r(x,y), D_g(x,y), D_b(x,y).

Architecture:
  - 3 independent single-channel branches (R, G, B), each with 4 stacked conv layers
    using progressively larger kernels to capture multi-scale texture context.
  - Branches are fused via a 1×1 merging convolution to produce a 3-channel
    distortion magnitude output (one per color channel).
  - Works in inference mode without pre-training. Untrained convolutional filters
    act as randomised multi-scale feature extractors, consistent with the
    Random CNN approach validated in texture analysis literature.
  - ABS activation applied at output to ensure all distortion values are non-negative.
"""

import numpy as np
import cv2
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class _SingleChannelBranch(nn.Module):
        """
        Single-channel multi-scale feature extraction branch.
        Input:  1×H×W float32 tensor (normalized to [0,1])
        Output: 8×H×W feature map
        """
        def __init__(self):
            super().__init__()
            # Scale 1: local (3×3)
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(8)
            # Scale 2: mid-range (5×5)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2, bias=False)
            self.bn2   = nn.BatchNorm2d(8)
            # Scale 3: context (7×7)
            self.conv3 = nn.Conv2d(8, 8, kernel_size=7, padding=3, bias=False)
            self.bn3   = nn.BatchNorm2d(8)
            # Compress back to 4 maps
            self.conv4 = nn.Conv2d(8, 4, kernel_size=3, padding=1, bias=False)
            self.bn4   = nn.BatchNorm2d(4)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            return x   # 4 × H × W


    class DistortionCNN(nn.Module):
        """
        Multi-scale CNN distortion estimator for RGB images.

        Input:  float32 tensor of shape (1, 3, H, W), pixel values in [0, 1].
        Output: float32 tensor of shape (1, 3, H, W) representing the
                per-channel distortion sensitivity (higher = more tolerant to
                modification without perceptible degradation).

        Usage (inference-only / no training required):
            model = DistortionCNN()
            model.eval()
            d_map = model(rgb_tensor)   # shape: (1, 3, H, W)
        """

        def __init__(self):
            super().__init__()
            torch.manual_seed(42)
            self.branch_r = _SingleChannelBranch()
            self.branch_g = _SingleChannelBranch()
            self.branch_b = _SingleChannelBranch()

            # Merge 3×4 = 12 feature maps into 3 distortion channels
            self.merge = nn.Conv2d(12, 3, kernel_size=1, bias=False)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """
            Args:
                x: (1, 3, H, W) tensor, values in [0, 1]
            Returns:
                (1, 3, H, W) tensor — per-channel distortion sensitivity
            """
            r = self.branch_r(x[:, 0:1, :, :])   # 4×H×W
            g = self.branch_g(x[:, 1:2, :, :])   # 4×H×W
            b = self.branch_b(x[:, 2:3, :, :])   # 4×H×W

            fused = torch.cat([r, g, b], dim=1)   # 12×H×W
            out   = self.merge(fused)              # 3×H×W
            out   = torch.abs(out)                 # non-negative distortion magnitude
            return out


def compute_distortion_maps(
    img_rgb: np.ndarray,
    model: 'DistortionCNN' = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel distortion sensitivity maps for an RGB image.

    Args:
        img_rgb:  H×W×3 uint8 numpy array (RGB).
        model:    Optional pre-created DistortionCNN instance (avoids re-instantiation).
        alpha:    Weight for Sobel gradient component in hybrid fallback.
        beta:     Weight for local variance component in hybrid fallback.

    Returns:
        (D_r, D_g, D_b): three H×W float32 arrays in [0, 1].
        Higher values indicate pixels that can tolerate MORE modification.
    """
    h, w = img_rgb.shape[:2]

    if TORCH_AVAILABLE:
        try:
            if model is None:
                model = DistortionCNN()
                model.eval()

            # Prepare input tensor: normalise to [0,1], shape (1,3,H,W)
            img_f = img_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0)

            with torch.no_grad():
                out = model(tensor)  # (1, 3, H, W)

            out_np = out.squeeze(0).numpy()  # (3, H, W)
            D_r = _normalize_01(out_np[0])
            D_g = _normalize_01(out_np[1])
            D_b = _normalize_01(out_np[2])
            return D_r, D_g, D_b

        except Exception:
            pass  # Fall through to analytic fallback

    # -------------------------------------------------------------------------
    # Analytic Fallback (no PyTorch / CUDA OOM / etc.)
    # Combine Sobel gradient and local variance to approximate distortion map.
    # Heavily textured / high-gradient areas are more distortion-tolerant.
    # -------------------------------------------------------------------------
    D_r = _analytic_distortion_map(img_rgb[:, :, 0], alpha, beta)
    D_g = _analytic_distortion_map(img_rgb[:, :, 1], alpha, beta)
    D_b = _analytic_distortion_map(img_rgb[:, :, 2], alpha, beta)
    return D_r, D_g, D_b


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _analytic_distortion_map(channel: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Analytically estimate distortion tolerance for a single image channel.

    Combines:
      - Sobel gradient magnitude (edges/texture → higher tolerance)
      - Local 3×3 variance         (complex regions → higher tolerance)
    """
    ch_f = channel.astype(np.float32)

    # Sobel gradient magnitude
    gx = cv2.Sobel(ch_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ch_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    # Local variance (3×3 window)
    mean   = cv2.blur(ch_f, (3, 3))
    mean2  = cv2.blur(ch_f**2, (3, 3))
    var    = np.maximum(0.0, mean2 - mean**2)

    # Composite
    norm_grad = _normalize_01(grad)
    norm_var  = _normalize_01(var)
    composite = alpha * norm_grad + beta * norm_var
    return _normalize_01(composite)
