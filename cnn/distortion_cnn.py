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
from pathlib import Path
from typing import Tuple, Optional

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


_TRAINED_CNN_SINGLETON = None


def load_trained_distortion_cnn(device: Optional['torch.device'] = None) -> 'Optional[DistortionCNN]':
    """
    Loads and caches the trained DistortionCNN model from models/distortion_cnn.pth.
    Raises RuntimeError if weights file is missing or invalid when CNN is required.
    """
    global _TRAINED_CNN_SINGLETON
    if not TORCH_AVAILABLE:
        return None
    if _TRAINED_CNN_SINGLETON is not None:
        if device is not None:
            _TRAINED_CNN_SINGLETON.to(device)
        return _TRAINED_CNN_SINGLETON

    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_dir = Path(__file__).resolve().parent
    candidate_paths = [
        current_dir.parent / "models" / "distortion_cnn.pth",
        current_dir / "distortion_cnn.pth",
        Path("models/distortion_cnn.pth").resolve(),
    ]

    model_path = None
    for p in candidate_paths:
        if p.is_file():
            model_path = p
            break

    if model_path is not None and model_path.exists():
        try:
            model = DistortionCNN()
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            if model_keys != loaded_keys:
                raise RuntimeError(
                    f"DistortionCNN architecture mismatch with {model_path}!\n"
                    f"Missing: {model_keys - loaded_keys}, Unexpected: {loaded_keys - model_keys}"
                )

            model.load_state_dict(state_dict, strict=True)
            model.to(target_device)
            model.eval()
            _TRAINED_CNN_SINGLETON = model
            print("Loaded trained DistortionCNN from models/distortion_cnn.pth")
            return _TRAINED_CNN_SINGLETON
        except Exception as e:
            print(f"Error loading trained DistortionCNN from {model_path}: {e}")
            raise RuntimeError(f"Failed to load trained DistortionCNN from {model_path}: {e}") from e
    return None


def compute_distortion_maps(
    img_rgb: np.ndarray,
    model: Optional['DistortionCNN'] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.6,
    use_cnn: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel distortion sensitivity maps for an RGB image.

    Blends trained CNN distortion sensitivity with analytic (Sobel + variance) map:
        final_distortion = gamma * CNN_distortion + (1 - gamma) * analytic_distortion

    Args:
        img_rgb:  H×W×3 uint8 numpy array (RGB).
        model:    Optional pre-created DistortionCNN instance.
        alpha:    Weight for Sobel gradient in analytic map.
        beta:     Weight for local variance in analytic map.
        gamma:    CNN blend factor (1.0 = pure CNN, 0.0 = pure analytic baseline).
        use_cnn:  Whether CNN inference should be used if gamma > 0.

    Returns:
        (D_r, D_g, D_b): three H×W float32 arrays in [0, 1].
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    requires_cnn = (gamma > 0.0) and use_cnn

    # 1. Compute CNN distortion if required
    D_cnn_r = D_cnn_g = D_cnn_b = None
    if requires_cnn:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not available, but CNN distortion mode is enabled (gamma > 0). "
                "The trained CNN was NOT used. Set gamma=0.0 or use_cnn=False for analytic baseline."
            )

        if model is None:
            model = load_trained_distortion_cnn()

        if model is None:
            raise RuntimeError(
                "Trained DistortionCNN weights not found or could not be loaded from models/distortion_cnn.pth. "
                "The trained CNN was NOT used. Set gamma=0.0 or use_cnn=False for analytic baseline."
            )

        try:
            device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
            img_f = img_rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)  # (1, 3, H, W)

            out_np = out.squeeze(0).detach().cpu().numpy()
            D_cnn_r = _normalize_01(out_np[0])
            D_cnn_g = _normalize_01(out_np[1])
            D_cnn_b = _normalize_01(out_np[2])

        except Exception as e:
            raise RuntimeError(
                f"DistortionCNN inference failed: {e}. The trained CNN was NOT used."
            ) from e

    # 2. Pure CNN (gamma == 1.0)
    if requires_cnn and gamma >= 1.0:
        return D_cnn_r, D_cnn_g, D_cnn_b

    # 3. Compute Analytic component (needed when gamma < 1.0)
    D_ana_r = _analytic_distortion_map(img_rgb[:, :, 0], alpha, beta)
    D_ana_g = _analytic_distortion_map(img_rgb[:, :, 1], alpha, beta)
    D_ana_b = _analytic_distortion_map(img_rgb[:, :, 2], alpha, beta)

    # 4. Pure Analytic baseline (gamma == 0.0 or use_cnn is False)
    if not requires_cnn or gamma <= 0.0:
        return D_ana_r, D_ana_g, D_ana_b

    # 5. Hybrid blend: final_distortion = gamma * CNN + (1 - gamma) * Analytic
    D_r = _normalize_01(gamma * D_cnn_r + (1.0 - gamma) * D_ana_r)
    D_g = _normalize_01(gamma * D_cnn_g + (1.0 - gamma) * D_ana_g)
    D_b = _normalize_01(gamma * D_cnn_b + (1.0 - gamma) * D_ana_b)
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
