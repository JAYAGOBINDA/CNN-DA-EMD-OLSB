"""
Model 6 (Proposed): CNN-Guided Distortion-Aware Adaptive EMD-OLSB
              (CNN-DA-EMD-OLSB) Model Wrapper

Wraps core dual-stego algorithm (`core/cnn_da_emd_olsb.py`) in a clean class
interface compatible with the project's BaseModelAdapter pattern.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import torch

from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb


class CNNDAEMDOLSBModel:
    """
    CNN-Guided Distortion-Aware Adaptive EMD-OLSB Model (Dual Stego Image).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta:  float = 0.5,
        gamma: float = 0.6,
        t1:    float = 0.33,
        t2:    float = 0.66,
        use_cnn: bool = True
    ):
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.t1      = t1
        self.t2      = t2
        self.use_cnn = use_cnn
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cnn_model = None
        self._cnn_trained = False

        if use_cnn:
            from cnn.distortion_cnn import DistortionCNN

            # Project-relative candidate paths to locate distortion_cnn.pth robustly
            current_dir = Path(__file__).resolve().parent
            candidate_paths = [
                current_dir / "distortion_cnn.pth",
                current_dir.parent / "models" / "distortion_cnn.pth",
                Path("models/distortion_cnn.pth").resolve(),
            ]

            model_path = None
            for p in candidate_paths:
                if p.is_file():
                    model_path = p
                    break

            if model_path is None or not model_path.exists():
                raise FileNotFoundError(
                    f"Trained DistortionCNN weights not found in candidate paths: {candidate_paths}. "
                    "Cannot initialize CNN-DA-EMD-OLSB with use_cnn=True without trained weights. "
                    "Set use_cnn=False for analytic baseline mode."
                )

            try:
                model = DistortionCNN()
                state_dict = torch.load(
                    model_path,
                    map_location="cpu",
                    weights_only=True
                )

                # Verify that architecture strictly matches the saved weights
                model_keys = set(model.state_dict().keys())
                loaded_keys = set(state_dict.keys())
                missing_keys = model_keys - loaded_keys
                unexpected_keys = loaded_keys - model_keys

                if missing_keys or unexpected_keys:
                    raise RuntimeError(
                        f"DistortionCNN architecture mismatch with {model_path}!\n"
                        f"Missing keys: {missing_keys}\n"
                        f"Unexpected keys: {unexpected_keys}"
                    )

                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()

                self._cnn_model = model
                self._cnn_trained = True

                print("Loaded trained DistortionCNN from models/distortion_cnn.pth")

            except Exception as e:
                self._cnn_model = None
                self._cnn_trained = False
                raise RuntimeError(
                    f"Failed to load trained DistortionCNN from {model_path}: {e}. "
                    "Will NOT proceed with random/untrained CNN."
                ) from e
        else:
            self._cnn_model = None
            self._cnn_trained = False

    def embed(
        self,
        cover_rgb: np.ndarray,
        secret_bytes: bytes,
        password: str = "Pass123!",
        payload_type: int = 0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        """
        Embed secret payload into cover RGB image.

        Returns:
            (stego1_rgb, stego2_rgb): Dual stego images.
            stats_dict: Embedding statistics.
        """
        stego_dual, stats = embed_cnn_da_emd_olsb(
            cover_rgb    = cover_rgb,
            secret_data  = secret_bytes,
            password     = password,
            alpha        = self.alpha,
            beta         = self.beta,
            gamma        = self.gamma,
            t1           = self.t1,
            t2           = self.t2,
            payload_type = payload_type,
            model        = self._cnn_model
        )

        stats['model_name'] = 'CNN-DA-EMD-OLSB'
        stats['cnn_enabled'] = (self._cnn_model is not None)
        stats['cnn_trained'] = getattr(self, '_cnn_trained', False)
        return stego_dual, stats

    def extract(
        self,
        stego_dual: Tuple[np.ndarray, np.ndarray],
        password: str = "Pass123!",
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
        """
        Extract secret payload and recover cover image from dual stego images.

        Args:
            stego_dual: Tuple (stego1_rgb, stego2_rgb).

        Returns:
            secret_data, recovered_cover_rgb, metadata_dict
        """
        secret_data, recovered_cover, meta = extract_cnn_da_emd_olsb(
            stego_dual = stego_dual,
            password   = password,
            alpha      = self.alpha,
            beta       = self.beta,
            gamma      = gamma if gamma is not None else self.gamma,
            t1         = t1 if t1 is not None else self.t1,
            t2         = t2 if t2 is not None else self.t2,
            model      = self._cnn_model
        )
        return secret_data, recovered_cover, meta
