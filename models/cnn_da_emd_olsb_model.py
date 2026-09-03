"""
Model 6 (Proposed): CNN-Guided Distortion-Aware Adaptive EMD-OLSB
              (CNN-DA-EMD-OLSB) Model Wrapper

Wraps core dual-stego algorithm (`core/cnn_da_emd_olsb.py`) in a clean class
interface compatible with the project's BaseModelAdapter pattern.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

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
        self._cnn_model = None

        if use_cnn:
            try:
                from cnn.distortion_cnn import DistortionCNN
                self._cnn_model = DistortionCNN()
                self._cnn_model.eval()
            except Exception:
                self._cnn_model = None

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
        return stego_dual, stats

    def extract(
        self,
        stego_dual: Tuple[np.ndarray, np.ndarray],
        password: str = "Pass123!",
        t1: Optional[float] = None,
        t2: Optional[float] = None
    ) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
        """
        Extract secret payload and recover cover image from dual stego images.

        Args:
            stego_dual: Tuple (stego1_rgb, stego2_rgb).

        Returns:
            secret_bytes, recovered_cover_rgb, metadata_dict
        """
        secret_data, recovered_cover, meta = extract_cnn_da_emd_olsb(
            stego_dual = stego_dual,
            password   = password,
            alpha      = self.alpha,
            beta       = self.beta,
            gamma      = self.gamma,
            t1         = t1 if t1 is not None else self.t1,
            t2         = t2 if t2 is not None else self.t2,
            model      = self._cnn_model
        )
        return secret_data, recovered_cover, meta
