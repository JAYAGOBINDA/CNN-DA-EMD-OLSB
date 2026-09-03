"""
Standard Model Interface & Adapter Framework for 6-Model Comparative Benchmark Suite.
Handles differences across input formats (RGB/Grayscale), payload types (Binary/Text/Image),
single vs dual stego outputs, and Reversible Data Hiding (RDH) carrier image recovery.

Slot 6 (Proposed): CNN-DA-EMD-OLSB
    CNN-Guided Distortion-Aware Adaptive EMD-OLSB Framework for Reversible Data Hiding
    in RGB Images — proposed system integrating DistortionCNN guidance with adaptive
    EMD pixel-pair modification and OLSB complementary embedding. Dual-stego output.
"""

from abc import ABC, abstractmethod
import time
import numpy as np
from typing import Tuple, Dict, Any, Union

from models.mpeh_rdh import MPEHRDH
from models.mcsh_rdh import MCSHRDH
from models.cnn_rdh import CNNRDHPredictor
from models.srdnn_stego import SRDNNStego
from models.emd_olsb import EMDOLSBRDH
from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from core.metrics import evaluate_quality, calculate_psnr, calculate_ssim, compute_mse, compute_ber, calculate_wpsnr


class BaseModelAdapter(ABC):
    """Standard Abstract Model Adapter Interface."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[Any, Dict[str, Any]]:
        pass

    @abstractmethod
    def extract(self, stego_output: Any, stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, Any]:
        pass

    @abstractmethod
    def get_compatibility_protocol(self) -> Dict[str, Any]:
        pass


class MPEHAdapter(BaseModelAdapter):
    """Model 1: MPEH-RDH Adapter"""
    def __init__(self):
        super().__init__("MPEH-RDH")
        self.model = MPEHRDH()

    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[np.ndarray, Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes)

    def extract(self, stego_output: np.ndarray, stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, np.ndarray]:
        return self.model.extract(stego_output, stats)

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'MPEH-RDH',
            'Input Type': 'Grayscale / RGB',
            'Payload Type': 'Text / Binary',
            'Output Type': 'Single Stego Photo',
            'Reversible?': 'Yes (Bit-Exact Cover Recovery)',
            'Deep Learning?': 'No',
            'Special Requirements': 'Paper-inspired 8-neighbor fluctuation & histogram shifting'
        }


class MCSHAdapter(BaseModelAdapter):
    """Model 2: MCSH-RDH Adapter"""
    def __init__(self):
        super().__init__("MCSH-RDH")
        self.model = MCSHRDH()

    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[np.ndarray, Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes)

    def extract(self, stego_output: np.ndarray, stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, np.ndarray]:
        return self.model.extract(stego_output, stats)

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'MCSH-RDH',
            'Input Type': 'RGB Color Photo',
            'Payload Type': 'Text / Binary',
            'Output Type': 'Single Stego Photo',
            'Reversible?': 'Yes (Bit-Exact Cover Recovery)',
            'Deep Learning?': 'No',
            'Special Requirements': 'Inter-channel correlation & adaptive R/G/B variance allocation'
        }


class CNNRDHAdapter(BaseModelAdapter):
    """Model 3: CNN-RDH Predictor Adapter"""
    def __init__(self):
        super().__init__("CNN-RDH Predictor")
        self.model = CNNRDHPredictor()

    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[np.ndarray, Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes)

    def extract(self, stego_output: np.ndarray, stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, np.ndarray]:
        return self.model.extract(stego_output, stats)

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'CNN-RDH Predictor',
            'Input Type': 'Grayscale / RGB',
            'Payload Type': 'Text / Binary',
            'Output Type': 'Single Stego Photo',
            'Reversible?': 'Yes (Bit-Exact Cover Recovery)',
            'Deep Learning?': 'Yes (PyTorch CNN Predictor)',
            'Special Requirements': 'Prediction difference G(i,j) = P(i,j) - C(i,j) histogram RDH'
        }


class SRDNNAdapter(BaseModelAdapter):
    """Model 4: SRDNN-Stego Adapter"""
    def __init__(self):
        super().__init__("SRDNN-Stego")
        self.model = SRDNNStego()

    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[np.ndarray, Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes, password=password)

    def extract(self, stego_output: np.ndarray, stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, None]:
        total_bits = stats.get('total_bits_embedded', 512)
        perm_idx = stats.get('perm_idx', np.arange(total_bits))
        extracted_bytes = self.model.extract(stego_output, total_bits, perm_idx, password=password)
        return extracted_bytes, None

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'SRDNN-Stego',
            'Input Type': 'RGB Photo',
            'Payload Type': 'Secret Image / Binary File',
            'Output Type': 'Super-Resolution Stego Photo',
            'Reversible?': 'No (High-Capacity Payload Hiding)',
            'Deep Learning?': 'Yes (SRDNN Reconstruction Network)',
            'Special Requirements': '3D Lorenz Chaotic Map Permutation & ECC Key Security'
        }


class EMDOLSBAdapter(BaseModelAdapter):
    """Model 5: EMD-OLSB RDH Adapter"""
    def __init__(self):
        super().__init__("EMD-OLSB RDH")
        self.model = EMDOLSBRDH()

    def embed(self, cover_rgb: np.ndarray, payload: Union[bytes, str], password: str = "Pass123!") -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes)

    def extract(self, stego_output: Tuple[np.ndarray, np.ndarray], stats: Dict[str, Any], password: str = "Pass123!") -> Tuple[bytes, np.ndarray]:
        total_digits = stats.get('total_digits_embedded', 256)
        return self.model.extract(stego_output, total_digits)

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'EMD-OLSB RDH',
            'Input Type': 'RGB Photo',
            'Payload Type': 'Text / Binary',
            'Output Type': 'Dual Stego Photos (S1, S2)',
            'Reversible?': 'Yes (Bit-Exact Cover Recovery)',
            'Deep Learning?': 'No',
            'Special Requirements': 'Modulo base-5 EMD function f(p1,p2)=(p1+2*p2)mod5 & Dual-Image OLSB'
        }


class CNNDAEMDOLSBAdapter(BaseModelAdapter):
    """
    Model 6 (Proposed): CNN-Guided Distortion-Aware Adaptive EMD-OLSB Adapter.

    Key innovations:
    - DistortionCNN computes per-channel (R, G, B) distortion sensitivity maps
    - Adaptive capacity per pixel driven by distortion class:
      Class 0/1 → EMD mod-5 on R-G channel pairs (≈2.32 bits per pair)
      Class 2   → Adaptive multi-bit OLSB on Blue channel (1-3 bits)
    - Dual Stego Image Output (S1, S2) for exact cover recovery via averaging
    - AES-256-GCM authenticated encryption for payload security
    """
    def __init__(self):
        super().__init__("CNN-DA-EMD-OLSB")
        self.model = CNNDAEMDOLSBModel()

    def embed(
        self,
        cover_rgb: np.ndarray,
        payload: Union[bytes, str],
        password: str = "Pass123!",
        payload_type: int = 0
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        payload_bytes = payload.encode('utf-8') if isinstance(payload, str) else payload
        return self.model.embed(cover_rgb, payload_bytes, password=password, payload_type=payload_type)

    def extract(
        self,
        stego_output: Tuple[np.ndarray, np.ndarray],
        stats: Dict[str, Any],
        password: str = "Pass123!"
    ) -> Tuple[bytes, np.ndarray]:
        secret_bytes, recovered_cover, meta = self.model.extract(
            stego_output,
            password=password,
            t1=stats.get('t1', 0.33),
            t2=stats.get('t2', 0.66),
            gamma=stats.get('gamma', 0.6)
        )
        return secret_bytes, recovered_cover

    def get_compatibility_protocol(self) -> Dict[str, Any]:
        return {
            'Model': 'CNN-DA-EMD-OLSB (Proposed)',
            'Input Type': 'RGB Color Photo',
            'Payload Type': 'Text / Binary / Secret Image',
            'Output Type': 'Dual Stego Photos (S1, S2)',
            'Reversible?': 'Yes (Bit-Exact Cover Recovery via Dual-Image Averaging & AES-256 Authentication)',
            'Deep Learning?': 'Yes (DistortionCNN Per-Channel Map Network)',
            'Special Requirements': 'CNN Distortion-Guided EMD mod-5 + Adaptive OLSB, Dual-Stego Reversibility, AES-256-GCM'
        }
