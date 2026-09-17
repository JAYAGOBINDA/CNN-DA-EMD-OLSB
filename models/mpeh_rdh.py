"""Portable MPEH-inspired payload-hiding comparison baseline.

The original placeholder relied on Python-only embedding statistics to extract
or recover a saved stego image. This replacement derives its order from the
LSB-invariant cover representation and embeds an in-band payload length. It is
explicitly non-reversible: it is a steganography comparison baseline, not a
claim of a complete prediction-error-histogram RDH reproduction.
"""

import struct
import numpy as np
import cv2
from typing import Tuple, Dict, Any

from utils.payload_utils import bytes_to_bits, bits_to_bytes
from utils.image_utils import rgb_to_gray

_HEADER_BITS = 32


def compute_local_fluctuation(img_gray: np.ndarray) -> np.ndarray:
    img_f = img_gray.astype(np.float32)
    mean = cv2.blur(img_f, (3, 3))
    return cv2.blur(np.abs(img_f - mean), (3, 3))


def _embedding_order(image_rgb: np.ndarray) -> np.ndarray:
    """Return a stable low-fluctuation channel order unaffected by LSB writes."""
    stable_gray = rgb_to_gray(image_rgb & 0xFE)
    pixel_order = np.argsort(compute_local_fluctuation(stable_gray).reshape(-1), kind='stable')
    return np.asarray([pixel * 3 + channel for pixel in pixel_order for channel in range(3)], dtype=np.intp)


class MPEHRDH:
    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        if cover_rgb.ndim != 3 or cover_rgb.shape[2] != 3 or cover_rgb.dtype != np.uint8:
            raise ValueError("MPEH baseline requires an HxWx3 uint8 RGB cover image.")
        payload_bits = bytes_to_bits(secret_bytes)
        bitstream = np.concatenate((bytes_to_bits(struct.pack('!I', len(secret_bytes))), payload_bits))
        order = _embedding_order(cover_rgb)
        if len(bitstream) > len(order):
            raise ValueError(f"MPEH baseline payload needs {len(bitstream)} bits but capacity is {len(order)} bits.")

        flat_stego = cover_rgb.reshape(-1).copy()
        for index, bit in zip(order[:len(bitstream)], bitstream):
            flat_stego[int(index)] = (flat_stego[int(index)] & 0xFE) | int(bit)
        stego = flat_stego.reshape(cover_rgb.shape)
        return stego, {
            'total_bits_embedded': int(len(bitstream)),
            'payload_bits_embedded': int(len(payload_bits)),
            'bpp': float(len(bitstream) / (cover_rgb.shape[0] * cover_rgb.shape[1])),
            'self_contained_extraction': True,
            'reversible': False,
            'model_name': 'MPEH-RDH',
        }

    def extract(self, stego_rgb: np.ndarray, stats: Dict[str, Any] = None) -> Tuple[bytes, None]:
        order = _embedding_order(stego_rgb)
        if len(order) < _HEADER_BITS:
            raise ValueError("Stego image is too small for the MPEH payload header.")
        flat_stego = stego_rgb.reshape(-1)
        header_bits = np.asarray([flat_stego[int(i)] & 1 for i in order[:_HEADER_BITS]], dtype=np.uint8)
        payload_length = struct.unpack('!I', bits_to_bytes(header_bits))[0]
        total_bits = _HEADER_BITS + payload_length * 8
        if total_bits > len(order):
            raise ValueError("MPEH payload length exceeds stego capacity.")
        bits = np.asarray([flat_stego[int(i)] & 1 for i in order[_HEADER_BITS:total_bits]], dtype=np.uint8)
        return bits_to_bytes(bits)[:payload_length], None
