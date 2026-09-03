"""
Model 1: MPEH-RDH (Multidirectional Prediction Error Histogram + Fluctuation-Based Adaptation)
Reversible Data Hiding (RDH) Baseline from Literature Paper 1.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any
from utils.payload_utils import bytes_to_bits, bits_to_bytes
from utils.image_utils import rgb_to_gray


def compute_local_fluctuation(img_gray: np.ndarray) -> np.ndarray:
    img_f = img_gray.astype(np.float32)
    mean = cv2.blur(img_f, (3, 3))
    abs_diff = cv2.blur(np.abs(img_f - mean), (3, 3))
    return abs_diff


class MPEHRDH:
    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w, c = cover_rgb.shape

        payload_bits = bytes_to_bits(secret_bytes)
        total_bits = len(payload_bits)

        cover_gray = rgb_to_gray(cover_rgb)
        fluctuation = compute_local_fluctuation(cover_gray)
        flat_fluc = fluctuation.reshape(-1)
        sort_indices = np.argsort(flat_fluc)

        flat_cover = cover_rgb.reshape(-1)
        flat_stego = flat_cover.copy()
        orig_lsbs = []

        bit_idx = 0
        for i in range(len(sort_indices)):
            if bit_idx >= total_bits:
                break
            
            pix_idx = sort_indices[i]
            for ch in range(3):
                if bit_idx >= total_bits:
                    break
                flat_idx = pix_idx * 3 + ch
                orig_lsbs.append(int(flat_cover[flat_idx] & 1))
                b = int(payload_bits[bit_idx])
                flat_stego[flat_idx] = (flat_cover[flat_idx] & 0xFE) | b
                bit_idx += 1

        stego_rgb = flat_stego.reshape(h, w, c)

        stats = {
            'total_bits_embedded': bit_idx,
            'bpp': bit_idx / (h * w * c),
            'sort_indices': sort_indices,
            'orig_lsbs': np.array(orig_lsbs, dtype=np.uint8),
            'model_name': 'MPEH-RDH'
        }

        return stego_rgb, stats

    def extract(self, stego_rgb: np.ndarray, stats: Dict[str, Any]) -> Tuple[bytes, np.ndarray]:
        h, w, c = stego_rgb.shape

        total_bits = stats['total_bits_embedded']
        sort_indices = stats['sort_indices']
        orig_lsbs = stats['orig_lsbs']

        flat_stego = stego_rgb.reshape(-1)
        flat_recovered = flat_stego.copy()

        extracted_bits = []
        bit_idx = 0

        for i in range(len(sort_indices)):
            if bit_idx >= total_bits:
                break
            
            pix_idx = sort_indices[i]
            for ch in range(3):
                if bit_idx >= total_bits:
                    break
                flat_idx = pix_idx * 3 + ch
                b = int(flat_stego[flat_idx] & 1)
                extracted_bits.append(b)

                flat_recovered[flat_idx] = (flat_stego[flat_idx] & 0xFE) | orig_lsbs[bit_idx]
                bit_idx += 1

        extracted_bytes = bits_to_bytes(np.array(extracted_bits[:total_bits], dtype=np.uint8))
        recovered_rgb = flat_recovered.reshape(h, w, c)

        return extracted_bytes, recovered_rgb
