"""
Model 5: EMD-OLSB RDH (Enhanced Exploiting Modification Direction + Optimal LSB Dual-Image RDH)
Reversible Data Hiding (RDH) Baseline from Literature Paper 5.
"""

import numpy as np
from typing import Tuple, Dict, Any
from utils.payload_utils import bytes_to_bits, bits_to_bytes
from utils.image_utils import rgb_to_gray, gray_to_rgb


def bytes_to_base5_digits(data: bytes) -> np.ndarray:
    bits = bytes_to_bits(data)
    digits = []
    for i in range(0, len(bits) - 1, 2):
        val = (int(bits[i]) << 1) | int(bits[i+1])
        digits.append(val)
    return np.array(digits, dtype=np.uint8)


def base5_digits_to_bytes(digits: np.ndarray) -> bytes:
    bits = []
    for d in digits:
        b0 = (int(d) >> 1) & 1
        b1 = int(d) & 1
        bits.extend([b0, b1])
    return bits_to_bytes(np.array(bits, dtype=np.uint8))


class EMDOLSBRDH:
    """EMD-OLSB Dual-Image RDH Model Implementation."""

    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        h, w, c = cover_rgb.shape
        digits = bytes_to_base5_digits(secret_bytes)
        total_digits = len(digits)

        s1_rgb = cover_rgb.copy()
        s2_rgb = cover_rgb.copy()

        flat_c = cover_rgb.reshape(-1).astype(np.int16)
        flat_s1 = flat_c.copy()
        flat_s2 = flat_c.copy()

        digit_idx = 0
        num_pairs = len(flat_c) // 2

        for i in range(num_pairs):
            if digit_idx >= total_digits:
                break

            p1 = int(flat_c[2*i])
            p2 = int(flat_c[2*i + 1])
            s = int(digits[digit_idx])

            # EMD Extraction Function: f = (p1 * 1 + p2 * 2) mod 5
            f = (p1 * 1 + p2 * 2) % 5
            d = (s - f) % 5

            if d == 0:
                p1_1, p2_1 = p1, p2
                p1_2, p2_2 = p1, p2
            elif d == 1:
                p1_1, p2_1 = p1 + 1, p2
                p1_2, p2_2 = p1 - 1, p2
            elif d == 2:
                p1_1, p2_1 = p1, p2 + 1
                p1_2, p2_2 = p1, p2 - 1
            elif d == 3:
                p1_1, p2_1 = p1, p2 - 1
                p1_2, p2_2 = p1, p2 + 1
            else:  # d == 4
                p1_1, p2_1 = p1 - 1, p2
                p1_2, p2_2 = p1 + 1, p2

            flat_s1[2*i] = np.clip(p1_1, 0, 255)
            flat_s1[2*i + 1] = np.clip(p2_1, 0, 255)

            flat_s2[2*i] = np.clip(p1_2, 0, 255)
            flat_s2[2*i + 1] = np.clip(p2_2, 0, 255)

            digit_idx += 1

        s1_final = flat_s1.reshape(h, w, c).astype(np.uint8)
        s2_final = flat_s2.reshape(h, w, c).astype(np.uint8)

        stats = {
            'total_digits_embedded': digit_idx,
            'total_bits_embedded': digit_idx * 2,
            'bpp': (digit_idx * 2) / (h * w * c),
            'dual_images': True,
            'model_name': 'EMD-OLSB RDH'
        }

        return (s1_final, s2_final), stats

    def extract(self, stego_dual: Tuple[np.ndarray, np.ndarray], total_digits: int) -> Tuple[bytes, np.ndarray]:
        s1_rgb, s2_rgb = stego_dual
        h, w, c = s1_rgb.shape

        flat_s1 = s1_rgb.reshape(-1).astype(np.int16)
        flat_s2 = s2_rgb.reshape(-1).astype(np.int16)
        flat_rec = flat_s1.copy()

        extracted_digits = []
        num_pairs = len(flat_s1) // 2

        for i in range(num_pairs):
            if len(extracted_digits) >= total_digits:
                break

            p1_1, p2_1 = int(flat_s1[2*i]), int(flat_s1[2*i + 1])
            p1_2, p2_2 = int(flat_s2[2*i]), int(flat_s2[2*i + 1])

            # Extract secret digit
            s_digit = (p1_1 * 1 + p2_1 * 2) % 5
            extracted_digits.append(s_digit)

            # Reconstruct original carrier pair via dual averaging
            p1_orig = int(round((p1_1 + p1_2) / 2.0))
            p2_orig = int(round((p2_1 + p2_2) / 2.0))

            flat_rec[2*i] = p1_orig
            flat_rec[2*i + 1] = p2_orig

        extracted_bytes = base5_digits_to_bytes(np.array(extracted_digits[:total_digits], dtype=np.uint8))
        recovered_rgb = flat_rec.reshape(h, w, c).astype(np.uint8)

        return extracted_bytes, recovered_rgb
