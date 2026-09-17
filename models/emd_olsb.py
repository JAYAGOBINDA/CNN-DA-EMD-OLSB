"""Dual-image, boundary-safe EMD reversible-data-hiding baseline.

The model name is retained for compatibility with the project's comparison UI.
This implementation uses the EMD component: each safe channel pair carries one
two-bit symbol in the first stego image while equal and opposite changes in the
second image make exact cover recovery possible by averaging. It embeds an
in-band length header, so extraction needs only the two stego images.
"""

import struct
import numpy as np
from typing import Tuple, Dict, Any

from utils.payload_utils import bytes_to_bits, bits_to_bytes


_LENGTH_HEADER_BYTES = 4
_LENGTH_HEADER_DIGITS = _LENGTH_HEADER_BYTES * 4


def bytes_to_base5_digits(data: bytes) -> np.ndarray:
    """Encode bytes as two-bit symbols in the EMD digit range 0..3."""
    bits = bytes_to_bits(data)
    return np.array(
        [(int(bits[i]) << 1) | int(bits[i + 1]) for i in range(0, len(bits), 2)],
        dtype=np.uint8,
    )


def base5_digits_to_bytes(digits: np.ndarray) -> bytes:
    """Decode the two-bit EMD symbols used by this baseline."""
    bits = []
    for digit in digits:
        bits.extend([(int(digit) >> 1) & 1, int(digit) & 1])
    return bits_to_bytes(np.asarray(bits, dtype=np.uint8))


def _safe_pair_indices(flat_cover: np.ndarray) -> np.ndarray:
    """Return pairs where symmetric +/-1 EMD changes are always in range."""
    pair_count = len(flat_cover) // 2
    p1 = flat_cover[:pair_count * 2:2]
    p2 = flat_cover[1:pair_count * 2:2]
    return np.flatnonzero((p1 >= 1) & (p1 <= 254) & (p2 >= 1) & (p2 <= 254))


class EMDOLSBRDH:
    """Self-contained dual-image EMD baseline with exact boundary-safe recovery."""

    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
        if cover_rgb.ndim != 3 or cover_rgb.shape[2] != 3 or cover_rgb.dtype != np.uint8:
            raise ValueError("EMD-OLSB baseline requires an HxWx3 uint8 RGB cover image.")

        payload = struct.pack('!I', len(secret_bytes)) + secret_bytes
        digits = bytes_to_base5_digits(payload)
        flat_cover = cover_rgb.reshape(-1).astype(np.int16)
        safe_pairs = _safe_pair_indices(flat_cover)

        if len(digits) > len(safe_pairs):
            raise ValueError(
                f"EMD-OLSB baseline payload needs {len(digits)} safe pairs, "
                f"but the cover provides {len(safe_pairs)}."
            )

        flat_s1 = flat_cover.copy()
        flat_s2 = flat_cover.copy()
        # Each tuple changes f(p1, p2) = p1 + 2*p2 (mod 5) by its index.
        deltas = ((0, 0), (1, 0), (0, 1), (0, -1), (-1, 0))

        for pair_idx, symbol in zip(safe_pairs[:len(digits)], digits):
            i = int(pair_idx) * 2
            p1, p2 = int(flat_cover[i]), int(flat_cover[i + 1])
            current = (p1 + 2 * p2) % 5
            delta = (int(symbol) - current) % 5
            d1, d2 = deltas[delta]
            flat_s1[i], flat_s1[i + 1] = p1 + d1, p2 + d2
            flat_s2[i], flat_s2[i + 1] = p1 - d1, p2 - d2

        s1 = flat_s1.reshape(cover_rgb.shape).astype(np.uint8)
        s2 = flat_s2.reshape(cover_rgb.shape).astype(np.uint8)
        return (s1, s2), {
            'total_digits_embedded': int(len(digits)),
            'payload_digits_embedded': int(len(digits) - _LENGTH_HEADER_DIGITS),
            'total_bits_embedded': int(len(digits) * 2),
            'bpp': float((len(digits) * 2) / (cover_rgb.shape[0] * cover_rgb.shape[1])),
            'dual_images': True,
            'self_contained_extraction': True,
            'model_name': 'EMD-OLSB RDH',
        }

    def extract(self, stego_dual: Tuple[np.ndarray, np.ndarray], total_digits: int = None) -> Tuple[bytes, np.ndarray]:
        s1_rgb, s2_rgb = stego_dual
        if s1_rgb.shape != s2_rgb.shape or s1_rgb.ndim != 3 or s1_rgb.shape[2] != 3:
            raise ValueError("EMD-OLSB extraction requires two same-shaped RGB stego images.")

        # Embedding uses strictly symmetric integer deltas, so averaging recovers
        # every carrier value before selecting safe EMD positions.
        summed = s1_rgb.astype(np.uint16) + s2_rgb.astype(np.uint16)
        recovered = (summed // 2).astype(np.uint8)
        flat_s1 = s1_rgb.reshape(-1).astype(np.int16)
        safe_pairs = _safe_pair_indices(recovered.reshape(-1))
        if len(safe_pairs) < _LENGTH_HEADER_DIGITS:
            raise ValueError("Stego pair does not have room for the EMD length header.")

        def extract_symbols(count: int) -> np.ndarray:
            symbols = []
            for pair_idx in safe_pairs[:count]:
                i = int(pair_idx) * 2
                symbols.append((int(flat_s1[i]) + 2 * int(flat_s1[i + 1])) % 5)
            return np.asarray(symbols, dtype=np.uint8)

        header = base5_digits_to_bytes(extract_symbols(_LENGTH_HEADER_DIGITS))
        payload_length = struct.unpack('!I', header[:_LENGTH_HEADER_BYTES])[0]
        required_digits = _LENGTH_HEADER_DIGITS + payload_length * 4
        if required_digits > len(safe_pairs):
            raise ValueError("Embedded EMD payload length exceeds the available safe pair capacity.")

        decoded = base5_digits_to_bytes(extract_symbols(required_digits))
        return decoded[_LENGTH_HEADER_BYTES:_LENGTH_HEADER_BYTES + payload_length], recovered
