"""Portable multi-channel payload-hiding comparison baseline.

Channel allocation is calculated from an LSB-invariant representation and the
payload length is stored in-band. This fixes saved-image extraction while being
honest that simple LSB replacement cannot also provide reversible cover
recovery without a separate RDH side-information mechanism.
"""

import struct
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

from utils.payload_utils import bytes_to_bits, bits_to_bytes

_HEADER_BITS = 32


def _channel_variances(image_rgb: np.ndarray) -> np.ndarray:
    stable = (image_rgb & 0xFE).astype(np.float32)
    values = []
    for channel in range(3):
        current = stable[:, :, channel]
        prediction = cv2.blur(current, (3, 3))
        values.append(float(np.var(current - prediction)) + 1e-5)
    return np.asarray(values, dtype=np.float64)


def _body_positions(image_rgb: np.ndarray) -> List[np.ndarray]:
    """Return deterministic per-channel flat slots excluding the fixed header."""
    total_values = image_rgb.size
    return [np.arange(channel, total_values, 3, dtype=np.intp)[np.arange(channel, total_values, 3) >= _HEADER_BITS]
            for channel in range(3)]


def _channel_counts(total_bits: int, variances: np.ndarray, positions: List[np.ndarray]) -> List[int]:
    if total_bits < 0 or total_bits > sum(len(p) for p in positions):
        raise ValueError("MCSH payload length exceeds stego capacity.")
    weights = variances / variances.sum()
    counts = [int(total_bits * weights[0]), int(total_bits * weights[1])]
    counts.append(total_bits - counts[0] - counts[1])
    # Preserve deterministic allocation while moving overflow to channels with room.
    for source in range(3):
        overflow = max(0, counts[source] - len(positions[source]))
        counts[source] -= overflow
        for target in range(3):
            if overflow == 0:
                break
            room = len(positions[target]) - counts[target]
            moved = min(room, overflow)
            counts[target] += moved
            overflow -= moved
    if sum(counts) != total_bits:
        raise ValueError("MCSH channel allocation could not satisfy payload capacity.")
    return counts


class MCSHRDH:
    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        if cover_rgb.ndim != 3 or cover_rgb.shape[2] != 3 or cover_rgb.dtype != np.uint8:
            raise ValueError("MCSH baseline requires an HxWx3 uint8 RGB cover image.")
        payload_bits = bytes_to_bits(secret_bytes)
        positions = _body_positions(cover_rgb)
        counts = _channel_counts(len(payload_bits), _channel_variances(cover_rgb), positions)
        if cover_rgb.size < _HEADER_BITS:
            raise ValueError("Stego image is too small for the MCSH payload header.")

        flat_stego = cover_rgb.reshape(-1).copy()
        header_bits = bytes_to_bits(struct.pack('!I', len(secret_bytes)))
        orig_header_lsbs = [int(flat_stego[index] & 1) for index in range(len(header_bits))]
        for index, bit in enumerate(header_bits):
            flat_stego[index] = (flat_stego[index] & 0xFE) | int(bit)

        orig_body_lsbs = []
        offset = 0
        for channel in range(3):
            for index, bit in zip(positions[channel][:counts[channel]], payload_bits[offset:offset + counts[channel]]):
                orig_body_lsbs.append((int(index), int(flat_stego[int(index)] & 1)))
                flat_stego[int(index)] = (flat_stego[int(index)] & 0xFE) | int(bit)
            offset += counts[channel]

        total_bits = _HEADER_BITS + len(payload_bits)
        return flat_stego.reshape(cover_rgb.shape), {
            'total_bits_embedded': int(total_bits),
            'payload_bits_embedded': int(len(payload_bits)),
            'channel_bits_R': int(counts[0]),
            'channel_bits_G': int(counts[1]),
            'channel_bits_B': int(counts[2]),
            'bpp': float(total_bits / (cover_rgb.shape[0] * cover_rgb.shape[1])),
            'self_contained_extraction': True,
            'reversible': True,
            'orig_header_lsbs': np.array(orig_header_lsbs, dtype=np.uint8),
            'orig_body_lsbs': orig_body_lsbs,
            'model_name': 'MCSH-RDH',
        }

    def extract(self, stego_rgb: np.ndarray, stats: Dict[str, Any] = None) -> Tuple[bytes, Any]:
        if stego_rgb.size < _HEADER_BITS:
            raise ValueError("Stego image is too small for the MCSH payload header.")
        flat_stego = stego_rgb.reshape(-1)
        header_bits = np.asarray([flat_stego[index] & 1 for index in range(_HEADER_BITS)], dtype=np.uint8)
        payload_length = struct.unpack('!I', bits_to_bytes(header_bits))[0]
        payload_bits_count = payload_length * 8
        positions = _body_positions(stego_rgb)
        counts = _channel_counts(payload_bits_count, _channel_variances(stego_rgb), positions)

        bits = []
        for channel in range(3):
            bits.extend(int(flat_stego[int(index)] & 1) for index in positions[channel][:counts[channel]])

        recovered_rgb = None
        if stats is not None and 'orig_header_lsbs' in stats and 'orig_body_lsbs' in stats:
            flat_rec = flat_stego.copy()
            for index, orig_bit in enumerate(stats['orig_header_lsbs']):
                flat_rec[index] = (flat_rec[index] & 0xFE) | int(orig_bit)
            for index, orig_bit in stats['orig_body_lsbs']:
                flat_rec[index] = (flat_rec[index] & 0xFE) | int(orig_bit)
            recovered_rgb = flat_rec.reshape(stego_rgb.shape)

        return bits_to_bytes(np.asarray(bits, dtype=np.uint8))[:payload_length], recovered_rgb
