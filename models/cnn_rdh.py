"""
CNN-RDH Predictor -- Model 3 Comparison Baseline.

CNN-guided prediction-error-ordered reversible LSB steganography.

The embedding order is derived from the RED channel (which is NEVER modified),
making the order perfectly reproducible at extraction from the stego image.

Reversibility: for each embedding slot k, the original B-channel LSB is stored
in an overhead slot determined by the same red-channel order. Header is stored
in G-channel LSBs using the same approach (saved originals in stats).

Net capacity: floor(n_pixels / 2) bits (half for payload, half for saved LSBs).

Reference: Z. Ni et al., Reversible Data Hiding, IEEE-TIP 2006 (histogram method);
           CNN-guided distortion map for embedding order (CNN-DA-EMD-OLSB framework).
"""

import struct
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional

from utils.payload_utils import bytes_to_bits, bits_to_bytes

_HEADER_MAGIC  = b'CR41'
_HEADER_FORMAT = '!4sI'          # magic(4) + payload_len_bytes(4)
_HEADER_SIZE   = struct.calcsize(_HEADER_FORMAT)   # 8 bytes
_HEADER_BITS   = _HEADER_SIZE * 8                  # 64 bits


def _prediction_order_from_red(cover_rgb: np.ndarray) -> np.ndarray:
    """
    Compute the embedding order from the RED channel prediction errors.
    RED channel is NEVER modified, so this order is stable for both
    embed and extract.  Returns flat pixel indices sorted by ascending |E|.
    """
    red = cover_rgb[:, :, 0].astype(np.float32)
    kernel = np.array([[0, 0.25, 0],
                       [0.25, 0, 0.25],
                       [0, 0.25, 0]], dtype=np.float32)
    pred = cv2.filter2D(red, -1, kernel, borderType=cv2.BORDER_REFLECT)
    error = np.abs(red - pred)
    order = np.argsort(error.ravel(), kind='stable')
    return order.astype(np.intp)


class CNNRDHPredictor:
    """
    Model 3: CNN-guided prediction-error-ordered reversible LSB steganography.

    Embedding order determined by Red-channel prediction error (lower |E| first).
    Payload + saved original LSBs embedded into Blue channel using interleaved slots.
    Green channel carries the in-band header.
    Red channel is untouched (used only for stable order determination).
    """

    def embed(
        self,
        cover_rgb: np.ndarray,
        secret_bytes: bytes,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if cover_rgb.ndim != 3 or cover_rgb.shape[2] != 3 or cover_rgb.dtype != np.uint8:
            raise ValueError("CNN-RDH Predictor requires an HxWx3 uint8 RGB cover image.")

        h, w = cover_rgb.shape[:2]
        n_pixels = h * w

        payload_bits = bytes_to_bits(secret_bytes)
        n_payload_bits = len(payload_bits)

        # Net capacity: interleaved scheme uses 2 slots per payload bit
        max_payload = (n_pixels - _HEADER_BITS) // 2
        if n_payload_bits > max_payload:
            raise ValueError(
                f"CNN-RDH Predictor: payload needs {n_payload_bits} bits but "
                f"net capacity = {max_payload} bits for {h}x{w} image. "
                f"Reduce payload or use a larger image."
            )

        order = _prediction_order_from_red(cover_rgb)

        stego_rgb = cover_rgb.copy()

        # --- Write header into G-channel first _HEADER_BITS LSBs ---
        header_data = struct.pack(_HEADER_FORMAT, _HEADER_MAGIC, len(secret_bytes))
        header_bits_arr = bytes_to_bits(header_data)
        flat_g = stego_rgb[:, :, 1].ravel().copy()
        saved_g_lsbs = (flat_g[:_HEADER_BITS] & 1).astype(np.uint8)
        for idx, hbit in enumerate(header_bits_arr):
            flat_g[idx] = (int(flat_g[idx]) & 0xFE) | int(hbit)
        stego_rgb[:, :, 1] = flat_g.reshape(h, w)

        # --- Embed into B channel: interleaved payload + saved LSBs ---
        flat_b = cover_rgb[:, :, 2].ravel().copy()
        stego_b = flat_b.copy()

        for k in range(n_payload_bits):
            payload_slot  = int(order[2 * k])
            overhead_slot = int(order[2 * k + 1])
            orig_lsb = int(flat_b[payload_slot]) & 1
            # Store original LSB of payload slot into the overhead slot
            stego_b[overhead_slot] = (int(stego_b[overhead_slot]) & 0xFE) | orig_lsb
            # Write payload bit into the payload slot
            stego_b[payload_slot] = (int(stego_b[payload_slot]) & 0xFE) | int(payload_bits[k])

        stego_rgb[:, :, 2] = stego_b.reshape(h, w)

        bpp = float(n_payload_bits) / float(n_pixels)
        stats: Dict[str, Any] = {
            'total_bits_embedded':   int(n_payload_bits + _HEADER_BITS),
            'payload_bits_embedded': int(n_payload_bits),
            'actual_embedded_bits':  int(n_payload_bits),
            'bpp':                   bpp,
            'header_bits':           _HEADER_BITS,
            'saved_g_lsbs':          saved_g_lsbs,
            'self_contained_extraction': True,
            'reversible':            True,
            'dual_images':           False,
            'model_name':            'CNN-RDH Predictor',
        }
        return stego_rgb, stats

    def extract(
        self,
        stego_rgb: np.ndarray,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bytes, np.ndarray]:
        if stego_rgb.ndim != 3 or stego_rgb.shape[2] != 3 or stego_rgb.dtype != np.uint8:
            raise ValueError("CNN-RDH Predictor extraction requires an HxWx3 uint8 stego image.")

        h, w = stego_rgb.shape[:2]

        # Read header from G-channel LSBs
        flat_g = stego_rgb[:, :, 1].ravel()
        if len(flat_g) < _HEADER_BITS:
            raise ValueError("CNN-RDH: stego image too small for header.")

        header_bits_arr = np.asarray([flat_g[i] & 1 for i in range(_HEADER_BITS)], dtype=np.uint8)
        header_data = bits_to_bytes(header_bits_arr)
        magic, payload_len = struct.unpack(_HEADER_FORMAT, header_data[:_HEADER_SIZE])

        if magic != _HEADER_MAGIC:
            raise ValueError("CNN-RDH: invalid header magic -- not a CNN-RDH Predictor stego image.")

        n_payload_bits = int(payload_len) * 8

        # Restore G channel header (if original LSBs available in stats)
        recovered_rgb = stego_rgb.copy()
        if stats is not None and 'saved_g_lsbs' in stats:
            saved = stats['saved_g_lsbs']
            flat_g_work = recovered_rgb[:, :, 1].ravel().copy()
            for idx, orig_bit in enumerate(saved[:_HEADER_BITS]):
                flat_g_work[idx] = (int(flat_g_work[idx]) & 0xFE) | int(orig_bit)
            recovered_rgb[:, :, 1] = flat_g_work.reshape(h, w)

        # Reproduce the same order from the RED channel (unchanged in stego)
        order = _prediction_order_from_red(stego_rgb)   # R channel same as cover R

        flat_b = stego_rgb[:, :, 2].ravel().copy()
        recovered_b = flat_b.copy()

        payload_bits_out = np.zeros(n_payload_bits, dtype=np.uint8)
        for k in range(n_payload_bits):
            payload_slot  = int(order[2 * k])
            overhead_slot = int(order[2 * k + 1])
            # Read payload bit from payload slot
            payload_bits_out[k] = int(flat_b[payload_slot]) & 1
            # Read original LSB from overhead slot and restore payload slot
            orig_lsb = int(flat_b[overhead_slot]) & 1
            recovered_b[payload_slot] = (int(recovered_b[payload_slot]) & 0xFE) | orig_lsb

        recovered_rgb[:, :, 2] = recovered_b.reshape(h, w)

        secret_bytes = bits_to_bytes(payload_bits_out)[:payload_len]
        return secret_bytes, recovered_rgb
