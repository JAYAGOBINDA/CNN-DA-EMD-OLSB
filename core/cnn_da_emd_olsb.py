"""
CNN-Guided Distortion-Aware Adaptive EMD-OLSB (CNN-DA-EMD-OLSB)
Core Algorithm Engine — Single-Stego Reversible Data Hiding

Design Notes
============
1. Single Stego Image Output:
   Produces ONE stego image. Cover is recovered exactly via a compressed
   Location Map that stores the original lower bits of every modified pixel.
   This guarantees np.array_equal(cover, recovered_cover) == True.

2. Distortion-Aware Adaptive Routing:
   Per-channel distortion maps D_r, D_g, D_b are computed from the upper
   bitplanes (& 0xF8). Pixels classified into:
     Class 0 (smooth):   EMD embedding via R-G pixel pairs
     Class 1 (moderate): EMD embedding via R-G pixel pairs
     Class 2 (textured): Adaptive k-bit OLSB on Blue channel

3. EMD (Exploiting Modification Direction):
   Uses mod-5 extraction function f(p1,p2) = (p1*1 + p2*2) mod 5 on R-G
   channel pairs. Single minimal-distortion candidate is chosen.
   Secret digits are base-5 encoded.

4. OLSB (Optimal LSB Substitution):
   For class 2 (high-texture) pixels, k-bit LSB replacement is applied
   to the Blue channel.

5. Location Map for Reversible Recovery:
   Before embedding, the original lower 3 bits of every pixel that will
   be modified are collected, compressed with zlib, and embedded as part
   of the payload. During extraction, this map is decompressed and used
   to restore every modified pixel to its exact original value.

6. Deterministic Upper-Bitplane Capacity Recovery:
   Because only lower LSBs (bits 0..2) are modified during embedding while
   upper bits (& 0xF8) remain completely untouched, the capacity maps
   computed during extraction match the embedding maps exactly.

7. AES-256-GCM Authenticated Encryption:
   Payload is compressed, encrypted via AES-256-GCM, and packed with a
   64-byte deterministic header before embedding.
"""

import numpy as np
import zlib
import struct
from typing import Tuple, Dict, Any, Optional

from core.payload import (
    prepare_payload, parse_payload,
    bytes_to_bits, bits_to_bytes,
    HEADER_SIZE_BYTES, HEADER_MAGIC
)
from cnn.distortion_cnn import compute_distortion_maps


# ===========================================================================
# CONSTANTS
# ===========================================================================

CAP_CLASS0 = 0   # Smooth regions   — EMD embedding
CAP_CLASS1 = 1   # Moderate texture — EMD embedding
CAP_CLASS2 = 2   # High texture     — OLSB embedding


# ===========================================================================
# BASE-5 DIGIT CONVERSION (mirrors models/emd_olsb.py)
# ===========================================================================

def bytes_to_base5_digits(data: bytes) -> np.ndarray:
    """Convert bytes to base-5 digit array (2 bits → 1 digit, values 0-3)."""
    bits = bytes_to_bits(data)
    digits = []
    for i in range(0, len(bits) - 1, 2):
        val = (int(bits[i]) << 1) | int(bits[i + 1])
        digits.append(val)
    return np.array(digits, dtype=np.uint8)


def base5_digits_to_bytes(digits: np.ndarray) -> bytes:
    """Convert base-5 digit array back to bytes (1 digit → 2 bits)."""
    bits = []
    for d in digits:
        b0 = (int(d) >> 1) & 1
        b1 = int(d) & 1
        bits.extend([b0, b1])
    return bits_to_bytes(np.array(bits, dtype=np.uint8))


# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def _normalize_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _get_cap_maps(
    img_rgb: np.ndarray,
    alpha: float, beta: float, gamma: float,
    t1: float, t2: float,
    model=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel distortion class maps (H×W uint8, values ∈ {0, 1, 2})
    using CNN distortion guidance blended with analytic Sobel+Var map:
        final_distortion = gamma * CNN_distortion + (1 - gamma) * analytic_distortion

    Class 0: D < t1  (smooth — EMD)
    Class 1: t1 ≤ D < t2 (moderate — EMD)
    Class 2: D ≥ t2  (textured — OLSB)
    """
    D_r_raw, D_g_raw, D_b_raw = compute_distortion_maps(
        img_rgb, model=model, alpha=alpha, beta=beta, gamma=gamma
    )

    def _to_class(d_raw: np.ndarray) -> np.ndarray:
        d = _normalize_01(d_raw)
        cls = np.zeros(d.shape, dtype=np.uint8)  # class 0 by default
        cls[d >= t1] = CAP_CLASS1
        cls[d >= t2] = CAP_CLASS2
        return cls

    return _to_class(D_r_raw), _to_class(D_g_raw), _to_class(D_b_raw)


def _is_valid_header(header_bytes: bytes, h: int, w: int, c: int) -> bool:
    """Check if header bytes unpack to a valid deterministic header."""
    if len(header_bytes) < HEADER_SIZE_BYTES or header_bytes[:4] != HEADER_MAGIC:
        return False
    try:
        magic, _, _, _, cipher_len, _, _, t1, t2, gamma, _, locmap_size, _ = struct.unpack(
            '!4sBB2sI16s12sfffII4s', header_bytes[:HEADER_SIZE_BYTES]
        )
        return (
            magic == HEADER_MAGIC and
            0.0 <= t1 <= 1.0 and
            0.0 <= t2 <= 1.0 and
            0.0 <= gamma <= 1.0 and
            0 < cipher_len <= (h * w * c) and
            0 <= locmap_size <= (h * w * c)
        )
    except Exception:
        return False


def _parse_header(header_bytes: bytes, h: int, w: int, c: int) -> Tuple[int, float, float, float, int]:
    """Parse embedded header; return (cipher_len, t1, t2, gamma, locmap_size)."""
    try:
        magic, _, _, _, cipher_len, _, _, t1, t2, gamma, _, locmap_size, _ = struct.unpack(
            '!4sBB2sI16s12sfffII4s', header_bytes
        )
        if magic == HEADER_MAGIC and 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0 and 0.0 <= gamma <= 1.0:
            return int(cipher_len), float(t1), float(t2), float(gamma), int(locmap_size)
    except Exception:
        pass
    try:
        magic, _, _, _, cipher_len, _, _, t1, t2, gamma, _, _ = struct.unpack(
            '!4sBB2sI16s12sfffI8s', header_bytes
        )
        if magic == HEADER_MAGIC and 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0 and 0.0 <= gamma <= 1.0:
            return int(cipher_len), float(t1), float(t2), float(gamma), 0
    except Exception:
        pass
    try:
        magic, _, _, _, cipher_len, _, _, t1, t2, _, _ = struct.unpack(
            '!4sBB2sI16s12sffI12s', header_bytes
        )
        if magic == HEADER_MAGIC and 0.0 <= t1 <= 1.0 and 0.0 <= t2 <= 1.0:
            return int(cipher_len), float(t1), float(t2), 0.6, 0
    except Exception:
        pass
    return max(64, (h * w * c) // 32), 0.33, 0.66, 0.6, 0



# ===========================================================================
# CAPACITY COMPUTATION
# ===========================================================================

def compute_capacity(
    cls_r: np.ndarray,
    cls_g: np.ndarray,
    cls_b: np.ndarray,
    upper_rgb: np.ndarray
) -> Dict[str, Any]:
    """
    Compute Usable Capacity vs Theoretical Upper Bound.

    - Usable Capacity: accounts for exact EMD routing (R-G pairs where both are
      class 0/1) AND pixel boundary safety constraints (8 <= upper <= 240) preventing
      overflow/underflow clipping during embedding/recovery, plus Blue channel Class 2
      carrying 3 bits OLSB.
    - Theoretical Capacity: theoretical upper bound without boundary clipping constraints.

    Returns dict containing exact counts, masks, and capacity metrics in bits and bytes.
    """
    # EMD routing: pixel positions where BOTH R and G are class 0/1 with boundary safety
    emd_mask = (
        (cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1) &
        (upper_rgb[:, :, 0] >= 8) & (upper_rgb[:, :, 0] <= 240) &
        (upper_rgb[:, :, 1] >= 8) & (upper_rgb[:, :, 1] <= 240)
    )  # (H, W) bool
    usable_emd_pairs = int(np.sum(emd_mask))
    usable_emd_bits = usable_emd_pairs * 2  # each base-5 digit carries 2 bits

    # OLSB routing: Blue channel where class == 2 (carries 3 bits OLSB)
    olsb_mask = (cls_b == CAP_CLASS2)  # (H, W) bool
    usable_olsb_pixels = int(np.sum(olsb_mask))
    usable_olsb_bits = usable_olsb_pixels * 3

    usable_capacity_bits = usable_emd_bits + usable_olsb_bits

    # Theoretical capacity without boundary clipping
    theo_emd_mask = ((cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1))
    theo_emd_pairs = int(np.sum(theo_emd_mask))
    theoretical_capacity_bits = (theo_emd_pairs * 2) + usable_olsb_bits

    return {
        'usable_capacity_bits': usable_capacity_bits,
        'usable_capacity_bytes': usable_capacity_bits // 8,
        'usable_emd_pairs': usable_emd_pairs,
        'usable_emd_bits': usable_emd_bits,
        'emd_capacity_bits': usable_emd_bits,
        'usable_olsb_pixels': usable_olsb_pixels,
        'usable_olsb_bits': usable_olsb_bits,
        'olsb_capacity_bits': usable_olsb_bits,
        'theoretical_capacity_bits': theoretical_capacity_bits,
        'theoretical_capacity_bytes': theoretical_capacity_bits // 8,
        'theoretical_emd_bits': theo_emd_pairs * 2,
        'emd_mask': emd_mask,
        'olsb_mask': olsb_mask,
    }


def _compute_emd_olsb_capacity(
    cls_r: np.ndarray,
    cls_g: np.ndarray,
    cls_b: np.ndarray,
    upper_rgb: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """Legacy helper returning (total_digits_capacity, total_olsb_bits_capacity)."""
    if upper_rgb is not None:
        cap = compute_capacity(cls_r, cls_g, cls_b, upper_rgb)
        return cap['usable_emd_pairs'], cap['usable_olsb_bits']
    # Fallback if upper_rgb not provided
    emd_mask = ((cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1))
    olsb_mask = (cls_b == CAP_CLASS2)
    return int(np.sum(emd_mask)), int(np.sum(olsb_mask)) * 3


# ===========================================================================
# LOCATION MAP BUILDER
# ===========================================================================

def _build_location_map(
    cover_rgb: np.ndarray,
    emd_positions: np.ndarray,
    n_emd_used: int,
    olsb_positions: np.ndarray,
    n_olsb_used: int,
    k_olsb: int = 3
) -> bytes:
    """
    Build a compressed location map storing the original lower bits of every
    pixel that will be modified during embedding.

    For EMD positions: store lower 3 bits of R and G (6 bits per position).
    For OLSB positions: store lower k_olsb bits of B (3 bits per position).

    The map is structured as:
      [4 bytes: n_emd_used (uint32)] [4 bytes: n_olsb_used (uint32)]
      [EMD original bits: n_emd_used * 6 bits, packed]
      [OLSB original bits: n_olsb_used * 3 bits, packed]

    Returns zlib-compressed bytes.
    """
    # Collect all original lower bits
    raw_bits = []

    # EMD positions: R lower 3 bits + G lower 3 bits = 6 bits per position
    for i in range(n_emd_used):
        y, x = emd_positions[i]
        r_val = int(cover_rgb[y, x, 0])
        g_val = int(cover_rgb[y, x, 1])
        for bit_pos in range(2, -1, -1):  # bits 2,1,0 of R
            raw_bits.append((r_val >> bit_pos) & 1)
        for bit_pos in range(2, -1, -1):  # bits 2,1,0 of G
            raw_bits.append((g_val >> bit_pos) & 1)

    # OLSB positions: B lower k bits
    for i in range(n_olsb_used):
        y, x = olsb_positions[i]
        b_val = int(cover_rgb[y, x, 2])
        for bit_pos in range(k_olsb - 1, -1, -1):
            raw_bits.append((b_val >> bit_pos) & 1)

    # Pack bits to bytes
    if raw_bits:
        bits_arr = np.array(raw_bits, dtype=np.uint8)
        raw_data = bits_to_bytes(bits_arr)
    else:
        raw_data = b''

    # Prepend counts header
    counts_header = struct.pack('!II', n_emd_used, n_olsb_used)
    uncompressed = counts_header + raw_data

    # Compress
    return zlib.compress(uncompressed, level=9)


def _apply_location_map(
    stego_rgb: np.ndarray,
    location_map_data: bytes,
    emd_positions: np.ndarray,
    olsb_positions: np.ndarray,
    k_olsb: int = 3
) -> np.ndarray:
    """
    Restore original pixel values from the location map for exact cover recovery.

    Args:
        stego_rgb: The stego image (will be copied, not modified in-place).
        location_map_data: Compressed location map bytes.
        emd_positions: Array of (y, x) EMD positions (same order as embedding).
        olsb_positions: Array of (y, x) OLSB positions (same order as embedding).
        k_olsb: Number of LSB bits used in OLSB.

    Returns:
        Recovered cover image (H×W×3 uint8).
    """
    # Decompress
    uncompressed = zlib.decompress(location_map_data)

    # Parse counts
    n_emd_used, n_olsb_used = struct.unpack('!II', uncompressed[:8])
    raw_data = uncompressed[8:]

    # Unpack bits
    if raw_data:
        bits_arr = bytes_to_bits(raw_data)
    else:
        bits_arr = np.array([], dtype=np.uint8)

    recovered = stego_rgb.copy()
    bit_idx = 0

    # Restore EMD positions: R lower 3 bits + G lower 3 bits
    for i in range(n_emd_used):
        y, x = emd_positions[i]
        # Restore R lower 3 bits
        r_orig_low = 0
        for bit_pos in range(2, -1, -1):
            r_orig_low |= (int(bits_arr[bit_idx]) << bit_pos)
            bit_idx += 1
        # Restore G lower 3 bits
        g_orig_low = 0
        for bit_pos in range(2, -1, -1):
            g_orig_low |= (int(bits_arr[bit_idx]) << bit_pos)
            bit_idx += 1
        # Reconstruct original: upper bits from stego + original lower bits
        recovered[y, x, 0] = (int(stego_rgb[y, x, 0]) & 0xF8) | r_orig_low
        recovered[y, x, 1] = (int(stego_rgb[y, x, 1]) & 0xF8) | g_orig_low

    # Restore OLSB positions: B lower k bits
    mask_hi = (~((1 << k_olsb) - 1)) & 0xFF
    for i in range(n_olsb_used):
        y, x = olsb_positions[i]
        b_orig_low = 0
        for bit_pos in range(k_olsb - 1, -1, -1):
            b_orig_low |= (int(bits_arr[bit_idx]) << bit_pos)
            bit_idx += 1
        recovered[y, x, 2] = (int(stego_rgb[y, x, 2]) & mask_hi) | b_orig_low

    return recovered


# ===========================================================================
# PUBLIC EMBED FUNCTION (SINGLE STEGO OUTPUT)
# ===========================================================================

def embed_cnn_da_emd_olsb(
    cover_rgb: np.ndarray,
    secret_data: bytes,
    password: str,
    alpha: float = 0.5,
    beta:  float = 0.5,
    gamma: float = 0.6,
    t1:    float = 0.33,
    t2:    float = 0.66,
    payload_type: int = 0,
    model=None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Embed secret payload into cover RGB image using CNN-DA-EMD-OLSB.

    Returns:
        stego_rgb: Single uint8 stego image (H×W×3).
        stats: Embedding statistics dictionary.
    """
    h, w, c = cover_rgb.shape

    # 1 — Compute CNN distortion maps from upper 5 bitplanes (& 0xF8)
    upper = (cover_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper, alpha, beta, gamma, t1, t2, model)

    # 2 — Compute usable and theoretical capacity using actual masks
    cap_info = compute_capacity(cls_r, cls_g, cls_b, upper)
    emd_mask = cap_info['emd_mask']
    olsb_mask = cap_info['olsb_mask']
    usable_capacity_bits = cap_info['usable_capacity_bits']
    theoretical_capacity_bits = cap_info['theoretical_capacity_bits']
    total_emd_bits_cap = cap_info['usable_emd_bits']

    # 3 — Determine embedding positions (deterministic order)
    emd_positions = np.argwhere(emd_mask)    # (K, 2) array of (y, x) positions
    olsb_positions = np.argwhere(olsb_mask)  # (M, 2) array of (y, x) positions

    # 4 — Build location map: collect original lower bits of ALL potential positions
    #     We need a two-pass approach:
    #     Pass 1: Estimate payload size to know how many positions will be used
    #     Pass 2: Build location map for exactly those positions, then embed

    # First, prepare payload WITHOUT location map to estimate size
    payload_bytes_est = prepare_payload(
        secret_data, password, t1, t2, payload_type, gamma=gamma,
        location_map_data=None
    )
    payload_bits_est = bytes_to_bits(payload_bytes_est)
    est_total_bits = len(payload_bits_est)

    # Estimate how many EMD and OLSB positions will be used
    est_emd_bits = min(est_total_bits, total_emd_bits_cap)
    est_emd_bits = (est_emd_bits // 2) * 2
    est_emd_positions_used = est_emd_bits // 2
    est_olsb_bits = est_total_bits - est_emd_bits
    est_olsb_positions_used = (est_olsb_bits + 2) // 3  # ceil division

    # Build location map for the estimated positions
    locmap_data = _build_location_map(
        cover_rgb, emd_positions, min(est_emd_positions_used, len(emd_positions)),
        olsb_positions, min(est_olsb_positions_used, len(olsb_positions))
    )

    # Now prepare the real payload WITH location map
    payload_bytes = prepare_payload(
        secret_data, password, t1, t2, payload_type, gamma=gamma,
        location_map_data=locmap_data
    )
    payload_bits = bytes_to_bits(payload_bytes)
    total_bits = len(payload_bits)

    # Re-estimate positions with actual payload size (location map may change the count)
    # Iterate until convergence (typically 4-7 iterations)
    for _iter in range(15):
        emd_bits_needed = min(total_bits, total_emd_bits_cap)
        emd_bits_needed = (emd_bits_needed // 2) * 2
        emd_positions_used = emd_bits_needed // 2
        olsb_bits_needed = total_bits - emd_bits_needed
        olsb_positions_used = (olsb_bits_needed + 2) // 3

        new_locmap = _build_location_map(
            cover_rgb, emd_positions, min(emd_positions_used, len(emd_positions)),
            olsb_positions, min(olsb_positions_used, len(olsb_positions))
        )

        prev_len = len(locmap_data)
        locmap_data = new_locmap
        payload_bytes = prepare_payload(
            secret_data, password, t1, t2, payload_type, gamma=gamma,
            location_map_data=locmap_data
        )
        payload_bits = bytes_to_bits(payload_bytes)
        total_bits = len(payload_bits)

        if len(new_locmap) == prev_len:
            break  # Converged

    # Final capacity check
    if total_bits > usable_capacity_bits:
        raise ValueError(
            f"CNN-DA-EMD-OLSB: Payload + location map ({total_bits} bits) exceeds usable "
            f"image capacity ({usable_capacity_bits} bits = {cap_info['usable_emd_pairs']} "
            f"EMD pairs × 2 + {cap_info['usable_olsb_bits']} OLSB bits; theoretical upper "
            f"bound: {theoretical_capacity_bits} bits). "
            "Use a smaller payload or larger image."
        )

    # 5 — Split payload: first portion goes to EMD (as digits), remainder to OLSB (as bits)
    emd_bits_needed = min(total_bits, total_emd_bits_cap)
    emd_bits_needed = (emd_bits_needed // 2) * 2
    olsb_bits_needed = total_bits - emd_bits_needed

    # Convert EMD portion to digits
    emd_payload_bits = payload_bits[:emd_bits_needed]
    emd_digits = []
    for i in range(0, len(emd_payload_bits), 2):
        val = (int(emd_payload_bits[i]) << 1) | int(emd_payload_bits[i + 1])
        emd_digits.append(val)
    emd_digits = np.array(emd_digits, dtype=np.uint8)
    total_emd_digits_used = len(emd_digits)

    # OLSB portion remains as bits
    olsb_payload_bits = payload_bits[emd_bits_needed:emd_bits_needed + olsb_bits_needed]

    # 6 — Create single stego image
    stego = cover_rgb.copy().astype(np.int16)

    # --- EMD Embedding on R-G channel pairs ---
    actual_emd_bits = 0
    digit_idx = 0

    for pos_idx in range(len(emd_positions)):
        if digit_idx >= total_emd_digits_used:
            break

        y, x = emd_positions[pos_idx]
        p1 = int(cover_rgb[y, x, 0])  # R channel
        p2 = int(cover_rgb[y, x, 1])  # G channel
        s = int(emd_digits[digit_idx])

        # EMD Extraction Function: f = (p1 * 1 + p2 * 2) mod 5
        f = (p1 * 1 + p2 * 2) % 5
        d = (s - f) % 5

        # Single-stego EMD: choose minimal distortion modification within same 8-block (& 0xF8 immutable)
        r1 = p1 & 7
        r2 = p2 & 7
        if d == 0:
            p1_new, p2_new = p1, p2
        elif d == 1:
            p1_new = p1 + 1 if r1 < 7 else p1 - 4
            p2_new = p2
        elif d == 2:
            p1_new = p1
            p2_new = p2 + 1 if r2 < 7 else p2 - 4
        elif d == 3:
            p1_new = p1
            p2_new = p2 - 1 if r2 > 0 else p2 + 4
        else:  # d == 4
            p1_new = p1 - 1 if r1 > 0 else p1 + 4
            p2_new = p2

        stego[y, x, 0] = p1_new
        stego[y, x, 1] = p2_new
        digit_idx += 1
        actual_emd_bits += 2

    # --- OLSB Embedding on Blue channel (class 2 pixels) ---
    actual_olsb_bits = 0
    olsb_bit_idx = 0
    k_olsb = 3  # class 2 = 3 bits OLSB

    for pos_idx in range(len(olsb_positions)):
        if olsb_bit_idx >= len(olsb_payload_bits):
            break

        y, x = olsb_positions[pos_idx]
        orig_b = int(cover_rgb[y, x, 2])  # Blue channel

        bits_left = len(olsb_payload_bits) - olsb_bit_idx
        bits_to_embed = min(k_olsb, bits_left)

        # Build k-bit value from payload bits
        bit_val = 0
        for b in range(bits_to_embed):
            bit_val = (bit_val << 1) | int(olsb_payload_bits[olsb_bit_idx + b])

        # Shift remaining positions if fewer than k bits
        bit_val <<= (k_olsb - bits_to_embed)

        # Apply k-bit LSB replacement
        mask_hi = (~((1 << k_olsb) - 1)) & 0xFF
        new_b = (orig_b & mask_hi) | bit_val

        stego[y, x, 2] = np.clip(new_b, 0, 255)
        olsb_bit_idx += bits_to_embed
        actual_olsb_bits += bits_to_embed

    stego_rgb = stego.clip(0, 255).astype(np.uint8)

    # 7 — Verify embedding bit count
    actual_total_bits = actual_emd_bits + actual_olsb_bits
    if actual_total_bits < total_bits:
        raise RuntimeError(
            f"CNN-DA-EMD-OLSB: Embedding verification FAILED. "
            f"Expected {total_bits} bits but only embedded {actual_total_bits} bits "
            f"(EMD: {actual_emd_bits}, OLSB: {actual_olsb_bits}). "
            "This indicates a capacity accounting error."
        )

    # 8 — Compute statistics
    raw_secret_bits = len(secret_data) * 8
    raw_bpp = round(raw_secret_bits / (h * w), 6)
    embedded_bpp = round(total_bits / (h * w), 6)
    cap_util = round((total_bits / usable_capacity_bits * 100.0), 2) if usable_capacity_bits > 0 else 0.0

    stats = {
        'algorithm':                'CNN-DA-EMD-OLSB',
        'total_bits_embedded':      raw_secret_bits,
        'raw_payload_bits':         raw_secret_bits,
        'raw_payload_bytes':        len(secret_data),
        'raw_bpp':                  raw_bpp,
        'embedded_bitstream_bits':  total_bits,
        'embedded_bitstream_bytes': len(payload_bytes),
        'embedded_bpp':             embedded_bpp,
        'actual_emd_bits_embedded': actual_emd_bits,
        'actual_olsb_bits_embedded': actual_olsb_bits,
        'actual_total_bits_embedded': actual_total_bits,
        'total_emd_digits_embedded': total_emd_digits_used,
        'total_emd_bits_embedded':  emd_bits_needed,
        'total_olsb_bits_embedded': int(olsb_bit_idx),
        'payload_bytes':            len(payload_bytes),
        'internal_bits_embedded':   total_bits,
        'usable_capacity_bits':     usable_capacity_bits,
        'usable_capacity_bytes':    usable_capacity_bits // 8,
        'theoretical_capacity_bits': theoretical_capacity_bits,
        'theoretical_capacity_bytes': theoretical_capacity_bits // 8,
        'max_capacity_bits':        usable_capacity_bits,
        'capacity_utilization_%':   cap_util,
        'bpp':                      raw_bpp,
        'single_stego':             True,
        'dual_images':              False,
        'location_map_bytes':       len(locmap_data),
        'location_map_overhead_%':  round(len(locmap_data) * 8 / total_bits * 100, 2) if total_bits > 0 else 0.0,
        'total_digits_embedded':    total_emd_digits_used,
        'cnn_enabled':              gamma > 0.0,
        'cnn_trained':              model is not None,
        't1': t1, 't2': t2,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'model_name':               'CNN-DA-EMD-OLSB'
    }
    return stego_rgb, stats


# ===========================================================================
# PUBLIC EXTRACT FUNCTION (SINGLE STEGO INPUT)
# ===========================================================================

def extract_cnn_da_emd_olsb(
    stego_input=None,
    password: str = "Pass123!",
    alpha: float = 0.5,
    beta:  float = 0.5,
    gamma: float = 0.6,
    t1:    float = 0.33,
    t2:    float = 0.66,
    model=None,
    stego_rgb=None
) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
    """
    Extract payload and recover cover from a single stego image.

    Args:
        stego_input: Single stego image (H×W×3 uint8) or legacy tuple
                     (stego1, stego2) for backward compatibility.
        password:    Decryption password.
        stego_rgb:   Alias for stego_input.

    Returns:
        secret_data:     Decoded plaintext bytes.
        recovered_cover: Bit-exact recovered cover image (H×W×3 uint8).
        metadata:        Extraction metadata dict.
    """
    if stego_input is None and stego_rgb is not None:
        stego_input = stego_rgb

    # Handle both single-stego and legacy dual-stego input
    if isinstance(stego_input, tuple) and len(stego_input) == 2:
        # Legacy dual-stego: use S1 for extraction
        stego_rgb = stego_input[0]
    elif isinstance(stego_input, np.ndarray) and stego_input.ndim == 3:
        stego_rgb = stego_input
    else:
        raise ValueError(
            "CNN-DA-EMD-OLSB extraction requires a single stego image (H×W×3 uint8) "
            "or a legacy (stego1, stego2) tuple."
        )

    h, w, c = stego_rgb.shape

    # 1 — Recompute capacity class maps from stego upper bitplanes (& 0xF8)
    upper_stego = (stego_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, alpha, beta, gamma, t1, t2, model)
    cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_stego)
    emd_mask = cap_info['emd_mask']
    olsb_mask = cap_info['olsb_mask']

    # 2 — First pass: extract header bits to determine payload size & parameters
    header_bits_needed = HEADER_SIZE_BYTES * 8
    header_bits = _extract_bits(stego_rgb, emd_mask, olsb_mask, header_bits_needed)
    hdr_bytes = bits_to_bytes(np.array(header_bits[:header_bits_needed], dtype=np.uint8))

    if not _is_valid_header(hdr_bytes, h, w, c):
        # Search candidate gammas if header wasn't found with default/passed gamma
        for g_cand in [0.6, 0.5, 0.0, 1.0, 0.7, 0.8, 0.75, 0.65, 0.55, 0.45, 0.35, 0.85, 0.25, 0.15, 0.05, 0.95, 0.1, 0.2, 0.3, 0.4, 0.9]:
            if abs(g_cand - gamma) < 0.001:
                continue
            cls_r_c, cls_g_c, cls_b_c = _get_cap_maps(upper_stego, alpha, beta, g_cand, t1, t2, model)
            cap_info_c = compute_capacity(cls_r_c, cls_g_c, cls_b_c, upper_stego)
            hb_c = _extract_bits(stego_rgb, cap_info_c['emd_mask'], cap_info_c['olsb_mask'], header_bits_needed)
            hdr_c = bits_to_bytes(np.array(hb_c[:header_bits_needed], dtype=np.uint8))
            if _is_valid_header(hdr_c, h, w, c):
                gamma = g_cand
                hdr_bytes = hdr_c
                cls_r, cls_g, cls_b = cls_r_c, cls_g_c, cls_b_c
                cap_info = cap_info_c
                emd_mask = cap_info['emd_mask']
                olsb_mask = cap_info['olsb_mask']
                break

    cipher_len, t1_h, t2_h, gamma_h, locmap_size = _parse_header(hdr_bytes, h, w, c)

    # Recompute maps if stored parameters differ from passed parameters
    if abs(t1_h - t1) > 0.001 or abs(t2_h - t2) > 0.001 or abs(gamma_h - gamma) > 0.001:
        t1, t2, gamma = t1_h, t2_h, gamma_h
        cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, alpha, beta, gamma, t1, t2, model)
        cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_stego)
        emd_mask = cap_info['emd_mask']
        olsb_mask = cap_info['olsb_mask']

    # 3 — Second pass: extract full payload (header + locmap + ciphertext)
    total_bits_needed = (HEADER_SIZE_BYTES + locmap_size + cipher_len) * 8
    all_bits = _extract_bits(stego_rgb, emd_mask, olsb_mask, total_bits_needed)
    full_payload = bits_to_bytes(np.array(all_bits[:total_bits_needed], dtype=np.uint8))
    secret_data, metadata = parse_payload(full_payload, password)

    # 4 — Recover cover image via location map
    emd_positions = np.argwhere(emd_mask)
    olsb_positions = np.argwhere(olsb_mask)

    location_map_data = metadata.get('location_map_data')
    if location_map_data is not None and len(location_map_data) > 0:
        recovered_cover = _apply_location_map(
            stego_rgb, location_map_data, emd_positions, olsb_positions
        )
    else:
        # No location map — legacy dual-stego mode or no modifications
        # If we received a dual-stego tuple, use averaging as fallback
        if isinstance(stego_input, tuple) and len(stego_input) == 2:
            s1 = stego_input[0].astype(np.float64)
            s2 = stego_input[1].astype(np.float64)
            recovered_cover = np.round((s1 + s2) / 2.0).clip(0, 255).astype(np.uint8)
        else:
            recovered_cover = stego_rgb.copy()

    metadata['algorithm'] = 'CNN-DA-EMD-OLSB'
    metadata['alpha'] = alpha
    metadata['beta'] = beta
    metadata['gamma'] = gamma
    metadata['single_stego'] = True
    metadata['dual_images'] = False
    return secret_data, recovered_cover, metadata


def _extract_bits(
    stego_rgb: np.ndarray,
    emd_mask: np.ndarray,
    olsb_mask: np.ndarray,
    total_bits_needed: int
) -> list:
    """
    Extract payload bits from stego image using EMD (R-G pairs) + OLSB (Blue channel).
    Returns list of int bits.
    """
    extracted_bits = []

    # --- EMD extraction from R-G pairs (class 0/1 positions) ---
    emd_positions = np.argwhere(emd_mask)  # (K, 2)

    for pos_idx in range(len(emd_positions)):
        if len(extracted_bits) >= total_bits_needed:
            break

        y, x = emd_positions[pos_idx]
        p1 = int(stego_rgb[y, x, 0])  # R
        p2 = int(stego_rgb[y, x, 1])  # G

        # Extract secret digit: s = f(p1, p2) = (p1*1 + p2*2) mod 5
        s_digit = (p1 * 1 + p2 * 2) % 5

        # Convert digit to 2 bits
        b0 = (s_digit >> 1) & 1
        b1 = s_digit & 1
        extracted_bits.append(b0)
        if len(extracted_bits) < total_bits_needed:
            extracted_bits.append(b1)

    # --- OLSB extraction from Blue channel (class 2 positions) ---
    olsb_positions = np.argwhere(olsb_mask)  # (M, 2)
    k_olsb = 3

    for pos_idx in range(len(olsb_positions)):
        if len(extracted_bits) >= total_bits_needed:
            break

        y, x = olsb_positions[pos_idx]
        pixel_val = int(stego_rgb[y, x, 2])  # Blue channel

        mask_lo = (1 << k_olsb) - 1
        bit_val = pixel_val & mask_lo

        # Extract k bits (MSB first)
        for b in range(k_olsb):
            if len(extracted_bits) >= total_bits_needed:
                break
            bit = (bit_val >> (k_olsb - 1 - b)) & 1
            extracted_bits.append(bit)

    return extracted_bits
