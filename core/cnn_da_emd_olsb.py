"""
CNN-Guided Distortion-Aware Adaptive EMD-OLSB (CNN-DA-EMD-OLSB)
Core Algorithm Engine — Dual-Stego Reversible Data Hiding

Design Notes
============
1. Dual Stego Image Output:
   Produces TWO stego images (S1, S2). Cover is recovered exactly via
   p_orig = round((S1 + S2) / 2) for EMD-routed pixel pairs.

2. Distortion-Aware Adaptive Routing:
   Per-channel distortion maps D_r, D_g, D_b are computed from the upper
   bitplanes (& 0xF8). Pixels classified into:
     Class 0 (smooth):   EMD embedding via R-G pixel pairs
     Class 1 (moderate): EMD embedding via R-G pixel pairs
     Class 2 (textured): Adaptive k-bit OLSB on Blue channel

3. EMD (Exploiting Modification Direction):
   Uses mod-5 extraction function f(p1,p2) = (p1*1 + p2*2) mod 5 on R-G
   channel pairs, producing two candidate pairs per the standard dual-image
   EMD rule. Secret digits are base-5 encoded.

4. OLSB (Optimal LSB Substitution):
   For class 2 (high-texture) pixels, k-bit LSB replacement is applied
   to the Blue channel. The same modified value is written into both S1 and
   S2 (OLSB pixels are identical in both stego images).

5. Deterministic Upper-Bitplane Capacity Recovery:
   Because only lower LSBs (bits 0..2) are modified during embedding while
   upper bits (& 0xF8) remain completely untouched, the capacity maps
   computed during extraction match the embedding maps exactly.

6. AES-256-GCM Authenticated Encryption:
   Payload is compressed, encrypted via AES-256-GCM, and packed with a
   64-byte deterministic header before embedding.
"""

import numpy as np
import struct
from typing import Tuple, Dict, Any

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
    alpha: float, beta: float,
    t1: float, t2: float,
    model=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-channel distortion class maps (H×W uint8, values ∈ {0, 1, 2})
    using CNN distortion guidance (or analytic Sobel+Var fallback).

    Class 0: D < t1  (smooth — EMD)
    Class 1: t1 ≤ D < t2 (moderate — EMD)
    Class 2: D ≥ t2  (textured — OLSB)
    """
    D_r_raw, D_g_raw, D_b_raw = compute_distortion_maps(
        img_rgb, model=model, alpha=alpha, beta=beta
    )

    def _to_class(d_raw: np.ndarray) -> np.ndarray:
        d = _normalize_01(d_raw)
        cls = np.zeros(d.shape, dtype=np.uint8)  # class 0 by default
        cls[d >= t1] = CAP_CLASS1
        cls[d >= t2] = CAP_CLASS2
        return cls

    return _to_class(D_r_raw), _to_class(D_g_raw), _to_class(D_b_raw)


def _parse_header(header_bytes: bytes, h: int, w: int, c: int) -> Tuple[int, float, float]:
    """Parse embedded header; return (cipher_len, t1, t2)."""
    try:
        magic, _, _, _, cipher_len, _, _, t1, t2, _, _ = struct.unpack(
            '!4sBB2sI16s12sffI12s', header_bytes
        )
        if magic == HEADER_MAGIC:
            return int(cipher_len), float(t1), float(t2)
    except Exception:
        pass
    return max(64, (h * w * c) // 32), 0.33, 0.66


# ===========================================================================
# CAPACITY COMPUTATION
# ===========================================================================

def _compute_emd_olsb_capacity(
    cls_r: np.ndarray,  # (H, W) uint8 class map for Red
    cls_g: np.ndarray,  # (H, W) uint8 class map for Green
    cls_b: np.ndarray,  # (H, W) uint8 class map for Blue
) -> Tuple[int, int]:
    """
    Compute total embeddable capacity.

    EMD pairs (R-G): A pixel position is EMD-routed if BOTH R and G channels
    are class 0 or class 1 at that position. Each such pixel pair carries
    one base-5 digit (≈2.32 bits, but exactly 2 bits in our encoding).

    OLSB pixels (B): Blue channel at class 2 positions carries k bits via LSB.
    For simplicity, class 2 on Blue = 3 bits OLSB.

    Returns (total_digits_capacity, total_olsb_bits_capacity).
    """
    h, w = cls_r.shape

    # EMD routing: pixel positions where BOTH R and G are class 0 or 1
    emd_mask = ((cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1))  # (H, W) bool
    emd_pixel_count = int(np.sum(emd_mask))
    # Each EMD pixel position contributes one R-G pair → one base-5 digit
    total_emd_digits = emd_pixel_count

    # OLSB routing: Blue channel where class == 2
    olsb_mask = (cls_b == CAP_CLASS2)  # (H, W) bool
    olsb_pixel_count = int(np.sum(olsb_mask))
    # Class 2 carries 3 bits OLSB per pixel on Blue channel
    total_olsb_bits = olsb_pixel_count * 3

    return total_emd_digits, total_olsb_bits


# ===========================================================================
# PUBLIC EMBED FUNCTION (DUAL STEGO OUTPUT)
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
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """
    Embed secret payload into cover RGB image using CNN-DA-EMD-OLSB.

    Returns:
        (stego1_rgb, stego2_rgb): Dual uint8 stego images (H×W×3).
        stats: Embedding statistics dictionary.
    """
    h, w, c = cover_rgb.shape

    # 1 — Prepare AES-256-GCM payload with header
    payload_bytes = prepare_payload(secret_data, password, t1, t2, payload_type)

    # 2 — Compute CNN distortion maps from upper 5 bitplanes (& 0xF8)
    upper = (cover_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper, alpha, beta, t1, t2, model)

    # 3 — Build routing masks with boundary safety (avoid 0 and 255 clipping)
    emd_mask = (
        (cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1) &
        (upper[:, :, 0] >= 8) & (upper[:, :, 0] <= 240) &
        (upper[:, :, 1] >= 8) & (upper[:, :, 1] <= 240)
    )  # (H, W) bool
    olsb_mask = (cls_b == CAP_CLASS2)  # (H, W) bool

    # 4 — Compute capacity
    emd_digit_cap, olsb_bit_cap = _compute_emd_olsb_capacity(cls_r, cls_g, cls_b)
    # EMD: 1 digit = 2 bits of payload, OLSB: direct bits
    total_emd_bits_cap = emd_digit_cap * 2
    total_capacity_bits = total_emd_bits_cap + olsb_bit_cap

    # Convert payload to base-5 digits (for EMD portion) and bits (for OLSB portion)
    payload_bits = bytes_to_bits(payload_bytes)
    total_bits = len(payload_bits)

    if total_bits > total_capacity_bits:
        raise ValueError(
            f"CNN-DA-EMD-OLSB: Payload ({total_bits} bits) exceeds image "
            f"capacity ({total_capacity_bits} bits = {emd_digit_cap} EMD digits × 2 + "
            f"{olsb_bit_cap} OLSB bits). Use a smaller payload or larger image."
        )

    # Split payload: first portion goes to EMD (as digits), remainder to OLSB (as bits)
    emd_bits_needed = min(total_bits, total_emd_bits_cap)
    # Make emd_bits_needed even (digits carry 2 bits each)
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

    # 5 — Create dual stego images
    s1 = cover_rgb.copy().astype(np.int16)
    s2 = cover_rgb.copy().astype(np.int16)

    # --- EMD Embedding on R-G channel pairs ---
    emd_positions = np.argwhere(emd_mask)  # (K, 2) array of (y, x) positions
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

        if d == 0:
            p1_1, p2_1 = p1, p2
            p1_2, p2_2 = p1, p2
        elif d == 1:
            p1_1 = p1 - 4 if (p1 & 7) == 7 else p1 + 1
            p2_1 = p2
            p1_2 = p1 + 4 if (p1 & 7) == 0 else p1 - 1
            p2_2 = p2
        elif d == 2:
            p1_1 = p1
            p2_1 = p2 - 4 if (p2 & 7) == 7 else p2 + 1
            p1_2 = p1
            p2_2 = p2 + 4 if (p2 & 7) == 0 else p2 - 1
        elif d == 3:
            p1_1 = p1
            p2_1 = p2 + 4 if (p2 & 7) == 0 else p2 - 1
            p1_2 = p1
            p2_2 = p2 - 4 if (p2 & 7) == 7 else p2 + 1
        else:  # d == 4
            p1_1 = p1 + 4 if (p1 & 7) == 0 else p1 - 1
            p2_1 = p2
            p1_2 = p1 - 4 if (p1 & 7) == 7 else p1 + 1
            p2_2 = p2

        s1[y, x, 0] = p1_1
        s1[y, x, 1] = p2_1
        s2[y, x, 0] = p1_2
        s2[y, x, 1] = p2_2

        digit_idx += 1

    # --- OLSB Embedding on Blue channel (class 2 pixels) ---
    olsb_positions = np.argwhere(olsb_mask)  # (M, 2) array of (y, x) positions
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

        # Write identical value to both stego images
        s1[y, x, 2] = np.clip(new_b, 0, 255)
        s2[y, x, 2] = np.clip(new_b, 0, 255)

        olsb_bit_idx += bits_to_embed

    stego1_rgb = s1.clip(0, 255).astype(np.uint8)
    stego2_rgb = s2.clip(0, 255).astype(np.uint8)

    # Report total_bits_embedded as the raw secret payload size (len(secret_data) * 8)
    # to match the convention used by all other benchmark models (MPEH-RDH, MCSH-RDH,
    # CNN-RDH, SRDNN-Stego, EMD-OLSB). The internal prepared payload (header + AES
    # ciphertext) is larger due to the 64-byte header and encryption overhead, but BPP
    # must be computed on the same basis for fair cross-model comparison.
    raw_secret_bits = len(secret_data) * 8

    stats = {
        'algorithm':                'CNN-DA-EMD-OLSB',
        'total_bits_embedded':      raw_secret_bits,
        'total_emd_digits_embedded': total_emd_digits_used,
        'total_emd_bits_embedded':  emd_bits_needed,
        'total_olsb_bits_embedded': int(olsb_bit_idx),
        'payload_bytes':            len(payload_bytes),
        'internal_bits_embedded':   total_bits,
        'max_capacity_bits':        total_capacity_bits,
        'bpp':                      round(raw_secret_bits / (h * w * c), 4),
        'dual_images':              True,
        'total_digits_embedded':    total_emd_digits_used,
        't1': t1, 't2': t2,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'model_name':               'CNN-DA-EMD-OLSB'
    }
    return (stego1_rgb, stego2_rgb), stats


# ===========================================================================
# PUBLIC EXTRACT FUNCTION (DUAL STEGO INPUT)
# ===========================================================================

def extract_cnn_da_emd_olsb(
    stego_dual: Tuple[np.ndarray, np.ndarray],
    password: str,
    alpha: float = 0.5,
    beta:  float = 0.5,
    gamma: float = 0.6,
    t1:    float = 0.33,
    t2:    float = 0.66,
    model=None
) -> Tuple[bytes, np.ndarray, Dict[str, Any]]:
    """
    Extract payload and recover cover from dual stego images.

    Args:
        stego_dual: Tuple (stego1_rgb, stego2_rgb), each H×W×3 uint8.
        password:   Decryption password.

    Returns:
        secret_data:     Decoded plaintext bytes.
        recovered_cover: Bit-exact recovered cover image (H×W×3 uint8).
        metadata:        Extraction metadata dict.
    """
    # Accept tuple or single image (legacy compat)
    if isinstance(stego_dual, tuple) and len(stego_dual) == 2:
        stego1_rgb, stego2_rgb = stego_dual
    else:
        raise ValueError(
            "CNN-DA-EMD-OLSB extraction requires a (stego1, stego2) tuple. "
            "Got a single image instead."
        )

    h, w, c = stego1_rgb.shape

    # 1 — Recompute capacity class maps from stego upper bitplanes (& 0xF8)
    #     Upper bitplanes are identical in S1 and S2 since only low bits differ.
    upper_stego = (stego1_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, alpha, beta, t1, t2, model)

    # Build routing masks (same as embedding)
    emd_mask = (
        (cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1) &
        (upper_stego[:, :, 0] >= 8) & (upper_stego[:, :, 0] <= 240) &
        (upper_stego[:, :, 1] >= 8) & (upper_stego[:, :, 1] <= 240)
    )
    olsb_mask = (cls_b == CAP_CLASS2)

    # 2 — First pass: extract header bits to determine payload size
    #     Header is in the first portion of the EMD+OLSB bitstream
    header_bits_needed = HEADER_SIZE_BYTES * 8

    # Extract enough bits for the header using combined EMD+OLSB extraction
    header_bits = _extract_bits(stego1_rgb, emd_mask, olsb_mask, header_bits_needed)
    hdr_bytes = bits_to_bytes(np.array(header_bits[:header_bits_needed], dtype=np.uint8))
    cipher_len, t1_h, t2_h = _parse_header(hdr_bytes, h, w, c)

    # Recompute maps if stored thresholds differ
    if abs(t1_h - t1) > 0.001 or abs(t2_h - t2) > 0.001:
        cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, alpha, beta, t1_h, t2_h, model)
        emd_mask = ((cls_r <= CAP_CLASS1) & (cls_g <= CAP_CLASS1))
        olsb_mask = (cls_b == CAP_CLASS2)

    # 3 — Second pass: extract full payload (header + ciphertext)
    total_bits_needed = (HEADER_SIZE_BYTES + cipher_len) * 8
    all_bits = _extract_bits(stego1_rgb, emd_mask, olsb_mask, total_bits_needed)
    full_payload = bits_to_bytes(np.array(all_bits[:total_bits_needed], dtype=np.uint8))
    secret_data, metadata = parse_payload(full_payload, password)

    # 4 — Recover cover image via dual-image averaging
    recovered_cover = _recover_cover(stego1_rgb, stego2_rgb, emd_mask, olsb_mask)

    metadata['algorithm'] = 'CNN-DA-EMD-OLSB'
    metadata['alpha'] = alpha
    metadata['beta'] = beta
    metadata['dual_images'] = True
    return secret_data, recovered_cover, metadata


def _extract_bits(
    stego1_rgb: np.ndarray,
    emd_mask: np.ndarray,
    olsb_mask: np.ndarray,
    total_bits_needed: int
) -> list:
    """
    Extract payload bits from stego1 using EMD (R-G pairs) + OLSB (Blue channel).
    Returns list of int bits.
    """
    extracted_bits = []

    # --- EMD extraction from R-G pairs (class 0/1 positions) ---
    emd_positions = np.argwhere(emd_mask)  # (K, 2)

    for pos_idx in range(len(emd_positions)):
        if len(extracted_bits) >= total_bits_needed:
            break

        y, x = emd_positions[pos_idx]
        p1_1 = int(stego1_rgb[y, x, 0])  # R from S1
        p2_1 = int(stego1_rgb[y, x, 1])  # G from S1

        # Extract secret digit: s = f(p1_1, p2_1) = (p1_1*1 + p2_1*2) mod 5
        s_digit = (p1_1 * 1 + p2_1 * 2) % 5

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
        pixel_val = int(stego1_rgb[y, x, 2])  # Blue channel

        mask_lo = (1 << k_olsb) - 1
        bit_val = pixel_val & mask_lo

        # Extract k bits (MSB first)
        for b in range(k_olsb):
            if len(extracted_bits) >= total_bits_needed:
                break
            bit = (bit_val >> (k_olsb - 1 - b)) & 1
            extracted_bits.append(bit)

    return extracted_bits


def _recover_cover(
    stego1_rgb: np.ndarray,
    stego2_rgb: np.ndarray,
    emd_mask: np.ndarray,
    olsb_mask: np.ndarray
) -> np.ndarray:
    """
    Recover original cover image from dual stego images with 8-block wrapping support.
    """
    s1 = stego1_rgb.astype(np.int16)
    s2 = stego2_rgb.astype(np.int16)
    recovered = np.zeros_like(stego1_rgb, dtype=np.uint8)

    # For non-EMD pixels or default averaging:
    rec_float = np.round((s1.astype(np.float64) + s2.astype(np.float64)) / 2.0)
    recovered[:, :, :] = np.clip(rec_float, 0, 255).astype(np.uint8)

    # For EMD-modified R and G channels, apply exact 8-block wrapping recovery:
    emd_positions = np.argwhere(emd_mask)
    for pos_idx in range(len(emd_positions)):
        y, x = emd_positions[pos_idx]
        for c in (0, 1):  # R and G channels
            v1 = int(s1[y, x, c])
            v2 = int(s2[y, x, c])
            if abs(v1 - v2) <= 2:
                p_rec = (v1 + v2) // 2
            elif (v1 & 7) == 3 and (v2 & 7) == 6:
                p_rec = v2 + 1
            elif (v1 & 7) == 1 and (v2 & 7) == 4:
                p_rec = v1 - 1
            elif (v1 & 7) == 6 and (v2 & 7) == 3:
                p_rec = v1 + 1
            elif (v2 & 7) == 1 and (v1 & 7) == 4:
                p_rec = v2 - 1
            else:
                p_rec = (v1 + v2) // 2
            recovered[y, x, c] = np.uint8(np.clip(p_rec, 0, 255))

    return recovered
