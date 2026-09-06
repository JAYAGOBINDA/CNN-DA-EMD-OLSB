"""
CNN-Guided Distortion-Aware Adaptive EMD-OLSB (CNN-DA-EMD-OLSB)
Core Algorithm Engine — Single-Stego Reversible Data Hiding

Design Notes
============
1. Single Stego Image Output:
   Produces ONE stego image. Cover is recovered exactly via compressed
   Recovery Side Information that stores the original lower bits of every
   modified pixel. This guarantees np.array_equal(cover, recovered_cover) == True.

2. Deterministic Bootstrap Header Region:
   The first BOOTSTRAP_N_PIXELS pixels (raster order) are RESERVED for the
   64-byte header using Blue-channel LSB embedding. This region is gamma-
   independent: the extractor always reads it first to recover gamma, t1, t2,
   payload length, and recovery side-information length — without any candidate
   guessing. The main adaptive embedding DOES NOT touch bootstrap pixels.

3. Distortion-Aware Adaptive Routing:
   Per-channel distortion maps D_r, D_g, D_b are computed from the upper
   bitplanes (& 0xF8). Pixels classified into:
     Class 0 (smooth):   EMD embedding via R-G pixel pairs
     Class 1 (moderate): EMD embedding via R-G pixel pairs
     Class 2 (textured): Adaptive k-bit OLSB on Blue channel

4. EMD (Exploiting Modification Direction):
   Uses mod-5 extraction function f(p1,p2) = (p1*1 + p2*2) mod 5 on R-G
   channel pairs. Single minimal-distortion candidate is chosen.
   Secret digits are base-5 encoded.

5. OLSB (Optimal LSB Substitution):
   For class 2 (high-texture) pixels, k-bit LSB replacement is applied
   to the Blue channel.

6. Compressed Recovery Side Information:
   Before embedding, the original lower 3 bits of every pixel that will
   be modified are collected, compressed with zlib, and embedded as part
   of the payload. During extraction, this map is decompressed and used
   to restore every modified pixel to its exact original value.
   The bootstrap region's original Blue LSBs are also stored.

7. Deterministic Upper-Bitplane Capacity Recovery:
   Because only lower LSBs (bits 0..2) are modified during embedding while
   upper bits (& 0xF8) remain completely untouched, the capacity maps
   computed during extraction match the embedding maps exactly.

8. AES-256-GCM Authenticated Encryption:
   Payload is compressed, encrypted via AES-256-GCM, and packed with a
   64-byte deterministic header before embedding. The recovery side
   information is authenticated as AES-GCM Associated Data (AAD).
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
from cnn.distortion_cnn import compute_distortion_maps, was_cnn_inference_executed


# ===========================================================================
# CONSTANTS
# ===========================================================================

CAP_CLASS0 = 0   # Smooth regions   — EMD embedding
CAP_CLASS1 = 1   # Moderate texture — EMD embedding
CAP_CLASS2 = 2   # High texture     — OLSB embedding

# Bootstrap: one Blue-channel LSB per pixel → need (HEADER_SIZE_BYTES * 8) pixels
BOOTSTRAP_N_PIXELS = HEADER_SIZE_BYTES * 8   # 64 * 8 = 512 pixels


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
    raise ValueError(
        "CNN-DA-EMD-OLSB: Cannot parse header. The stego image may be corrupted "
        "or was created with an incompatible format."
    )


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
# BOOTSTRAP REGION HELPERS
# ===========================================================================

def _get_bootstrap_positions(h: int, w: int) -> np.ndarray:
    """
    Return (BOOTSTRAP_N_PIXELS, 2) array of (y, x) bootstrap positions in
    raster-scan order (row by row, left to right).

    These positions are deterministic and gamma-independent, ensuring the
    extractor can always locate the bootstrap header.
    """
    n = min(BOOTSTRAP_N_PIXELS, h * w)
    flat_indices = np.arange(n, dtype=np.intp)
    ys = flat_indices // w
    xs = flat_indices % w
    return np.stack([ys, xs], axis=1)


# ===========================================================================
# COMPRESSED RECOVERY SIDE INFORMATION
# ===========================================================================

def _build_recovery_side_info(
    cover_rgb: np.ndarray,
    emd_positions: np.ndarray,
    n_emd_used: int,
    olsb_positions: np.ndarray,
    n_olsb_used: int,
    bootstrap_positions: np.ndarray,
    bootstrap_orig_blue_lsbs: np.ndarray,
    k_olsb: int = 3
) -> bytes:
    """
    Build compressed recovery side information storing the original lower bits
    of every pixel modified during embedding for exact cover recovery.

    For EMD positions: store lower 3 bits of R and G (6 bits per position).
    For OLSB positions: store lower k_olsb bits of B (3 bits per position).
    For Bootstrap positions: store original Blue LSB (1 bit per position).

    Structure:
      [4 bytes: n_emd_used (uint32)]
      [4 bytes: n_olsb_used (uint32)]
      [4 bytes: n_bootstrap (uint32)]
      [EMD original bits: n_emd_used * 6 bits, packed]
      [OLSB original bits: n_olsb_used * 3 bits, packed]
      [Bootstrap original bits: n_bootstrap * 1 bit, packed]

    Returns zlib-compressed bytes.
    """
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

    # Bootstrap positions: Blue LSB
    n_bootstrap = len(bootstrap_positions)
    for i in range(n_bootstrap):
        raw_bits.append(int(bootstrap_orig_blue_lsbs[i]))

    # Pack bits to bytes
    if raw_bits:
        bits_arr = np.array(raw_bits, dtype=np.uint8)
        raw_data = bits_to_bytes(bits_arr)
    else:
        raw_data = b''

    # Prepend counts header (3 × uint32 = 12 bytes)
    counts_header = struct.pack('!III', n_emd_used, n_olsb_used, n_bootstrap)
    uncompressed = counts_header + raw_data

    return zlib.compress(uncompressed, level=9)


def _apply_recovery_side_info(
    stego_rgb: np.ndarray,
    side_info_data: bytes,
    emd_positions: np.ndarray,
    olsb_positions: np.ndarray,
    bootstrap_positions: np.ndarray,
    k_olsb: int = 3
) -> np.ndarray:
    """
    Restore original pixel values from compressed recovery side information
    for exact cover recovery.

    Args:
        stego_rgb: The stego image (will be copied, not modified in-place).
        side_info_data: Compressed recovery side information bytes.
        emd_positions: Array of (y, x) EMD positions (same order as embedding).
        olsb_positions: Array of (y, x) OLSB positions (same order as embedding).
        bootstrap_positions: Array of (y, x) bootstrap positions.
        k_olsb: Number of LSB bits used in OLSB.

    Returns:
        Recovered cover image (H×W×3 uint8).
    """
    uncompressed = zlib.decompress(side_info_data)

    # Parse counts — new format (3 fields, 12 bytes header)
    if len(uncompressed) >= 12:
        n_emd_used, n_olsb_used, n_bootstrap = struct.unpack('!III', uncompressed[:12])
        raw_data = uncompressed[12:]
    else:
        # Legacy format (2 fields, 8 bytes header) — no bootstrap
        n_emd_used, n_olsb_used = struct.unpack('!II', uncompressed[:8])
        n_bootstrap = 0
        raw_data = uncompressed[8:]

    if raw_data:
        bits_arr = bytes_to_bits(raw_data)
    else:
        bits_arr = np.array([], dtype=np.uint8)

    recovered = stego_rgb.copy()
    bit_idx = 0

    # Restore EMD positions: R lower 3 bits + G lower 3 bits
    for i in range(n_emd_used):
        y, x = emd_positions[i]
        r_orig_low = 0
        for bit_pos in range(2, -1, -1):
            r_orig_low |= (int(bits_arr[bit_idx]) << bit_pos)
            bit_idx += 1
        g_orig_low = 0
        for bit_pos in range(2, -1, -1):
            g_orig_low |= (int(bits_arr[bit_idx]) << bit_pos)
            bit_idx += 1
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

    # Restore bootstrap positions: Blue LSB
    for i in range(n_bootstrap):
        y, x = bootstrap_positions[i]
        orig_lsb = int(bits_arr[bit_idx])
        bit_idx += 1
        recovered[y, x, 2] = (int(stego_rgb[y, x, 2]) & 0xFE) | orig_lsb

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

    The embedding uses a reserved bootstrap region (first 512 pixels in raster
    order) for the 64-byte header via Blue-channel LSB. The main adaptive
    EMD+OLSB embedding occupies the remaining non-bootstrap region. Recovery
    side information is embedded alongside the encrypted payload so that every
    modified pixel can be exactly restored.

    Returns:
        stego_rgb: Single uint8 stego image (H×W×3).
        stats: Embedding statistics dictionary.
    """
    h, w, c = cover_rgb.shape

    # ── Step 1: Reserve bootstrap region ──────────────────────────────────
    n_bootstrap = BOOTSTRAP_N_PIXELS
    total_pixels = h * w
    if total_pixels < n_bootstrap:
        raise ValueError(
            f"CNN-DA-EMD-OLSB: Image too small ({h}×{w} = {total_pixels} pixels). "
            f"Need at least {n_bootstrap} pixels for bootstrap header."
        )

    bootstrap_yx = _get_bootstrap_positions(h, w)
    bootstrap_mask = np.zeros((h, w), dtype=bool)
    bootstrap_mask[bootstrap_yx[:, 0], bootstrap_yx[:, 1]] = True

    # Save original Blue LSBs of bootstrap pixels for recovery side info
    bootstrap_orig_lsbs = np.array(
        [int(cover_rgb[y, x, 2]) & 1 for y, x in bootstrap_yx], dtype=np.uint8
    )

    # ── Step 2: Compute CNN distortion maps from upper 5 bitplanes (& 0xF8)
    upper = (cover_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper, alpha, beta, gamma, t1, t2, model)

    # Track CNN inference status
    cnn_inference_ran = was_cnn_inference_executed()

    # ── Step 3: Compute capacity with bootstrap exclusion ─────────────────
    cap_info = compute_capacity(cls_r, cls_g, cls_b, upper)
    emd_mask = cap_info['emd_mask']
    olsb_mask = cap_info['olsb_mask']
    theoretical_capacity_bits = cap_info['theoretical_capacity_bits']

    # Get EMD and OLSB positions, EXCLUDING bootstrap pixels
    all_emd = np.argwhere(emd_mask)
    all_olsb = np.argwhere(olsb_mask)

    if len(all_emd) > 0:
        keep_emd = ~bootstrap_mask[all_emd[:, 0], all_emd[:, 1]]
        emd_positions = all_emd[keep_emd]
    else:
        emd_positions = np.empty((0, 2), dtype=np.intp)

    if len(all_olsb) > 0:
        keep_olsb = ~bootstrap_mask[all_olsb[:, 0], all_olsb[:, 1]]
        olsb_positions = all_olsb[keep_olsb]
    else:
        olsb_positions = np.empty((0, 2), dtype=np.intp)

    # Recalculate usable capacity after bootstrap exclusion
    total_emd_bits_cap = len(emd_positions) * 2
    usable_olsb_bits = len(olsb_positions) * 3
    usable_body_capacity = total_emd_bits_cap + usable_olsb_bits

    # ── Step 4: Iterative convergence for side info sizing ────────────────
    # First estimate without side info
    payload_bytes_est = prepare_payload(
        secret_data, password, t1, t2, payload_type, gamma=gamma,
        location_map_data=None
    )
    body_est = payload_bytes_est[HEADER_SIZE_BYTES:]
    body_bits_est = len(body_est) * 8

    # Estimate positions used
    est_emd_bits = min(body_bits_est, total_emd_bits_cap)
    est_emd_bits = (est_emd_bits // 2) * 2
    est_emd_used = est_emd_bits // 2
    est_olsb_bits = body_bits_est - est_emd_bits
    est_olsb_used = (est_olsb_bits + 2) // 3

    # Build recovery side info for estimated positions + bootstrap
    side_info = _build_recovery_side_info(
        cover_rgb,
        emd_positions, min(est_emd_used, len(emd_positions)),
        olsb_positions, min(est_olsb_used, len(olsb_positions)),
        bootstrap_yx, bootstrap_orig_lsbs
    )

    # Rebuild payload with side info
    payload_bytes = prepare_payload(
        secret_data, password, t1, t2, payload_type, gamma=gamma,
        location_map_data=side_info
    )
    body = payload_bytes[HEADER_SIZE_BYTES:]
    body_bits = len(body) * 8

    # Iterate until converged (typically 4-7 iterations)
    for _iter in range(15):
        emd_bits_needed = min(body_bits, total_emd_bits_cap)
        emd_bits_needed = (emd_bits_needed // 2) * 2
        emd_used = emd_bits_needed // 2
        olsb_bits_needed = body_bits - emd_bits_needed
        olsb_used = (olsb_bits_needed + 2) // 3

        new_side_info = _build_recovery_side_info(
            cover_rgb,
            emd_positions, min(emd_used, len(emd_positions)),
            olsb_positions, min(olsb_used, len(olsb_positions)),
            bootstrap_yx, bootstrap_orig_lsbs
        )

        prev_len = len(side_info)
        side_info = new_side_info

        payload_bytes = prepare_payload(
            secret_data, password, t1, t2, payload_type, gamma=gamma,
            location_map_data=side_info
        )
        body = payload_bytes[HEADER_SIZE_BYTES:]
        body_bits = len(body) * 8

        if len(new_side_info) == prev_len:
            break  # Converged

    # ── Step 5: Final capacity check ──────────────────────────────────────
    if body_bits > usable_body_capacity:
        raise ValueError(
            f"CNN-DA-EMD-OLSB: Payload body ({body_bits} bits) exceeds "
            f"non-bootstrap capacity ({usable_body_capacity} bits = "
            f"{len(emd_positions)} EMD pairs × 2 + {len(olsb_positions)} OLSB × 3). "
            "Use a smaller payload or larger image."
        )

    # ── Step 6: Split payload: header → bootstrap, body → EMD+OLSB ───────
    header_bytes = payload_bytes[:HEADER_SIZE_BYTES]
    body_bytes = body

    header_bits = bytes_to_bits(header_bytes)   # exactly 512 bits
    body_bit_arr = bytes_to_bits(body_bytes)
    total_body_bits = len(body_bit_arr)

    # EMD/OLSB split for body
    emd_bits_needed = min(total_body_bits, total_emd_bits_cap)
    emd_bits_needed = (emd_bits_needed // 2) * 2
    olsb_bits_needed = total_body_bits - emd_bits_needed

    emd_payload_bits = body_bit_arr[:emd_bits_needed]
    olsb_payload_bits = body_bit_arr[emd_bits_needed:emd_bits_needed + olsb_bits_needed]

    # Convert EMD bits to digits
    emd_digits = []
    for i in range(0, len(emd_payload_bits), 2):
        val = (int(emd_payload_bits[i]) << 1) | int(emd_payload_bits[i + 1])
        emd_digits.append(val)
    emd_digits = np.array(emd_digits, dtype=np.uint8)
    total_emd_digits_used = len(emd_digits)

    # ── Step 7: Create single stego image ─────────────────────────────────
    stego = cover_rgb.copy().astype(np.int16)

    # 7a. Embed header in bootstrap region (Blue channel LSB)
    for i in range(n_bootstrap):
        y, x = int(bootstrap_yx[i, 0]), int(bootstrap_yx[i, 1])
        bit = int(header_bits[i])
        stego[y, x, 2] = (int(cover_rgb[y, x, 2]) & 0xFE) | bit

    # 7b. EMD Embedding on R-G channel pairs (non-bootstrap positions)
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

    # 7c. OLSB Embedding on Blue channel (non-bootstrap class 2 pixels)
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

    # ── Step 8: Verify embedding bit counts (exact equality) ──────────────
    actual_body_bits = actual_emd_bits + actual_olsb_bits
    if actual_body_bits != total_body_bits:
        raise RuntimeError(
            f"CNN-DA-EMD-OLSB: Embedding verification FAILED. "
            f"Expected {total_body_bits} body bits but embedded {actual_body_bits} "
            f"(EMD: {actual_emd_bits}, OLSB: {actual_olsb_bits})."
        )

    actual_total_bits = n_bootstrap + actual_body_bits

    # ── Step 9: Compute statistics ────────────────────────────────────────
    raw_secret_bits = len(secret_data) * 8
    raw_bpp = round(raw_secret_bits / (h * w), 6)
    embedded_bpp = round(actual_total_bits / (h * w), 6)
    usable_total = usable_body_capacity + n_bootstrap
    cap_util = round((actual_total_bits / usable_total * 100.0), 2) if usable_total > 0 else 0.0

    stats = {
        'algorithm':                  'CNN-DA-EMD-OLSB',
        # ── Payload accounting ──
        'raw_secret_bits':            raw_secret_bits,
        'raw_secret_bytes':           len(secret_data),
        'raw_bpp':                    raw_bpp,
        'prepared_payload_bits':      len(payload_bytes) * 8,
        'prepared_payload_bytes':     len(payload_bytes),
        'recovery_side_info_bits':    len(side_info) * 8,
        'recovery_side_info_bytes':   len(side_info),
        # ── Actual embedded counts ──
        'actual_embedded_bits':       actual_total_bits,
        'actual_bootstrap_bits':      n_bootstrap,
        'actual_emd_bits_embedded':   actual_emd_bits,
        'actual_olsb_bits_embedded':  actual_olsb_bits,
        'actual_body_bits':           actual_body_bits,
        'embedded_bpp':               embedded_bpp,
        # ── Capacity ──
        'usable_capacity_bits':       usable_total,
        'usable_capacity_bytes':      usable_total // 8,
        'theoretical_capacity_bits':  theoretical_capacity_bits,
        'theoretical_capacity_bytes': theoretical_capacity_bits // 8,
        'max_capacity_bits':          usable_total,
        'capacity_utilization_%':     cap_util,
        'bpp':                        raw_bpp,
        # ── Flags ──
        'single_stego':               True,
        'dual_images':                False,
        'cnn_enabled':                gamma > 0.0,
        'cnn_trained':                model is not None,
        'cnn_inference_executed':     cnn_inference_ran,
        # ── Parameters ──
        't1': t1, 't2': t2,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'model_name':                 'CNN-DA-EMD-OLSB',
        # ── Recovery side info overhead ──
        'recovery_side_info_overhead_%': round(len(side_info) * 8 / actual_total_bits * 100, 2) if actual_total_bits > 0 else 0.0,
        'total_digits_embedded':      total_emd_digits_used,
        # ── Legacy compatibility keys ──
        'total_bits_embedded':        actual_total_bits,
        'raw_payload_bits':           raw_secret_bits,
        'raw_payload_bytes':          len(secret_data),
        'payload_bytes':              len(payload_bytes),
        'internal_bits_embedded':     actual_total_bits,
        'location_map_bytes':         len(side_info),
        'location_map_overhead_%':    round(len(side_info) * 8 / actual_total_bits * 100, 2) if actual_total_bits > 0 else 0.0,
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

    Extraction process:
      1. Read bootstrap header from reserved region (Blue LSB of first 512 pixels)
      2. Parse gamma, t1, t2, cipher_len, recovery_side_info_len from header
      3. Reconstruct gamma-dependent capacity maps using recovered parameters
      4. Extract payload body from non-bootstrap EMD+OLSB positions
      5. Decrypt and decompress secret data
      6. Restore exact cover via recovery side information

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
        stego_rgb = stego_input[0]
    elif isinstance(stego_input, np.ndarray) and stego_input.ndim == 3:
        stego_rgb = stego_input
    else:
        raise ValueError(
            "CNN-DA-EMD-OLSB extraction requires a single stego image (H×W×3 uint8) "
            "or a legacy (stego1, stego2) tuple."
        )

    h, w, c = stego_rgb.shape

    # ── Step 1: Read bootstrap header (gamma-independent) ─────────────────
    n_bootstrap = BOOTSTRAP_N_PIXELS
    if h * w < n_bootstrap:
        raise ValueError(
            f"CNN-DA-EMD-OLSB: Image too small for bootstrap extraction "
            f"({h}×{w} = {h*w} pixels < {n_bootstrap} required)."
        )

    bootstrap_yx = _get_bootstrap_positions(h, w)

    # Extract header bits from Blue channel LSB of bootstrap pixels
    header_bits = []
    for i in range(n_bootstrap):
        y, x = int(bootstrap_yx[i, 0]), int(bootstrap_yx[i, 1])
        header_bits.append(int(stego_rgb[y, x, 2]) & 1)

    header_bytes = bits_to_bytes(
        np.array(header_bits[:HEADER_SIZE_BYTES * 8], dtype=np.uint8)
    )

    # ── Step 2: Validate and parse header ─────────────────────────────────
    if not _is_valid_header(header_bytes, h, w, c):
        raise ValueError(
            "CNN-DA-EMD-OLSB: Invalid bootstrap header. "
            "This image may not contain embedded data or was created with "
            "an incompatible version."
        )

    cipher_len, t1_h, t2_h, gamma_h, side_info_len = _parse_header(header_bytes, h, w, c)

    # Use parameters from header (deterministic recovery — no guessing)
    t1, t2, gamma = t1_h, t2_h, gamma_h

    # ── Step 3: Compute gamma-dependent capacity maps ─────────────────────
    upper_stego = (stego_rgb & 0xF8).astype(np.uint8)
    cls_r, cls_g, cls_b = _get_cap_maps(upper_stego, alpha, beta, gamma, t1, t2, model)
    cap_info = compute_capacity(cls_r, cls_g, cls_b, upper_stego)

    # ── Step 4: Get non-bootstrap EMD and OLSB positions ──────────────────
    bootstrap_mask = np.zeros((h, w), dtype=bool)
    bootstrap_mask[bootstrap_yx[:, 0], bootstrap_yx[:, 1]] = True

    all_emd = np.argwhere(cap_info['emd_mask'])
    all_olsb = np.argwhere(cap_info['olsb_mask'])

    if len(all_emd) > 0:
        emd_positions = all_emd[~bootstrap_mask[all_emd[:, 0], all_emd[:, 1]]]
    else:
        emd_positions = np.empty((0, 2), dtype=np.intp)

    if len(all_olsb) > 0:
        olsb_positions = all_olsb[~bootstrap_mask[all_olsb[:, 0], all_olsb[:, 1]]]
    else:
        olsb_positions = np.empty((0, 2), dtype=np.intp)

    # ── Step 5: Extract body from non-bootstrap region ────────────────────
    body_bits_needed = (side_info_len + cipher_len) * 8
    body_bits = _extract_bits_from_positions(
        stego_rgb, emd_positions, olsb_positions, body_bits_needed
    )
    body_bytes = bits_to_bytes(
        np.array(body_bits[:body_bits_needed], dtype=np.uint8)
    )

    # ── Step 6: Reconstruct full payload and parse ────────────────────────
    full_payload = header_bytes[:HEADER_SIZE_BYTES] + body_bytes
    secret_data, metadata = parse_payload(full_payload, password)

    # ── Step 7: Recover cover via recovery side information ───────────────
    side_info_data = metadata.get('location_map_data')
    if side_info_data is not None and len(side_info_data) > 0:
        recovered_cover = _apply_recovery_side_info(
            stego_rgb, side_info_data, emd_positions, olsb_positions,
            bootstrap_yx
        )
    else:
        recovered_cover = stego_rgb.copy()

    metadata['algorithm'] = 'CNN-DA-EMD-OLSB'
    metadata['alpha'] = alpha
    metadata['beta'] = beta
    metadata['gamma'] = gamma
    metadata['single_stego'] = True
    metadata['dual_images'] = False
    return secret_data, recovered_cover, metadata


def _extract_bits_from_positions(
    stego_rgb: np.ndarray,
    emd_positions: np.ndarray,
    olsb_positions: np.ndarray,
    total_bits_needed: int
) -> list:
    """
    Extract payload bits from stego image using explicit pre-filtered position
    arrays for EMD (R-G pairs) and OLSB (Blue channel).
    Returns list of int bits.
    """
    extracted_bits = []

    # --- EMD extraction from R-G pairs ---
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

    # --- OLSB extraction from Blue channel ---
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
