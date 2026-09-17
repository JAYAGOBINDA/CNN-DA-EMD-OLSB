"""
Payload Packaging and Header Serialization Module.
Handles zlib compression, AES payload structure, bitstream conversion, and binary header packing/unpacking.

Header Format (64 bytes, big-endian):
  Magic           4s   b'CHAL'
  is_compressed   B    0 or 1
  payload_type    B    0=binary, 1=text, 2=image
  format_marker   2s   b'V1' (authenticated, includes alpha/beta)
  cipher_len      I    length of ciphertext in bytes
  salt            16s  AES-256-GCM salt
  nonce           12s  AES-256-GCM nonce
  t1              f    distortion threshold 1
  t2              f    distortion threshold 2
  gamma           f    CNN blending weight
  crc             I    CRC32 of raw plaintext
  locmap_size     I    compressed recovery side information size in bytes (0 = no info)
  alpha, beta     HH   analytic-map weights, encoded as two uint16 values

All current-format header bytes and the recovery side information are AES-GCM
Additional Authenticated Data. This prevents an attacker from changing routing
parameters, payload metadata, or length fields without invalidating the tag.
"""

import struct
import zlib
import binascii
import os
import numpy as np
from typing import Tuple, Dict, Any, Optional

HEADER_MAGIC = b'CHAL'
HEADER_SIZE_BYTES = 64
HEADER_FORMAT = '!4sBB2sI16s12sfffIIHH'
HEADER_FORMAT_MARKER = b'V1'


def _encode_unit_interval(value: float) -> int:
    """Encode a [0, 1] parameter into an unsigned 16-bit integer."""
    return int(round(min(1.0, max(0.0, float(value))) * 65535.0))


def _decode_unit_interval(value: int) -> float:
    """Decode a unit-interval parameter stored by _encode_unit_interval."""
    return float(value) / 65535.0


def _pack_current_header(
    *,
    is_compressed: int,
    payload_type: int,
    cipher_len: int,
    salt: bytes,
    nonce: bytes,
    t1: float,
    t2: float,
    gamma: float,
    crc: int,
    locmap_size: int,
    alpha: float,
    beta: float,
) -> bytes:
    """Build the versioned, fixed-size authenticated payload header."""
    header = struct.pack(
        HEADER_FORMAT,
        HEADER_MAGIC,
        int(is_compressed),
        int(payload_type),
        HEADER_FORMAT_MARKER,
        int(cipher_len),
        salt,
        nonce,
        float(t1),
        float(t2),
        float(gamma),
        int(crc),
        int(locmap_size),
        _encode_unit_interval(alpha),
        _encode_unit_interval(beta),
    )
    assert len(header) == HEADER_SIZE_BYTES, (
        f"Header size mismatch: {len(header)} vs {HEADER_SIZE_BYTES}"
    )
    return header


def prepare_payload(
    data: bytes,
    password: str,
    t1: float = 0.33,
    t2: float = 0.66,
    payload_type: int = 0,
    gamma: float = 0.6,
    location_map_data: Optional[bytes] = None,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> bytes:
    """
    Compresses data (zlib), encrypts with AES-256-GCM, and prepends 64-byte
    deterministic header.

    If location_map_data is provided, it is prepended to the ciphertext
    so that the extractor can recover original pixel values for exact
    cover recovery from a single stego image.
    """
    from core.encryption import encrypt_payload

    # Step 1: Optional Compression
    compressed = zlib.compress(data, level=6)
    is_compressed = 1 if len(compressed) < len(data) else 0
    payload_to_encrypt = compressed if is_compressed else data

    # Step 2: Prepare the complete header before encryption. AES-GCM appends a
    # fixed 16-byte authentication tag, so its ciphertext length is known.
    crc = binascii.crc32(data) & 0xffffffff
    locmap_size = len(location_map_data) if location_map_data else 0
    salt = os.urandom(16)
    nonce = os.urandom(12)
    header = _pack_current_header(
        is_compressed=is_compressed,
        payload_type=payload_type,
        cipher_len=len(payload_to_encrypt) + 16,
        salt=salt,
        nonce=nonce,
        t1=t1,
        t2=t2,
        gamma=gamma,
        crc=crc,
        locmap_size=locmap_size,
        alpha=alpha,
        beta=beta,
    )

    # Step 3: Authenticate every header field and recovery byte, while keeping
    # only the encrypted user payload confidential.
    aad = header + (location_map_data or b'')
    salt_out, nonce_out, ciphertext = encrypt_payload(
        payload_to_encrypt,
        password,
        associated_data=aad,
        salt=salt,
        nonce=nonce,
    )
    assert salt_out == salt and nonce_out == nonce
    assert len(ciphertext) == len(payload_to_encrypt) + 16

    # Assemble: header + location_map_data + ciphertext
    if location_map_data:
        return header + location_map_data + ciphertext
    return header + ciphertext


def parse_payload(full_payload: bytes, password: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    Parses full bitstream payload, extracts header, decrypts ciphertext, and decompresses.
    Also extracts the location map if present (locmap_size > 0).
    """
    from core.encryption import decrypt_payload

    if len(full_payload) < HEADER_SIZE_BYTES:
        raise ValueError("Bitstream size is smaller than header size.")

    header_bytes = full_payload[:HEADER_SIZE_BYTES]
    remainder = full_payload[HEADER_SIZE_BYTES:]

    gamma_val = 0.6
    locmap_size = 0
    alpha_val = beta_val = 0.5
    header_authenticated = False
    try:
        (
            magic, is_compressed, payload_type, marker, cipher_len, salt, nonce,
            t1, t2, gamma_val, crc, locmap_size, alpha_q, beta_q,
        ) = struct.unpack(
            HEADER_FORMAT, header_bytes
        )
        if marker == HEADER_FORMAT_MARKER:
            alpha_val = _decode_unit_interval(alpha_q)
            beta_val = _decode_unit_interval(beta_q)
            header_authenticated = True
    except Exception:
        try:
            # Legacy format with 8-byte padding (no locmap)
            magic, is_compressed, payload_type, _, cipher_len, salt, nonce, t1, t2, gamma_val, crc, _ = struct.unpack(
                '!4sBB2sI16s12sfffI8s', header_bytes
            )
        except Exception:
            magic, is_compressed, payload_type, _, cipher_len, salt, nonce, t1, t2, crc, _ = struct.unpack(
                '!4sBB2sI16s12sffI12s', header_bytes
            )

    if magic != HEADER_MAGIC:
        raise ValueError(f"Invalid magic header signature: {magic}. Expected {HEADER_MAGIC}.")
    if is_compressed not in (0, 1):
        raise ValueError("Invalid compression flag in payload header.")
    if cipher_len < 16 or locmap_size < 0 or locmap_size > len(remainder):
        raise ValueError("Invalid payload lengths in header.")

    # Extract location map data (if present)
    location_map_data = None
    if locmap_size > 0:
        location_map_data = remainder[:locmap_size]
        remainder = remainder[locmap_size:]

    ciphertext = remainder[:cipher_len]
    if len(ciphertext) != cipher_len:
        raise ValueError("Stego payload ended before the authenticated ciphertext.")

    # Legacy stego files authenticated only recovery data. Current files bind
    # their full header as AAD as well.
    aad = (header_bytes if header_authenticated else b'') + (location_map_data or b'')
    decrypted = decrypt_payload(
        ciphertext, password, salt, nonce, associated_data=aad
    )

    # Decompress if needed
    if is_compressed == 1:
        data = zlib.decompress(decrypted)
    else:
        data = decrypted

    # CRC32 verification
    calc_crc = binascii.crc32(data) & 0xffffffff
    crc_match = (calc_crc == crc)
    if not crc_match:
        raise ValueError("Payload CRC verification failed.")

    metadata = {
        'payload_type': payload_type,
        'is_compressed': bool(is_compressed),
        't1': t1,
        't2': t2,
        'gamma': gamma_val,
        'alpha': alpha_val,
        'beta': beta_val,
        'crc_match': crc_match,
        'header_authenticated': header_authenticated,
        'data_size': len(data),
        'location_map_data': location_map_data,
        'location_map_size': locmap_size,
    }

    return data, metadata


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Converts bytes object to 1D numpy array of uint8 bits (0 or 1).
    """
    array = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(array)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Converts 1D numpy array of uint8 bits (0 or 1) to bytes object.
    """
    # Ensure length is multiple of 8
    pad = (8 - len(bits) % 8) % 8
    if pad > 0:
        bits = np.pad(bits, (0, pad), mode='constant', constant_values=0)
    packed = np.packbits(bits)
    return packed.tobytes()
