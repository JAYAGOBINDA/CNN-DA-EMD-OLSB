"""
Payload Packaging and Header Serialization Module.
Handles zlib compression, AES payload structure, bitstream conversion, and binary header packing/unpacking.

Header Format (64 bytes, big-endian):
  Magic           4s   b'CHAL'
  is_compressed   B    0 or 1
  payload_type    B    0=binary, 1=text, 2=image
  reserved        2s   b'\x00\x00'
  cipher_len      I    length of ciphertext in bytes
  salt            16s  AES-256-GCM salt
  nonce           12s  AES-256-GCM nonce
  t1              f    distortion threshold 1
  t2              f    distortion threshold 2
  gamma           f    CNN blending weight
  crc             I    CRC32 of raw plaintext
  locmap_size     I    compressed recovery side information size in bytes (0 = no info)
  padding         4s   reserved
"""

import struct
import zlib
import binascii
import numpy as np
from typing import Tuple, Dict, Any, Optional

HEADER_MAGIC = b'CHAL'
HEADER_SIZE_BYTES = 64


def prepare_payload(
    data: bytes,
    password: str,
    t1: float = 0.33,
    t2: float = 0.66,
    payload_type: int = 0,
    gamma: float = 0.6,
    location_map_data: Optional[bytes] = None
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

    # Step 2: AES-256-GCM Encryption (with recovery side info as AAD for integrity)
    salt, nonce, ciphertext = encrypt_payload(
        payload_to_encrypt, password, associated_data=location_map_data
    )

    # Step 3: Compute CRC32 of raw payload data
    crc = binascii.crc32(data) & 0xffffffff

    # Step 4: Location map handling
    locmap_size = len(location_map_data) if location_map_data else 0

    # Step 5: Construct 64-Byte Header
    # Format: Magic(4s) is_compressed(B) payload_type(B) reserved(2s)
    #         cipher_len(I) salt(16s) nonce(12s) t1(f) t2(f) gamma(f)
    #         crc(I) locmap_size(I) padding(4s)
    header = struct.pack(
        '!4sBB2sI16s12sfffII4s',
        HEADER_MAGIC,
        is_compressed,
        payload_type,
        b'\x00\x00',
        len(ciphertext),
        salt,
        nonce,
        float(t1),
        float(t2),
        float(gamma),
        crc,
        locmap_size,
        b'\x00' * 4
    )

    assert len(header) == HEADER_SIZE_BYTES, f"Header size mismatch: {len(header)} vs {HEADER_SIZE_BYTES}"

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
    try:
        # New header format with locmap_size field
        magic, is_compressed, payload_type, _, cipher_len, salt, nonce, t1, t2, gamma_val, crc, locmap_size, _ = struct.unpack(
            '!4sBB2sI16s12sfffII4s', header_bytes
        )
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

    # Extract location map data (if present)
    location_map_data = None
    if locmap_size > 0:
        location_map_data = remainder[:locmap_size]
        remainder = remainder[locmap_size:]

    ciphertext = remainder[:cipher_len]

    # Decrypt (with recovery side info as AAD for integrity verification)
    decrypted = decrypt_payload(
        ciphertext, password, salt, nonce, associated_data=location_map_data
    )

    # Decompress if needed
    if is_compressed == 1:
        data = zlib.decompress(decrypted)
    else:
        data = decrypted

    # CRC32 verification
    calc_crc = binascii.crc32(data) & 0xffffffff
    crc_match = (calc_crc == crc)

    metadata = {
        'payload_type': payload_type,
        'is_compressed': bool(is_compressed),
        't1': t1,
        't2': t2,
        'gamma': gamma_val,
        'crc_match': crc_match,
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
