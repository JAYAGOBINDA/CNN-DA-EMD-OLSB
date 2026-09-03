"""
Payload Packaging and Header Serialization Module.
Handles zlib compression, AES payload structure, bitstream conversion, and binary header packing/unpacking.
"""

import struct
import zlib
import binascii
import numpy as np
from typing import Tuple, Dict, Any

HEADER_MAGIC = b'CHAL'
HEADER_SIZE_BYTES = 64


def prepare_payload(data: bytes, password: str, t1: float = 0.33, t2: float = 0.66, payload_type: int = 0) -> bytes:
    """
    Compresses data (zlib), encrypts with AES-256-GCM, and prepends 64-byte deterministic header.
    """
    from core.encryption import encrypt_payload

    # Step 1: Optional Compression
    compressed = zlib.compress(data, level=6)
    is_compressed = 1 if len(compressed) < len(data) else 0
    payload_to_encrypt = compressed if is_compressed else data

    # Step 2: AES-256-GCM Encryption
    salt, nonce, ciphertext = encrypt_payload(payload_to_encrypt, password)

    # Step 3: Compute CRC32 of raw payload data
    crc = binascii.crc32(data) & 0xffffffff

    # Step 4: Construct 64-Byte Header
    # Format:
    # Magic (4s), is_compressed (B), payload_type (B), reserved (2s), payload_len (I),
    # salt (16s), nonce (12s), t1 (f), t2 (f), crc (I), padding (12s)
    header = struct.pack(
        '!4sBB2sI16s12sffI12s',
        HEADER_MAGIC,
        is_compressed,
        payload_type,
        b'\x00\x00',
        len(ciphertext),
        salt,
        nonce,
        float(t1),
        float(t2),
        crc,
        b'\x00' * 12
    )

    assert len(header) == HEADER_SIZE_BYTES, f"Header size mismatch: {len(header)} vs {HEADER_SIZE_BYTES}"
    return header + ciphertext


def parse_payload(full_payload: bytes, password: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    Parses full bitstream payload, extracts header, decrypts ciphertext, and decompresses.
    """
    from core.encryption import decrypt_payload

    if len(full_payload) < HEADER_SIZE_BYTES:
        raise ValueError("Bitstream size is smaller than header size.")

    header_bytes = full_payload[:HEADER_SIZE_BYTES]
    ciphertext = full_payload[HEADER_SIZE_BYTES:]

    magic, is_compressed, payload_type, _, cipher_len, salt, nonce, t1, t2, crc, _ = struct.unpack(
        '!4sBB2sI16s12sffI12s', header_bytes
    )

    if magic != HEADER_MAGIC:
        raise ValueError(f"Invalid magic header signature: {magic}. Expected {HEADER_MAGIC}.")

    ciphertext = ciphertext[:cipher_len]

    # Decrypt
    decrypted = decrypt_payload(ciphertext, password, salt, nonce)

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
        'crc_match': crc_match,
        'data_size': len(data)
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
