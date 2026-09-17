"""Self-contained non-reversible steganography reference baseline.

This module intentionally does not claim carrier recovery or an unbundled
super-resolution checkpoint. It provides a reproducible two-LSB payload-hiding
baseline with an in-band AES-GCM header and a password-derived Lorenz
permutation. The output can be extracted after being saved and reloaded.
"""

import struct
import numpy as np
from typing import Tuple, Dict, Any

from core.encryption import encrypt_payload, decrypt_payload
from utils.payload_utils import (
    bytes_to_bits,
    bits_to_bytes,
    apply_3d_chaotic_permute,
    apply_3d_chaotic_inverse,
    generate_3d_lorenz_sequence,
)


_HEADER_MAGIC = b'SR41'
_HEADER_FORMAT = '!4sI16s12s'
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


def hashlib_key(key_str: str) -> int:
    """Derive a deterministic permutation seed from a password."""
    import hashlib
    return int(hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:8], 16)


class SRDNNStego:
    """Portable encrypted LSB steganography baseline (not an RDH method)."""

    def embed(self, cover_rgb: np.ndarray, secret_data: bytes, password: str = "ECC_Key_2026") -> Tuple[np.ndarray, Dict[str, Any]]:
        if cover_rgb.ndim != 3 or cover_rgb.shape[2] != 3 or cover_rgb.dtype != np.uint8:
            raise ValueError("SRDNN-Stego baseline requires an HxWx3 uint8 RGB cover image.")

        salt, nonce, ciphertext = encrypt_payload(secret_data, password)
        header = struct.pack(_HEADER_FORMAT, _HEADER_MAGIC, len(ciphertext), salt, nonce)
        header_bits = bytes_to_bits(header)
        encrypted_bits = bytes_to_bits(ciphertext)
        permuted_bits, _ = apply_3d_chaotic_permute(encrypted_bits, key_seed=hashlib_key(password))

        flat_stego = cover_rgb.reshape(-1).copy()
        body_slots = len(flat_stego) - len(header_bits)
        needed_slots = (len(permuted_bits) + 1) // 2
        if body_slots < needed_slots:
            raise ValueError(
                f"SRDNN-Stego payload needs {needed_slots} two-bit slots after its header, "
                f"but the cover provides {max(0, body_slots)}."
            )

        for index, bit in enumerate(header_bits):
            flat_stego[index] = (flat_stego[index] & 0xFE) | int(bit)

        start = len(header_bits)
        for slot in range(needed_slots):
            chunk = permuted_bits[slot * 2:min((slot + 1) * 2, len(permuted_bits))]
            value = 0
            for bit in chunk:
                value = (value << 1) | int(bit)
            if len(chunk) == 1:
                value <<= 1
            flat_stego[start + slot] = (flat_stego[start + slot] & 0xFC) | value

        total_bits = len(header_bits) + len(permuted_bits)
        return flat_stego.reshape(cover_rgb.shape), {
            'total_bits_embedded': int(total_bits),
            'payload_ciphertext_bits': int(len(permuted_bits)),
            'bpp': float(total_bits / (cover_rgb.shape[0] * cover_rgb.shape[1])),
            'self_contained_extraction': True,
            'reversible': False,
            'model_name': 'SRDNN-Stego',
        }

    def extract(self, stego_rgb: np.ndarray, password: str = "ECC_Key_2026") -> bytes:
        if stego_rgb.ndim != 3 or stego_rgb.shape[2] != 3:
            raise ValueError("SRDNN-Stego extraction requires an HxWx3 RGB stego image.")
        flat_stego = stego_rgb.reshape(-1)
        header_bit_count = _HEADER_SIZE * 8
        if len(flat_stego) < header_bit_count:
            raise ValueError("Stego image is too small for the SRDNN-Stego header.")

        header_bits = np.array([int(value) & 1 for value in flat_stego[:header_bit_count]], dtype=np.uint8)
        magic, cipher_len, salt, nonce = struct.unpack(_HEADER_FORMAT, bits_to_bytes(header_bits))
        if magic != _HEADER_MAGIC:
            raise ValueError("Invalid SRDNN-Stego header.")
        if cipher_len < 16:
            raise ValueError("Invalid SRDNN-Stego ciphertext length.")

        total_bits = cipher_len * 8
        needed_slots = (total_bits + 1) // 2
        if header_bit_count + needed_slots > len(flat_stego):
            raise ValueError("SRDNN-Stego image ended before the declared ciphertext.")

        permuted = []
        for slot in range(needed_slots):
            bits_needed = min(2, total_bits - len(permuted))
            value = int(flat_stego[header_bit_count + slot]) & 0x03
            if bits_needed == 1:
                permuted.append((value >> 1) & 1)
            else:
                permuted.extend([(value >> 1) & 1, value & 1])

        perm_idx = generate_3d_lorenz_sequence(total_bits, x0=0.1 + (hashlib_key(password) % 100) / 1000.0)
        ciphertext_bits = apply_3d_chaotic_inverse(np.asarray(permuted, dtype=np.uint8), perm_idx)
        ciphertext = bits_to_bytes(ciphertext_bits)[:cipher_len]
        return decrypt_payload(ciphertext, password, salt, nonce)
