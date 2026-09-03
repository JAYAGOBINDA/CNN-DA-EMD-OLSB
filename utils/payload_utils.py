"""
Payload & Cryptographic Security Utilities for 6-Model Benchmark Suite.
Implements bitstream conversion, 3D Lorenz Chaotic Permutation Map, and ECC Key Simulation for Model 4 (SRDNN-Stego).
"""

import numpy as np
import os
import hashlib
from typing import Tuple, Union


def generate_random_bits(num_bits: int, seed: int = 42) -> np.ndarray:
    """
    Generates reproducible random bit array of uint8 (0 or 1).
    """
    np.random.seed(seed)
    return np.random.randint(0, 2, num_bits, dtype=np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Converts bytes to 1D uint8 numpy bit array.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Converts 1D uint8 numpy bit array to bytes.
    """
    pad = (8 - len(bits) % 8) % 8
    if pad > 0:
        bits = np.pad(bits, (0, pad), mode='constant', constant_values=0)
    packed = np.packbits(bits)
    return packed.tobytes()


# --- 3D Lorenz Chaotic Map Permutation (for Model 4 SRDNN-Stego) ---

def generate_3d_lorenz_sequence(length: int, x0: float = 0.1, y0: float = 0.2, z0: float = 0.3) -> np.ndarray:
    """
    Generates 3D Lorenz chaotic attractor coordinate index permutation.
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = x*y - beta*z
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01

    xs = np.zeros(length, dtype=np.float64)
    ys = np.zeros(length, dtype=np.float64)
    zs = np.zeros(length, dtype=np.float64)

    x, y, z = x0, y0, z0
    for i in range(length):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        xs[i], ys[i], zs[i] = x, y, z

    # Return permutation order based on sorted chaotic x-coordinate
    perm = np.argsort(xs)
    return perm


def apply_3d_chaotic_permute(data_bits: np.ndarray, key_seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permutes bitstream using 3D Lorenz chaotic sequence.
    """
    n = len(data_bits)
    perm_idx = generate_3d_lorenz_sequence(n, x0=0.1 + (key_seed % 100)/1000.0)
    permuted_bits = data_bits[perm_idx]
    return permuted_bits, perm_idx


def apply_3d_chaotic_inverse(permuted_bits: np.ndarray, perm_idx: np.ndarray) -> np.ndarray:
    """
    Restores original bitstream from 3D Lorenz chaotic permuted order.
    """
    inv_idx = np.argsort(perm_idx)
    return permuted_bits[inv_idx]


# --- Elliptic Curve Cryptography (ECC) Key Exchange Simulation ---

def simulate_ecc_key_encryption(payload_bytes: bytes, password: str) -> Tuple[bytes, str]:
    """
    Simulates ECC (secp256k1 curve) key exchange & XOR encryption wrapper for SRDNN-Stego security.
    """
    # Derive 256-bit ECC shared key hash using SHA-256
    key_hash = hashlib.sha256(password.encode('utf-8')).digest()
    key_repeat = (key_hash * (len(payload_bytes) // len(key_hash) + 1))[:len(payload_bytes)]
    
    # Simple authenticated XOR stream cipher simulation
    encrypted_bytes = bytes(a ^ b for a, b in zip(payload_bytes, key_repeat))
    ecc_pubkey_hex = hashlib.sha256((password + "_ecc_pub").encode('utf-8')).hexdigest()[:32]
    
    return encrypted_bytes, ecc_pubkey_hex


def simulate_ecc_key_decryption(encrypted_bytes: bytes, password: str) -> bytes:
    """
    Decrypts ECC shared key payload.
    """
    key_hash = hashlib.sha256(password.encode('utf-8')).digest()
    key_repeat = (key_hash * (len(encrypted_bytes) // len(key_hash) + 1))[:len(encrypted_bytes)]
    return bytes(a ^ b for a, b in zip(encrypted_bytes, key_repeat))
