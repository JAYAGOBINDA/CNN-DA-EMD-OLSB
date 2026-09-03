"""
Model 4: SRDNN-Stego (Super-Resolution Deep Neural Network Multi-Image Steganography)
High-Density Multi-Image Steganography Model from Literature Paper 4.
"""

import numpy as np
import cv2
import io
import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple, Dict, Any
from utils.payload_utils import apply_3d_chaotic_permute, apply_3d_chaotic_inverse, simulate_ecc_key_encryption, simulate_ecc_key_decryption, bytes_to_bits, bits_to_bytes
from utils.image_utils import load_image_rgb, resize_image


def hashlib_key(key_str: str) -> int:
    """Derives a compact integer key from a password string using MD5."""
    import hashlib
    return int(hashlib.md5(key_str.encode('utf-8')).hexdigest()[:6], 16)


class SRDNNReconstructionNetwork(nn.Module):
    """Super-Resolution Neural Network for high-frequency secret feature enhancement."""
    def __init__(self):
        super(SRDNNReconstructionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        out = self.conv3(h)
        return out


class SRDNNStego:
    """SRDNN Multi-Image Steganography Model Implementation."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.srdnn = SRDNNReconstructionNetwork().to(self.device)
        self.srdnn.eval()

    def embed(self, cover_rgb: np.ndarray, secret_data: bytes, password: str = "ECC_Key_2026") -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w, c = cover_rgb.shape

        # Step 1: Security Component 1 — ECC Encryption Simulation
        ecc_encrypted_bytes, pubkey_hex = simulate_ecc_key_encryption(secret_data, password)
        raw_bits = bytes_to_bits(ecc_encrypted_bytes)

        # Step 2: Security Component 2 — 3D Lorenz Chaotic Permutation Map
        permuted_bits, perm_idx = apply_3d_chaotic_permute(raw_bits, key_seed=int(hashlib_key(password)))
        total_bits = len(permuted_bits)

        # Step 3: SRDNN Feature Pass
        stego_tensor = torch.from_numpy(cover_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            enhanced_tensor = stego_tensor + 0.001 * self.srdnn(stego_tensor)
            enhanced_np = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0)

        stego_enhanced = np.clip(enhanced_np, 0, 255).astype(np.uint8)

        # Step 4: Bitstream Embedding into SRDNN-Enhanced Image
        flat_stego = stego_enhanced.reshape(-1)
        num_chunks = int(np.ceil(total_bits / 2))

        for i in range(num_chunks):
            chunk = permuted_bits[i*2 : min((i+1)*2, total_bits)]
            val = 0
            for b in chunk:
                val = (val << 1) | int(b)
            mask = ~((1 << len(chunk)) - 1) & 0xFF
            flat_stego[i] = (flat_stego[i] & mask) | val

        stego_final = flat_stego.reshape(h, w, c)

        stats = {
            'total_bits_embedded': total_bits,
            'bpp': total_bits / (h * w * 3),
            'ecc_pubkey': pubkey_hex,
            '3d_chaotic_seed': password,
            'perm_idx': perm_idx,
            'model_name': 'SRDNN-Stego'
        }

        return stego_final, stats

    def extract(self, stego_rgb: np.ndarray, total_bits: int, perm_idx: np.ndarray, password: str = "ECC_Key_2026") -> bytes:
        flat_stego = stego_rgb.reshape(-1)
        num_chunks = int(np.ceil(total_bits / 2))

        extracted_bits = []
        for i in range(num_chunks):
            bits_needed = min(2, total_bits - len(extracted_bits))
            val = flat_stego[i] & ((1 << bits_needed) - 1)
            chunk = [(val >> (bits_needed - 1 - b)) & 1 for b in range(bits_needed)]
            extracted_bits.extend(chunk)

        permuted_arr = np.array(extracted_bits[:total_bits], dtype=np.uint8)

        # Step 1: Inverse 3D Lorenz Chaotic Map Permutation
        raw_bits = apply_3d_chaotic_inverse(permuted_arr, perm_idx)
        ecc_encrypted_bytes = bits_to_bytes(raw_bits)

        # Step 2: ECC Decryption
        secret_bytes = simulate_ecc_key_decryption(ecc_encrypted_bytes, password)
        return secret_bytes
