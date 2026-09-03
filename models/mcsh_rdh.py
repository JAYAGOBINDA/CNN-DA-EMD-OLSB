"""
Model 2: MCSH-RDH (Multi-Channel Synchronized Histogram + Adaptive Capacity Allocation)
Reversible Data Hiding (RDH) Baseline from Literature Paper 2.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any
from utils.payload_utils import bytes_to_bits, bits_to_bytes


class MCSHRDH:
    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w, c = cover_rgb.shape
        payload_bits = bytes_to_bits(secret_bytes)
        total_bits = len(payload_bits)

        R = cover_rgb[:, :, 0].astype(np.float32)
        G = cover_rgb[:, :, 1].astype(np.float32)
        B = cover_rgb[:, :, 2].astype(np.float32)

        pred_R = cv2.blur(R, (3, 3))
        pred_G = cv2.blur(G, (3, 3))
        pred_B = cv2.blur(B, (3, 3))

        var_R = np.var(R - pred_R) + 1e-5
        var_G = np.var(G - pred_G) + 1e-5
        var_B = np.var(B - pred_B) + 1e-5
        total_var = var_R + var_G + var_B

        bits_R = int(total_bits * (var_R / total_var))
        bits_G = int(total_bits * (var_G / total_var))
        bits_B = total_bits - (bits_R + bits_G)

        channel_payloads = [
            payload_bits[:bits_R],
            payload_bits[bits_R : bits_R + bits_G],
            payload_bits[bits_R + bits_G : total_bits]
        ]

        stego_rgb = cover_rgb.copy()
        channel_orig_lsbs = [[], [], []]
        embedded_counts = [0, 0, 0]

        for ch in range(3):
            ch_bits = channel_payloads[ch]
            if len(ch_bits) == 0:
                continue

            flat_cover = cover_rgb[:, :, ch].reshape(-1)
            flat_stego = flat_cover.copy()

            bit_idx = 0
            for i in range(len(flat_cover)):
                if bit_idx >= len(ch_bits):
                    break

                channel_orig_lsbs[ch].append(int(flat_cover[i] & 1))
                b = int(ch_bits[bit_idx])
                flat_stego[i] = (flat_cover[i] & 0xFE) | b
                bit_idx += 1

            stego_rgb[:, :, ch] = flat_stego.reshape(h, w)
            embedded_counts[ch] = bit_idx

        stats = {
            'total_bits_embedded': sum(embedded_counts),
            'bpp': sum(embedded_counts) / (h * w * 3),
            'channel_bits_R': embedded_counts[0],
            'channel_bits_G': embedded_counts[1],
            'channel_bits_B': embedded_counts[2],
            'orig_lsbs_R': np.array(channel_orig_lsbs[0], dtype=np.uint8),
            'orig_lsbs_G': np.array(channel_orig_lsbs[1], dtype=np.uint8),
            'orig_lsbs_B': np.array(channel_orig_lsbs[2], dtype=np.uint8),
            'model_name': 'MCSH-RDH'
        }

        return stego_rgb, stats

    def extract(self, stego_rgb: np.ndarray, stats: Dict[str, Any]) -> Tuple[bytes, np.ndarray]:
        h, w, c = stego_rgb.shape
        recovered_rgb = stego_rgb.copy()
        extracted_channel_bits = [[], [], []]

        counts = [stats['channel_bits_R'], stats['channel_bits_G'], stats['channel_bits_B']]
        orig_lsbs = [stats['orig_lsbs_R'], stats['orig_lsbs_G'], stats['orig_lsbs_B']]

        for ch in range(3):
            target_count = counts[ch]
            if target_count == 0:
                continue

            flat_stego = stego_rgb[:, :, ch].reshape(-1)
            flat_rec = flat_stego.copy()

            for i in range(target_count):
                b = flat_stego[i] & 1
                extracted_channel_bits[ch].append(b)

                # Recover original carrier pixel
                flat_rec[i] = (flat_stego[i] & 0xFE) | orig_lsbs[ch][i]

            recovered_rgb[:, :, ch] = flat_rec.reshape(h, w)

        full_bits = np.array(extracted_channel_bits[0] + extracted_channel_bits[1] + extracted_channel_bits[2], dtype=np.uint8)
        extracted_bytes = bits_to_bytes(full_bits)

        return extracted_bytes, recovered_rgb
