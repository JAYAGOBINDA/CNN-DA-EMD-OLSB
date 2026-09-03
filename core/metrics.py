"""
Steganographic Performance & Visual Quality Metrics Module.
Calculates 10 quantitative evaluation metrics:
1. PSNR (dB)
2. SSIM
3. MSE
4. wPSNR (dB) [Weighted PSNR]
5. BPP (Bits Per Pixel)
6. Max_Capacity_Bits
7. BER (Bit Error Rate)
8. Payload Recovery Accuracy (%)
9. Embedding Time (s)
10. Extraction Time (s)
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from typing import Dict, Any, Union, Tuple


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes Mean Squared Error (MSE) between cover and stego images.
    """
    err = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    return float(err)


def calculate_psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in dB using scikit-image.
    """
    mse = compute_mse(cover, stego)
    if mse == 0:
        return float('inf')
    return float(compute_psnr(cover, stego, data_range=255))


def calculate_ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Computes Structural Similarity Index (SSIM) across color channels.
    """
    if cover.ndim == 3:
        return float(compute_ssim(cover, stego, channel_axis=2, data_range=255))
    return float(compute_ssim(cover, stego, data_range=255))


def calculate_wpsnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Computes Weighted PSNR (wPSNR) incorporating Human Visual System (HVS) texture masking.
    Texture regions tolerate higher noise without visual degradation.
    """
    gray = cv2.cvtColor(cover, cv2.COLOR_RGB2GRAY) if cover.ndim == 3 else cover
    mean = cv2.blur(gray.astype(np.float32), (3, 3))
    sqr_mean = cv2.blur(gray.astype(np.float32)**2, (3, 3))
    var = np.maximum(0.0, sqr_mean - (mean ** 2))
    norm_var = var / (np.max(var) + 1e-8)

    diff_sq = (cover.astype(np.float64) - stego.astype(np.float64)) ** 2
    if diff_sq.ndim == 3:
        norm_var = np.expand_dims(norm_var, axis=2)
    
    # Weight error by local variance (textured areas penalize perception less)
    wmse = np.mean(diff_sq / (1.0 + 2.0 * norm_var))
    if wmse < 1e-8:
        return float('inf')
    return float(10.0 * np.log10((255.0 ** 2) / wmse))


def compute_ber(original_bits: np.ndarray, extracted_bits: np.ndarray) -> float:
    """
    Computes Bit Error Rate (BER) between original payload bits and extracted bits.
    BER = (Number of incorrect bits) / (Total number of bits)
    """
    min_len = min(len(original_bits), len(extracted_bits))
    if min_len == 0:
        return 1.0
    errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
    return float(errors / min_len)


def compute_bpp(total_embedded_bits: int, image_shape: Tuple[int, ...]) -> float:
    """
    Computes Bits Per Pixel (BPP).
    BPP = Total Embedded Bits / (Height * Width)
    """
    h, w = image_shape[:2]
    return float(total_embedded_bits / (h * w))


def evaluate_quality(
    cover: np.ndarray,
    stego: np.ndarray,
    total_embedded_bits: int,
    original_payload: bytes = None,
    extracted_payload: bytes = None,
    embed_time: float = 0.0,
    extract_time: float = 0.0,
    max_capacity_bits: int = None
) -> Dict[str, Any]:
    """
    Comprehensive 10-parameter steganography evaluation pipeline.
    """
    h, w, c = cover.shape
    mse = compute_mse(cover, stego)
    psnr = calculate_psnr(cover, stego)
    ssim_val = calculate_ssim(cover, stego)
    wpsnr = calculate_wpsnr(cover, stego)
    bpp = compute_bpp(total_embedded_bits, cover.shape)

    if max_capacity_bits is None:
        max_capacity_bits = h * w * c * 3  # Maximum possible theoretical bits

    ber = 0.0
    payload_recovery_acc = 100.0

    if original_payload is not None and extracted_payload is not None:
        if original_payload == extracted_payload:
            ber = 0.0
            payload_recovery_acc = 100.0
        else:
            orig_arr = np.frombuffer(original_payload, dtype=np.uint8)
            extr_arr = np.frombuffer(extracted_payload, dtype=np.uint8)
            min_l = min(len(orig_arr), len(extr_arr))
            if min_l > 0:
                match_count = np.sum(orig_arr[:min_l] == extr_arr[:min_l])
                payload_recovery_acc = float((match_count / max(len(orig_arr), len(extr_arr))) * 100.0)
                
                # Bit level BER
                orig_bits = np.unpackbits(orig_arr)
                extr_bits = np.unpackbits(extr_arr)
                ber = compute_ber(orig_bits, extr_bits)
            else:
                ber = 1.0
                payload_recovery_acc = 0.0

    return {
        'PSNR_dB': round(psnr, 2),
        'SSIM': round(ssim_val, 4),
        'MSE': round(mse, 4),
        'wPSNR_dB': round(wpsnr, 2),
        'BPP': round(bpp, 4),
        'Max_Capacity_Bits': int(max_capacity_bits),
        'BER': round(ber, 4),
        'Recovery_Acc_%': round(payload_recovery_acc, 2),
        'Embed_Time_s': round(embed_time, 4),
        'Extract_Time_s': round(extract_time, 4)
    }
