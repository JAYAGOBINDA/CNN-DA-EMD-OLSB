"""
Unified Benchmarking Metrics Engine with N/A Handling & Carrier Image Recovery Calculation.
Computes PSNR, SSIM, MSE, wPSNR, BPP, Max Capacity, BER, Payload Recovery %, Carrier Image Recovery %, Embed/Extract Times.
"""

import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from typing import Dict, Any, Union, Tuple


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE)."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = compute_mse(cover, stego)
    if mse == 0:
        return float('inf')
    return float(compute_psnr(cover, stego, data_range=255))


def calculate_ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM)."""
    if cover.ndim == 3:
        return float(compute_ssim(cover, stego, channel_axis=2, data_range=255))
    return float(compute_ssim(cover, stego, data_range=255))


def calculate_wpsnr(cover: np.ndarray, stego: np.ndarray) -> float:
    """Computes Weighted PSNR (wPSNR) using local variance texture mask."""
    gray = cv2.cvtColor(cover, cv2.COLOR_RGB2GRAY) if cover.ndim == 3 else cover
    mean = cv2.blur(gray.astype(np.float32), (3, 3))
    sqr_mean = cv2.blur(gray.astype(np.float32)**2, (3, 3))
    var = np.maximum(0.0, sqr_mean - (mean ** 2))
    norm_var = var / (np.max(var) + 1e-8)

    diff_sq = (cover.astype(np.float64) - stego.astype(np.float64)) ** 2
    if diff_sq.ndim == 3:
        norm_var = np.expand_dims(norm_var, axis=2)
    
    wmse = np.mean(diff_sq / (1.0 + 2.0 * norm_var))
    if wmse < 1e-8:
        return float('inf')
    return float(10.0 * np.log10((255.0 ** 2) / wmse))


def calculate_carrier_recovery_accuracy(cover_original: np.ndarray, cover_recovered: Union[np.ndarray, None]) -> Union[float, str]:
    """
    Computes exact carrier photo reconstruction accuracy for Reversible Data Hiding (RDH) models.
    Returns percentage accuracy or "N/A (Non-RDH Payload Hiding Model)".
    """
    if cover_recovered is None:
        return "N/A (Non-RDH Payload Hiding Model)"
    
    if cover_original.shape != cover_recovered.shape:
        return "N/A (Dimension Mismatch)"

    match = np.sum(cover_original == cover_recovered)
    total_pixels = cover_original.size
    acc = float((match / total_pixels) * 100.0)
    return round(acc, 2)


def evaluate_model_performance(
    cover: np.ndarray,
    stego_output: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    total_embedded_bits: int,
    original_payload: bytes,
    extracted_payload: bytes,
    recovered_cover: Union[np.ndarray, None] = None,
    embed_time: float = 0.0,
    extract_time: float = 0.0,
    is_dual_stego: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive evaluation pipeline for the 6-Model Benchmark Suite.
    """
    # Handle Dual Stego Output (EMD-OLSB RDH)
    if is_dual_stego and isinstance(stego_output, tuple):
        stego1, stego2 = stego_output
        psnr1, psnr2 = calculate_psnr(cover, stego1), calculate_psnr(cover, stego2)
        ssim1, ssim2 = calculate_ssim(cover, stego1), calculate_ssim(cover, stego2)
        mse1, mse2 = compute_mse(cover, stego1), compute_mse(cover, stego2)
        wpsnr1, wpsnr2 = calculate_wpsnr(cover, stego1), calculate_wpsnr(cover, stego2)

        psnr_val = (psnr1 + psnr2) / 2.0
        ssim_val = (ssim1 + ssim2) / 2.0
        mse_val = (mse1 + mse2) / 2.0
        wpsnr_val = (wpsnr1 + wpsnr2) / 2.0
        stego_eval_img = stego1
    else:
        stego_eval_img = stego_output
        psnr_val = calculate_psnr(cover, stego_eval_img)
        ssim_val = calculate_ssim(cover, stego_eval_img)
        mse_val = compute_mse(cover, stego_eval_img)
        wpsnr_val = calculate_wpsnr(cover, stego_eval_img)

    h, w, c = cover.shape
    bpp = total_embedded_bits / (h * w)
    max_cap = h * w * c * 3

    # Bit Error Rate (BER) & Payload Recovery
    ber = 0.0
    payload_rec_acc = 100.0

    if original_payload is not None and extracted_payload is not None:
        if original_payload == extracted_payload:
            ber = 0.0
            payload_rec_acc = 100.0
        else:
            orig_arr = np.frombuffer(original_payload, dtype=np.uint8)
            extr_arr = np.frombuffer(extracted_payload, dtype=np.uint8)
            min_l = min(len(orig_arr), len(extr_arr))
            if min_l > 0:
                match_count = np.sum(orig_arr[:min_l] == extr_arr[:min_l])
                payload_rec_acc = float((match_count / max(len(orig_arr), len(extr_arr))) * 100.0)
                
                orig_bits = np.unpackbits(orig_arr)
                extr_bits = np.unpackbits(extr_arr)
                min_b = min(len(orig_bits), len(extr_bits))
                errors = np.sum(orig_bits[:min_b] != extr_bits[:min_b])
                ber = float(errors / min_b)
            else:
                ber = 1.0
                payload_rec_acc = 0.0

    carrier_acc = calculate_carrier_recovery_accuracy(cover, recovered_cover)

    return {
        'PSNR_dB': round(psnr_val, 2),
        'SSIM': round(ssim_val, 4),
        'MSE': round(mse_val, 4),
        'wPSNR_dB': round(wpsnr_val, 2),
        'BPP': round(bpp, 4),
        'Max_Capacity_Bits': int(max_cap),
        'BER': round(ber, 4),
        'Payload_Recovery_Acc_%': round(payload_rec_acc, 2),
        'Carrier_Recovery_Acc_%': carrier_acc,
        'Embed_Time_s': round(embed_time, 4),
        'Extract_Time_s': round(extract_time, 4)
    }
