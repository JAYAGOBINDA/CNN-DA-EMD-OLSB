"""
Robustness Attack Testing & Bit Error Analysis Module.
Evaluates stego image survival, PSNR degradation, exact Bit Error Rate (BER),
and bit-level spatial corruption maps under JPEG compression, Gaussian noise,
Salt & Pepper noise, Cropping, and Resizing.

Supports both single-stego and dual-stego (CNN-DA-EMD-OLSB) models.
For dual-stego: attacks are applied to S1 only; S2 remains clean as recovery reference.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Union
from core.metrics import compute_ber, calculate_psnr, calculate_ssim
from core.payload import HEADER_SIZE_BYTES, bits_to_bytes


def apply_jpeg_compression(stego_rgb: np.ndarray, quality: int = 80) -> np.ndarray:
    """
    Simulates lossy JPEG compression at specific quality factor (1-100).
    """
    stego_bgr = cv2.cvtColor(stego_rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, enc_img = cv2.imencode('.jpg', stego_bgr, encode_param)
    dec_bgr = cv2.imdecode(enc_img, 1)
    return cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)


def apply_gaussian_noise(stego_rgb: np.ndarray, mean: float = 0, std: float = 5.0) -> np.ndarray:
    """
    Applies Gaussian additive noise.
    """
    noise = np.random.normal(mean, std, stego_rgb.shape)
    noisy = stego_rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(stego_rgb: np.ndarray, amount: float = 0.005) -> np.ndarray:
    """
    Applies salt and pepper noise.
    """
    noisy = stego_rgb.copy()
    # Salt
    num_salt = np.ceil(amount * stego_rgb.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in stego_rgb.shape]
    noisy[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(amount * stego_rgb.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in stego_rgb.shape]
    noisy[tuple(coords)] = 0
    return noisy


def apply_cropping(stego_rgb: np.ndarray, crop_percent: float = 0.1) -> np.ndarray:
    """
    Crops peripheral outer region by crop_percent and zero-pads back to original dimensions.
    """
    h, w, c = stego_rgb.shape
    crop_h = int(h * crop_percent / 2)
    crop_w = int(w * crop_percent / 2)

    cropped_stego = stego_rgb.copy()
    cropped_stego[:crop_h, :, :] = 0
    cropped_stego[-crop_h:, :, :] = 0
    cropped_stego[:, :crop_w, :] = 0
    cropped_stego[:, -crop_w:, :] = 0
    return cropped_stego


def apply_resizing(stego_rgb: np.ndarray, scale: float = 0.8) -> np.ndarray:
    """
    Rescales image down and back up to original dimension.
    """
    h, w = stego_rgb.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    downscaled = cv2.resize(stego_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_CUBIC)
    return upscaled


def run_attack_suite(
    stego_rgb: np.ndarray,
    jpeg_quality: int = 80,
    gaussian_std: float = 5.0,
    salt_pepper_amount: float = 0.005,
    crop_percent: float = 0.10,
    resize_scale: float = 0.80
) -> Dict[str, np.ndarray]:
    """
    Runs configurable suite of standard image processing attacks.
    """
    return {
        f'JPEG Compression (Q={jpeg_quality})': apply_jpeg_compression(stego_rgb, quality=jpeg_quality),
        f'Gaussian Additive Noise (σ={gaussian_std})': apply_gaussian_noise(stego_rgb, std=gaussian_std),
        f'Salt & Pepper Noise ({salt_pepper_amount*100:.1f}%)': apply_salt_pepper_noise(stego_rgb, amount=salt_pepper_amount),
        f'Peripheral Cropping ({crop_percent*100:.0f}%)': apply_cropping(stego_rgb, crop_percent=crop_percent),
        f'Spatial Rescaling ({resize_scale:.2f}x)': apply_resizing(stego_rgb, scale=resize_scale)
    }


def evaluate_attack_robustness(
    clean_stego: np.ndarray,
    attacked_stego: np.ndarray,
    clean_stego_s2: np.ndarray = None,
    password: str = "Pass123!",
    original_payload: bytes = None,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.6,
    t1: float = 0.33,
    t2: float = 0.66
) -> Dict[str, Any]:
    """
    Evaluates robustness under attack for CNN-DA-EMD-OLSB (single-stego).

    Args:
        clean_stego:       Original stego image before attack.
        attacked_stego:    Stego image after attack.
        clean_stego_s2:    Legacy parameter for backward compatibility.
        password:          Decryption password.
        original_payload:  Ground truth payload bytes for genuine BER comparison.

    Returns:
        Dict with PSNR, SSIM, BER, Bit Recovery Accuracy, GCM status.
    """
    clean_stego_s1 = clean_stego
    attacked_stego_s1 = attacked_stego
    psnr_val = calculate_psnr(clean_stego_s1, attacked_stego_s1)
    ssim_val = calculate_ssim(clean_stego_s1, attacked_stego_s1)

    # Attempt payload extraction & GCM authentication using the real extraction function
    gcm_status = "FAILED (Authentication Tag Mismatch)"
    ber = 1.0
    bit_recovery_acc = 0.0
    extracted_payload = None

    try:
        from core.cnn_da_emd_olsb import extract_cnn_da_emd_olsb

        # Single-stego extraction from attacked image
        stego_input = attacked_stego_s1

        extracted_payload, rec_cov, meta = extract_cnn_da_emd_olsb(
            stego_input=stego_input, password=password, alpha=alpha, beta=beta, gamma=gamma, t1=t1, t2=t2
        )
        if meta.get('crc_match', False):
            gcm_status = "PASSED (100% Bit-Exact Recovery)"
            ber = 0.0
            bit_recovery_acc = 100.0
        elif original_payload is not None and extracted_payload is not None:
            gcm_status = "PARTIAL (Decrypted but CRC mismatch)"
            orig_arr = np.frombuffer(original_payload, dtype=np.uint8)
            extr_arr = np.frombuffer(extracted_payload, dtype=np.uint8)
            min_l = min(len(orig_arr), len(extr_arr))
            max_l = max(len(orig_arr), len(extr_arr))
            if min_l > 0:
                orig_bits = np.unpackbits(orig_arr)
                extr_bits = np.unpackbits(extr_arr)
                min_b = min(len(orig_bits), len(extr_bits))
                max_b = max(len(orig_bits), len(extr_bits))
                bit_errors = int(np.sum(orig_bits[:min_b] != extr_bits[:min_b]))
                bit_errors += abs(len(orig_bits) - len(extr_bits))
                ber = float(bit_errors / max_b) if max_b > 0 else 1.0
                bit_recovery_acc = float((1.0 - ber) * 100.0)
            else:
                ber = 1.0
                bit_recovery_acc = 0.0
        else:
            gcm_status = "PARTIAL (Decrypted with bit errors, CRC mismatch)"
            ber = 1.0
            bit_recovery_acc = 0.0
    except Exception:
        pass

    # Create spatial error map visualization
    error_diff = np.abs(clean_stego_s1.astype(np.float32) - attacked_stego_s1.astype(np.float32))
    spatial_error_map = np.clip(np.mean(error_diff, axis=2) * 5.0, 0, 255).astype(np.uint8)
    error_heatmap = cv2.applyColorMap(spatial_error_map, cv2.COLORMAP_JET)

    return {
        'PSNR_dB': round(psnr_val, 2),
        'SSIM': round(ssim_val, 4),
        'BER': round(ber, 4),
        'Bit_Recovery_Acc_%': round(bit_recovery_acc, 2),
        'GCM_Payload_Status': gcm_status,
        'Extracted_Payload': extracted_payload,
        'Error_Heatmap_RGB': cv2.cvtColor(error_heatmap, cv2.COLOR_BGR2RGB)
    }
