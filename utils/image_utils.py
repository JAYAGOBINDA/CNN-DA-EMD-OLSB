"""
Image Processing Utilities for 6-Model Benchmark Suite.
Handles color space conversions, histogram calculations, padding, and image quality helper functions.
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Union


def load_image_rgb(file_or_path: Union[str, bytes, io.BytesIO]) -> np.ndarray:
    """
    Loads an image and returns a 3D NumPy array in RGB format (uint8).
    """
    if isinstance(file_or_path, str):
        img_bgr = cv2.imread(file_or_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found at path: {file_or_path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(file_or_path, bytes):
        pil_img = Image.open(io.BytesIO(file_or_path)).convert('RGB')
        return np.array(pil_img)
    elif isinstance(file_or_path, io.BytesIO):
        pil_img = Image.open(file_or_path).convert('RGB')
        return np.array(pil_img)
    else:
        raise ValueError("Unsupported input type for image loading.")


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    """
    Converts RGB image to 2D Grayscale uint8 array.
    """
    if img_rgb.ndim == 2:
        return img_rgb.copy()
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def gray_to_rgb(img_gray: np.ndarray) -> np.ndarray:
    """
    Converts 2D Grayscale image to 3D RGB uint8 array by replicating channels.
    """
    if img_gray.ndim == 3:
        return img_gray.copy()
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def resize_image(img: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Resizes image to target_size (width, height).
    """
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def pad_to_multiple(img: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pads image height and width to be divisible by `multiple`.
    """
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    if img.ndim == 3:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')

    return padded, (h, w)


def unpad_image(img: np.ndarray, orig_size: Tuple[int, int]) -> np.ndarray:
    """
    Crops image back to orig_size (orig_h, orig_w).
    """
    orig_h, orig_w = orig_size
    return img[:orig_h, :orig_w]


def optimize_secret_image(
    secret_rgb: Union[np.ndarray, Image.Image, bytes],
    max_bytes: int
) -> Tuple[bytes, int, int]:
    """
    Optimizes secret image (any dimension/filesize) to strictly fit within max_bytes.
    Uses multi-tier adaptive strategy:
      1. Lossless PNG at original resolution (0% quality loss if budget permits)
      2. Adaptive WebP / JPEG with binary-searched resolution & quality
      3. Compact palette / low-bitrate fallbacks for extreme small-budget covers

    Returns:
        (optimized_bytes, target_width, target_height)
    """
    if max_bytes < 48:
        max_bytes = 48

    if isinstance(secret_rgb, np.ndarray):
        pil_img = Image.fromarray(secret_rgb).convert('RGB')
    elif isinstance(secret_rgb, Image.Image):
        pil_img = secret_rgb.convert('RGB')
    elif isinstance(secret_rgb, (bytes, io.BytesIO)):
        bio = secret_rgb if isinstance(secret_rgb, io.BytesIO) else io.BytesIO(secret_rgb)
        pil_img = Image.open(bio).convert('RGB')
    else:
        raise ValueError(f"Unsupported secret_rgb type: {type(secret_rgb)}")

    orig_w, orig_h = pil_img.size

    # ── Tier 1: Try Lossless PNG at original resolution ─────────────────────────
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', compress_level=9)
    raw_png = buf.getvalue()
    if len(raw_png) <= max_bytes:
        return raw_png, orig_w, orig_h

    # Helper function to encode image with smallest valid format (WebP preferred, JPEG fallback)
    def _encode(img: Image.Image, quality: int) -> bytes:
        b_webp = io.BytesIO()
        data_w = None
        try:
            img.save(b_webp, format='WEBP', quality=quality)
            data_w = b_webp.getvalue()
        except Exception:
            data_w = None

        b_jpg = io.BytesIO()
        try:
            img.save(b_jpg, format='JPEG', quality=quality)
            data_j = b_jpg.getvalue()
        except Exception:
            data_j = None

        if data_w is not None and data_j is not None:
            return data_w if len(data_w) <= len(data_j) else data_j
        elif data_w is not None:
            return data_w
        elif data_j is not None:
            return data_j
        else:
            # Fallback to PNG
            b_png = io.BytesIO()
            img.save(b_png, format='PNG', compress_level=6)
            return b_png.getvalue()

    # Pre-downscale for search efficiency if image is massive (>1024 on any side)
    max_dim = max(orig_w, orig_h)
    if max_dim > 1024:
        proxy_scale = 1024.0 / max_dim
        proxy_w = max(16, int(round(orig_w * proxy_scale)))
        proxy_h = max(16, int(round(orig_h * proxy_scale)))
        search_base = pil_img.resize((proxy_w, proxy_h), Image.Resampling.BILINEAR)
        search_w, search_h = proxy_w, proxy_h
    else:
        search_base = pil_img
        search_w, search_h = orig_w, orig_h

    # ── Tier 2: Adaptive Binary Search on Resolution & Quality ──────────────────
    low_s = 16.0 / max(search_w, search_h)
    best_candidate = None
    best_w, best_h = 16, 16

    # Test qualities from high to moderate
    for q in [85, 75, 60, 45]:
        low = low_s
        high = 1.0
        candidate_found = False
        for _ in range(7):
            mid = (low + high) / 2
            w = max(16, int(round(search_w * mid)))
            h = max(16, int(round(search_h * mid)))
            cand_img = search_base.resize((w, h), Image.Resampling.BILINEAR)
            enc = _encode(cand_img, quality=q)
            if len(enc) <= max_bytes:
                best_candidate = (w, h, q)
                candidate_found = True
                low = mid  # Can we fit a larger resolution?
            else:
                high = mid

        if candidate_found:
            break

    # ── Tier 3: Render final image using high-quality LANCZOS from original ─────
    if best_candidate is not None:
        target_w, target_h, target_q = best_candidate
        # Scale back up to original aspect ratio scale
        final_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
        final_data = _encode(final_img, quality=target_q)

        # If LANCZOS filtering pushed bytes slightly over budget, nudge down
        attempts = 0
        while len(final_data) > max_bytes and attempts < 5:
            target_w = max(16, int(target_w * 0.92))
            target_h = max(16, int(target_h * 0.92))
            target_q = max(25, int(target_q * 0.9))
            final_img = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            final_data = _encode(final_img, quality=target_q)
            attempts += 1

        if len(final_data) <= max_bytes:
            return final_data, target_w, target_h

    # ── Tier 4: Ultra-Compact Fallback for Tiny Budgets (e.g. < 300 bytes) ───────
    for q in [30, 20, 10]:
        tiny_img = pil_img.resize((16, 16), Image.Resampling.BILINEAR)
        final_data = _encode(tiny_img, quality=q)
        if len(final_data) <= max_bytes:
            return final_data, 16, 16

    # Grayscale compact fallback
    gray_img = pil_img.resize((16, 16), Image.Resampling.BILINEAR).convert('L')
    b_gray = io.BytesIO()
    gray_img.save(b_gray, format='WEBP', quality=15)
    data_gray = b_gray.getvalue()
    return data_gray, 16, 16

