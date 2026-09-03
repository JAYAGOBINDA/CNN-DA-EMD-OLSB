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


def optimize_secret_image(secret_rgb: np.ndarray, max_bytes: int) -> Tuple[bytes, int, int]:
    """
    Resizes and compresses secret image using PNG/JPEG optimization to guarantee payload fits max_bytes.
    Returns: (optimized_bytes, target_width, target_height)
    """
    if max_bytes < 256:
        max_bytes = 256

    pil_img = Image.fromarray(secret_rgb)
    
    # 1. Try lossless PNG format first
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG', compress_level=9)
    data = buf.getvalue()

    if len(data) <= max_bytes:
        return data, secret_rgb.shape[1], secret_rgb.shape[0]

    # 2. Try high-quality JPEG compression
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    data_jpg = buf.getvalue()
    if len(data_jpg) <= max_bytes:
        return data_jpg, secret_rgb.shape[1], secret_rgb.shape[0]

    # 3. Progressive Downscaling & Quality Adjustment
    curr_img = pil_img.copy()
    scale = 0.8
    while len(data_jpg) > max_bytes and curr_img.width > 16 and curr_img.height > 16:
        new_w = max(16, int(curr_img.width * scale))
        new_h = max(16, int(curr_img.height * scale))
        curr_img = curr_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        curr_img.save(buf, format='JPEG', quality=75)
        data_jpg = buf.getvalue()

        if len(data_jpg) <= max_bytes:
            break
        scale *= 0.8

    return data_jpg, curr_img.width, curr_img.height
