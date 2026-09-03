from pathlib import Path
from typing import List, Tuple
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Use the exact analytic map already present in your project as the
# supervision target. The CNN will learn to approximate this map.
from cnn.distortion_cnn import _analytic_distortion_map


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(root: str) -> List[Path]:
    root_path = Path(root)
    files = [p for p in root_path.rglob("*")
             if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    files.sort()
    if not files:
        raise RuntimeError(f"No images found under: {root_path.resolve()}")
    return files


def _target_map(rgb: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
    """Create the same type of [0,1] per-channel distortion target
    used by the existing analytic fallback."""
    maps = []
    for c in range(3):
        maps.append(_analytic_distortion_map(rgb[:, :, c], alpha, beta))
    return np.stack(maps, axis=2).astype(np.float32)


def _resize_keep_aspect(img: np.ndarray, min_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min_side / min(h, w)
    nh, nw = max(min_side, int(round(h * scale))), max(min_side, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _crop_pair(img: np.ndarray, target: np.ndarray, size: int, train: bool):
    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.resize(img, (max(size, w), max(size, h)), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (max(size, w), max(size, h)), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]

    if train:
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
    else:
        y = (h - size) // 2
        x = (w - size) // 2

    return img[y:y+size, x:x+size], target[y:y+size, x:x+size]


class DistortionMapDataset(Dataset):
    """RGB cover images -> target distortion-sensitivity maps."""

    def __init__(
        self,
        image_paths: List[Path],
        crop_size: int = 224,
        train: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.train = train
        self.alpha = alpha
        self.beta = beta

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_keep_aspect(rgb, self.crop_size)

        target = _target_map(rgb, self.alpha, self.beta)
        rgb, target = _crop_pair(rgb, target, self.crop_size, self.train)

        # Horizontal flip only; target must receive exactly the same transform.
        if self.train and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            target = np.ascontiguousarray(target[:, ::-1, :])

        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(target).permute(2, 0, 1)

        return x, y
