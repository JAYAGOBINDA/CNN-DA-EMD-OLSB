# -*- coding: utf-8 -*-
"""
Steganalysis Module — Cover vs Stego Binary Classification.

Research Goal:
Determine whether a classifier can distinguish between an original COVER image
and its corresponding STEGO image generated using the CNN-DA-EMD-OLSB method.

Strict Scientific Standards & Data Leakage Prevention:
------------------------------------------------------
1. Dataset is split strictly BY IMAGE PAIR (Cover + Stego together).
   A cover image and its corresponding stego image NEVER land in different splits.
2. Patches from the same image pair remain strictly within the same split.
3. Class Labels:
   - Cover = 0 (True Cover)
   - Stego = 1 (True Stego)
4. Architecture:
   - SRM-inspired high-pass residual preprocessing initialized with zero-sum directional filters.
   - TLU (Truncation Activation) to suppress high-variance content edges.
   - Feature extraction with BatchNorm and LeakyReLU.
   - Classifier collapse detection (identifying if predictions collapse to a single class).
"""

import os
import io
import re
import zipfile
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Callable, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

PATCH_SIZE   = 64
PATCH_STRIDE = 32


# ══════════════════════════════════════════════════════════════════════════════
# PAIRING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _extract_base_stem(filename: str) -> str:
    """Extract normalized base stem from filename for matching."""
    name = os.path.splitext(filename)[0].lower()
    # Strip common stego/cover prefixes and suffixes
    name = re.sub(r'^(cover_|stego_)', '', name)
    name = re.sub(r'(_stego|_cover|_s1|_s2|_embedded|-stego|-cover)$', '', name)
    return name.strip()


def pair_cover_and_stego(
    cover_images: List[np.ndarray],
    cover_names: List[str],
    stego_images: List[np.ndarray],
    stego_names: List[str],
) -> Dict[str, Any]:
    """
    Intelligently pair Cover and Stego images based on filename patterns and dimensions.

    Matching hierarchy:
    1. Exact filename match (e.g. uploaded from separate folders: '001.png' <-> '001.png')
    2. Normalized stem match (e.g. '001.png' <-> '001_stego.png' or 'cover_001.png' <-> 'stego_001.png')
    3. Prefix/Substring match (e.g. 'img1.png' <-> 'img1_embedded_s1.png')
    4. Fallback index match if counts are equal and names don't match

    Validates:
    - Cover and Stego dimensions are compatible (H, W, C).
    - Prevents duplicates and orphan images.
    """
    paired = []
    used_stego_indices = set()
    unmatched_covers = []
    mismatches = []

    # Map stego stems
    stego_stems = [_extract_base_stem(n) for n in stego_names]

    # Pass 1: Exact and Normalized Match
    for c_idx, (c_img, c_name) in enumerate(zip(cover_images, cover_names)):
        c_stem = _extract_base_stem(c_name)
        matched_s_idx = None
        match_type = ""

        # Check exact filename
        for s_idx, s_name in enumerate(stego_names):
            if s_idx in used_stego_indices:
                continue
            if c_name.lower() == s_name.lower():
                matched_s_idx = s_idx
                match_type = "Exact filename"
                break

        # Check normalized stem
        if matched_s_idx is None:
            for s_idx, s_stem in enumerate(stego_stems):
                if s_idx in used_stego_indices:
                    continue
                if c_stem == s_stem:
                    matched_s_idx = s_idx
                    match_type = "Stem match"
                    break

        # Check prefix/substring
        if matched_s_idx is None:
            for s_idx, (s_stem, s_name) in enumerate(zip(stego_stems, stego_names)):
                if s_idx in used_stego_indices:
                    continue
                if s_stem.startswith(c_stem) or c_stem.startswith(s_stem):
                    matched_s_idx = s_idx
                    match_type = "Prefix/Substring"
                    break

        if matched_s_idx is not None:
            s_img = stego_images[matched_s_idx]
            s_name = stego_names[matched_s_idx]

            # Validate dimensions
            if c_img.shape != s_img.shape:
                mismatches.append(
                    f"Dimension mismatch for pair '{c_name}' ({c_img.shape}) and '{s_name}' ({s_img.shape})"
                )
            else:
                used_stego_indices.add(matched_s_idx)
                paired.append({
                    "pair_id": len(paired) + 1,
                    "cover_name": c_name,
                    "stego_name": s_name,
                    "cover_img": c_img,
                    "stego_img": s_img,
                    "shape": c_img.shape,
                    "match_type": match_type,
                })
        else:
            unmatched_covers.append(c_name)

    # Pass 2: Fallback index match if no pairs were matched and lengths match
    if len(paired) == 0 and len(cover_images) == len(stego_images) and len(cover_images) > 0:
        for idx in range(len(cover_images)):
            c_img, c_name = cover_images[idx], cover_names[idx]
            s_img, s_name = stego_images[idx], stego_names[idx]
            if c_img.shape == s_img.shape:
                paired.append({
                    "pair_id": idx + 1,
                    "cover_name": c_name,
                    "stego_name": s_name,
                    "cover_img": c_img,
                    "stego_img": s_img,
                    "shape": c_img.shape,
                    "match_type": "Index alignment fallback",
                })
            else:
                mismatches.append(
                    f"Dimension mismatch for pair #{idx+1} '{c_name}' ({c_img.shape}) and '{s_name}' ({s_img.shape})"
                )

    unmatched_stegos = [
        stego_names[i] for i in range(len(stego_names)) if i not in used_stego_indices
    ]

    is_valid = len(paired) >= 2
    warning = None
    if len(cover_images) == 0 or len(stego_images) == 0:
        warning = "Please upload both Cover and Stego images."
    elif len(paired) < 2:
        warning = (
            f"Only {len(paired)} valid pair(s) found. Steganalysis experiment requires at least 2 valid pairs "
            "(at least 1 for training and 1 for unseen test evaluation)."
        )
    elif len(unmatched_covers) > 0 or len(unmatched_stegos) > 0:
        warning = (
            f"Matched {len(paired)} pairs. However, {len(unmatched_covers)} cover(s) and "
            f"{len(unmatched_stegos)} stego(s) could not be matched."
        )

    return {
        "pairs": paired,
        "n_covers": len(cover_images),
        "n_stegos": len(stego_images),
        "n_pairs": len(paired),
        "unmatched_covers": unmatched_covers,
        "unmatched_stegos": unmatched_stegos,
        "mismatches": mismatches,
        "is_valid": is_valid,
        "warning": warning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CNN MODEL (SRM-Inspired with Zero-Sum Residual Filters)
# ══════════════════════════════════════════════════════════════════════════════

def _get_srm_kernels() -> List[np.ndarray]:
    """
    Generate standard SRM (Spatial Rich Model) directional difference filters.
    All filters strictly satisfy the zero-sum condition (sum = 0) to eliminate
    natural image content and expose weak steganographic embedding residuals.
    """
    kernels = []
    # 1. 1st order horizontal & vertical
    k1 = np.zeros((5, 5), dtype=np.float32)
    k1[2, 2] = -1.0; k1[2, 3] = 1.0
    kernels.append(k1)

    k2 = np.zeros((5, 5), dtype=np.float32)
    k2[2, 2] = -1.0; k2[3, 2] = 1.0
    kernels.append(k2)

    # 2. 2nd order horizontal & vertical
    k3 = np.zeros((5, 5), dtype=np.float32)
    k3[2, 1] = 1.0; k3[2, 2] = -2.0; k3[2, 3] = 1.0
    kernels.append(k3)

    k4 = np.zeros((5, 5), dtype=np.float32)
    k4[1, 2] = 1.0; k4[2, 2] = -2.0; k4[3, 2] = 1.0
    kernels.append(k4)

    # 3. 2nd order diagonals
    k5 = np.zeros((5, 5), dtype=np.float32)
    k5[1, 1] = 1.0; k5[2, 2] = -2.0; k5[3, 3] = 1.0
    kernels.append(k5)

    k6 = np.zeros((5, 5), dtype=np.float32)
    k6[1, 3] = 1.0; k6[2, 2] = -2.0; k6[3, 1] = 1.0
    kernels.append(k6)

    # 4. 3x3 standard Laplacian
    k7 = np.zeros((5, 5), dtype=np.float32)
    k7[1, 2] = 1.0; k7[2, 1] = 1.0; k7[2, 2] = -4.0; k7[2, 3] = 1.0; k7[3, 2] = 1.0
    kernels.append(k7)

    # 5. 3x3 8-neighbor high-pass
    k8 = np.zeros((5, 5), dtype=np.float32)
    k8[1:4, 1:4] = -1.0; k8[2, 2] = 8.0
    kernels.append(k8)

    # 6. Edge detector filters (horizontal & vertical edges)
    k9 = np.zeros((5, 5), dtype=np.float32)
    k9[1, 1:4] = 1.0; k9[2, 1:4] = 0.0; k9[3, 1:4] = -1.0
    kernels.append(k9)

    k10 = np.zeros((5, 5), dtype=np.float32)
    k10[1:4, 1] = 1.0; k10[1:4, 2] = 0.0; k10[1:4, 3] = -1.0
    kernels.append(k10)

    # 7. 5x5 Laplacian KB filter
    k11 = np.array([
        [-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1]
    ], dtype=np.float32)
    kernels.append(k11)

    # 8. 3rd order filters
    k12 = np.zeros((5, 5), dtype=np.float32)
    k12[2, 0] = -1.0; k12[2, 1] = 3.0; k12[2, 2] = -3.0; k12[2, 3] = 1.0
    kernels.append(k12)

    k13 = np.zeros((5, 5), dtype=np.float32)
    k13[0, 2] = -1.0; k13[1, 2] = 3.0; k13[2, 2] = -3.0; k13[3, 2] = 1.0
    kernels.append(k13)

    # 9. Corner filters
    k14 = np.zeros((5, 5), dtype=np.float32)
    k14[1, 1] = 2.0; k14[1, 2] = -1.0; k14[2, 1] = -1.0
    kernels.append(k14)

    k15 = np.zeros((5, 5), dtype=np.float32)
    k15[1, 3] = 2.0; k15[1, 2] = -1.0; k15[2, 3] = -1.0
    kernels.append(k15)

    k16 = np.zeros((5, 5), dtype=np.float32)
    k16[3, 3] = 2.0; k16[2, 3] = -1.0; k16[3, 2] = -1.0
    kernels.append(k16)

    return kernels[:16]


if TORCH_AVAILABLE:
    class SteganalysisNet(nn.Module):
        """
        SRM-inspired Steganalysis CNN for 64×64 patches.
        Incorporates zero-sum high-pass filters, TLU (Truncation Activation),
        BatchNorm, LeakyReLU, and stable output layers to prevent classifier collapse.
        """

        def __init__(self, in_channels: int = 3, num_srm_filters: int = 16):
            super().__init__()
            torch.manual_seed(42)

            self.prep = nn.Conv2d(in_channels, num_srm_filters, kernel_size=5, padding=2, bias=False)
            self._init_srm_weights(in_channels, num_srm_filters)

            # Feature extractor
            self.features = nn.Sequential(
                nn.BatchNorm2d(num_srm_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_srm_filters, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),  # 64 -> 32

                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),  # 32 -> 16

                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d(4),  # 16 -> 4
            )

            # Dense Classification Head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, 1),
            )

            # Initialize final bias to 0 to prevent initial class bias
            nn.init.constant_(self.classifier[-1].bias, 0.0)

        def _init_srm_weights(self, in_channels: int, num_filters: int):
            kernels = _get_srm_kernels()
            with torch.no_grad():
                for f_idx in range(num_filters):
                    k = kernels[f_idx % len(kernels)]
                    for c in range(in_channels):
                        self.prep.weight[f_idx, c] = torch.from_numpy(k)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # 1. High-pass residual extraction
            residual = self.prep(x)
            # 2. Truncation activation (TLU) / absolute value: suppresses large natural edges
            x = torch.clamp(residual, -8.0, 8.0)
            # 3. Features & Classification
            feat = self.features(x)
            logits = self.classifier(feat).squeeze(1)
            return logits


    class _PatchDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# PATCH EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_patches(
    img: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
) -> np.ndarray:
    """
    Extract (N, 3, P, P) float32 patches from an RGB uint8 image.
    Normalized to [0, 1].
    """
    h, w = img.shape[:2]
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            p = img[y : y + patch_size, x : x + patch_size]
            patches.append(p.astype(np.float32).transpose(2, 0, 1) / 255.0)
    if patches:
        return np.stack(patches, axis=0)
    return np.zeros((0, 3, patch_size, patch_size), dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEGO GENERATION (FOR OPTIONAL SECONDARY MODE)
# ══════════════════════════════════════════════════════════════════════════════

def generate_stego_dataset(
    cover_images: List[np.ndarray],
    cover_names: List[str],
    password: str = "Pass123!",
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.6,
    t1: float = 0.33,
    t2: float = 0.66,
    payload_bpp: float = 0.05,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str]]:
    """
    Embed a payload into each cover image using trained CNN-DA-EMD-OLSB.
    Returns (cover_images, stego_images, names) - stego is None for failures.
    """
    from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb
    from cnn.distortion_cnn import load_trained_distortion_cnn
    OVERHEAD = 96

    model = load_trained_distortion_cnn() if gamma > 0.0 else None

    stegos: List[Optional[np.ndarray]] = []
    total  = len(cover_images)

    for idx, (cov, nm) in enumerate(zip(cover_images, cover_names)):
        if progress_callback:
            progress_callback(idx + 1, total, f"Embedding stego {idx+1}/{total}: {nm}")
        try:
            h, w, _ = cov.shape
            raw_bytes = max(1, int(payload_bpp * h * w / 8) - OVERHEAD)
            rng    = np.random.default_rng(idx + 7)
            secret = bytes(rng.integers(0, 256, raw_bytes, dtype=np.uint8))
            stego_img, _ = embed_cnn_da_emd_olsb(
                cover_rgb=cov, secret_data=secret, password=password,
                alpha=alpha, beta=beta, gamma=gamma, t1=t1, t2=t2, payload_type=0,
                model=model,
            )
            stegos.append(stego_img)
        except Exception:
            stegos.append(None)

    return cover_images, stegos, cover_names


# Convenience alias for consistency
generate_stego_for_steganalysis = generate_stego_dataset


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING & EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_steganalysis(
    paired_data: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    n_epochs:    int   = 15,
    batch_size:  int   = 32,
    lr:          float = 0.0005,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute Steganalysis Experiment with strict pair-based dataset splitting.

    Parameters:
    -----------
    paired_data: List of dicts with 'pair_id', 'cover_name', 'stego_name', 'cover_img', 'stego_img'
    train_ratio: Fraction of pairs allocated to training
    val_ratio: Fraction of pairs allocated to validation
    n_epochs: Training epochs
    batch_size: Mini-batch size
    lr: Learning rate for AdamW optimizer

    Splitting is strictly grouped by original image pairs to prevent
    data leakage: cover and stego remain together in either train, val, or test.
    Patches are extracted AFTER split assignment.
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch is not installed. Cannot execute steganalysis training."}

    n_pairs = len(paired_data)
    if n_pairs < 2:
        return {"error": f"Need at least 2 image pairs. Only {n_pairs} provided."}

    # ── Strict Pair-Based Dataset Splitting ──────────────────────────────────
    pair_indices = list(range(n_pairs))
    np.random.seed(42)
    np.random.shuffle(pair_indices)

    # Dynamic allocation ensuring at least 1 train and 1 test pair
    n_train = max(1, int(round(n_pairs * train_ratio)))
    if n_pairs >= 3 and val_ratio > 0.0:
        n_val = max(1, int(round(n_pairs * val_ratio)))
    else:
        n_val = 0

    n_test = n_pairs - n_train - n_val
    if n_test < 1:
        if n_val > 0:
            n_val -= 1
            n_test = 1
        else:
            n_train = max(1, n_train - 1)
            n_test = 1

    tr_pairs = [paired_data[i] for i in pair_indices[:n_train]]
    va_pairs = [paired_data[i] for i in pair_indices[n_train : n_train + n_val]]
    te_pairs = [paired_data[i] for i in pair_indices[n_train + n_val :]]

    split_info = {
        "total_pairs":        n_pairs,
        "total_samples":      n_pairs,
        "total_image_pairs":  n_pairs,
        "train_samples":      len(tr_pairs),
        "train_pairs":        len(tr_pairs),
        "val_samples":        len(va_pairs),
        "val_pairs":          len(va_pairs),
        "test_samples":       len(te_pairs),
        "test_pairs":         len(te_pairs),
        "train_ratio_target": train_ratio,
        "val_ratio_target":   val_ratio,
        "train_ratio_actual": round(len(tr_pairs) / n_pairs, 3),
        "val_ratio_actual":   round(len(va_pairs) / n_pairs, 3),
        "test_ratio_actual":  round(len(te_pairs) / n_pairs, 3),
        "split_policy":       "Pair-based splitting: cover and stego remain in identical split.",
        "split_method":       "Strict pair-based splitting — cover and stego remain together in same split (zero data leakage)",
        "patch_size":         PATCH_SIZE,
        "patch_stride":       PATCH_STRIDE,
        "n_epochs":           n_epochs,
        "batch_size":         batch_size,
        "learning_rate":      lr,
    }

    # ── Patch Extraction per Split ───────────────────────────────────────────
    def _extract_split_patches(pairs_subset: List[Dict[str, Any]]):
        X_list, y_list, meta_list = [], [], []
        for p in pairs_subset:
            p_id = p["pair_id"]
            c_name = p["cover_name"]
            s_name = p["stego_name"]

            # Cover patches -> Class 0
            c_patches = extract_patches(p["cover_img"])
            if len(c_patches) > 0:
                X_list.append(c_patches)
                y_list.append(np.zeros(len(c_patches), dtype=np.float32))
                meta_list.extend([{
                    "pair_id": p_id,
                    "image_name": c_name,
                    "true_label": 0,
                    "true_class": "Cover",
                }] * len(c_patches))

            # Stego patches -> Class 1
            s_patches = extract_patches(p["stego_img"])
            if len(s_patches) > 0:
                X_list.append(s_patches)
                y_list.append(np.ones(len(s_patches), dtype=np.float32))
                meta_list.extend([{
                    "pair_id": p_id,
                    "image_name": s_name,
                    "true_label": 1,
                    "true_class": "Stego",
                }] * len(s_patches))

        if X_list:
            return np.concatenate(X_list), np.concatenate(y_list), meta_list
        return (np.zeros((0, 3, PATCH_SIZE, PATCH_SIZE), dtype=np.float32),
                np.zeros(0, dtype=np.float32), [])

    if progress_callback:
        progress_callback(0, n_epochs, "Extracting patches from image pairs …")

    Xtr, ytr, tr_meta = _extract_split_patches(tr_pairs)
    Xva, yva, va_meta = _extract_split_patches(va_pairs)
    Xte, yte, te_meta = _extract_split_patches(te_pairs)

    split_info["train_patches"] = len(Xtr)
    split_info["val_patches"]   = len(Xva)
    split_info["test_patches"]  = len(Xte)
    split_info["total_patches"] = len(Xtr) + len(Xva) + len(Xte)

    if len(Xtr) == 0:
        return {"error": "No training patches extracted. Ensure image dimensions >= 64×64.", "split_info": split_info}
    if len(Xte) == 0:
        return {"error": "No test patches extracted. Ensure image dimensions >= 64×64.", "split_info": split_info}

    # ── Model Setup ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SteganalysisNet().to(device)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)
    crit   = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        _PatchDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    history = []

    # ── Training Loop ────────────────────────────────────────────────────────
    for ep in range(n_epochs):
        model.train()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = crit(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            ep_loss += loss.item() * len(Xb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            ep_correct += (preds == yb).sum().item()
            ep_total += len(Xb)

        sched.step()
        tr_loss = ep_loss / max(1, ep_total)
        tr_acc  = ep_correct / max(1, ep_total)

        # Validation evaluation
        va_loss = float("nan")
        va_acc  = float("nan")
        if len(Xva) > 0:
            model.eval()
            va_ep_loss = 0.0
            va_correct = 0
            with torch.no_grad():
                for ci in range(0, len(Xva), batch_size):
                    chunk_X = torch.from_numpy(Xva[ci : ci + batch_size]).to(device)
                    chunk_y = torch.from_numpy(yva[ci : ci + batch_size]).to(device)
                    v_logits = model(chunk_X)
                    v_loss = crit(v_logits, chunk_y)
                    va_ep_loss += v_loss.item() * len(chunk_X)
                    v_preds = (torch.sigmoid(v_logits) > 0.5).float()
                    va_correct += (v_preds == chunk_y).sum().item()

            va_loss = va_ep_loss / len(Xva)
            va_acc  = va_correct / len(Xva)

        history.append({
            "epoch": ep + 1,
            "train_loss": round(float(tr_loss), 5),
            "train_acc":  round(float(tr_acc),  4),
            "val_loss":   round(float(va_loss), 5) if not np.isnan(va_loss) else "N/A",
            "val_acc":    round(float(va_acc),  4) if not np.isnan(va_acc) else "N/A",
        })

        if progress_callback:
            val_msg = f" | Val Acc: {va_acc*100:.1f}%" if not np.isnan(va_acc) else ""
            progress_callback(
                ep + 1, n_epochs,
                f"Epoch {ep+1}/{n_epochs} | Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.1f}%{val_msg}"
            )

    # ── Test Set Inference (Chunked to prevent OOM) ──────────────────────────
    model.eval()
    with torch.no_grad():
        logits_chunks = []
        for ci in range(0, len(Xte), batch_size):
            chunk = torch.from_numpy(Xte[ci : ci + batch_size]).to(device)
            logits_chunks.append(model(chunk).cpu())
        logits_te = torch.cat(logits_chunks)
        probs_te  = torch.sigmoid(logits_te).numpy()
        preds_te  = (probs_te > 0.5).astype(int)

    labels_te = yte.astype(int)

    # ── Test Metrics Calculation ─────────────────────────────────────────────
    acc  = accuracy_score(labels_te, preds_te)
    prec = precision_score(labels_te, preds_te, zero_division=0)
    rec  = recall_score(labels_te, preds_te, zero_division=0)
    f1   = f1_score(labels_te, preds_te, zero_division=0)

    try:
        roc_auc = roc_auc_score(labels_te, probs_te)
        fpr, tpr, _ = roc_curve(labels_te, probs_te)
    except Exception:
        roc_auc = float("nan")
        fpr = tpr = np.array([0.0, 1.0])

    cm_test = confusion_matrix(labels_te, preds_te, labels=[0, 1])

    # ── Collapse Detection ───────────────────────────────────────────────────
    unique_preds = np.unique(preds_te)
    is_collapsed = (len(unique_preds) <= 1)

    act_cover = int(np.sum(labels_te == 0))
    act_stego = int(np.sum(labels_te == 1))
    pred_cover = int(np.sum(preds_te == 0))
    pred_stego = int(np.sum(preds_te == 1))

    collapse_warning = None
    if is_collapsed:
        collapsed_to = "Stego (class 1)" if pred_stego == len(preds_te) else "Cover (class 0)"
        collapse_warning = (
            f"Classifier collapsed to a single class ({collapsed_to}). "
            "All test patches were predicted as the same class. "
            "The current result should NOT be interpreted as successful steganalysis."
        )

    # Training and Validation Confusion Matrices for Diagnostics
    cm_train = None
    with torch.no_grad():
        tr_logits = []
        for ci in range(0, len(Xtr), batch_size):
            chunk = torch.from_numpy(Xtr[ci : ci + batch_size]).to(device)
            tr_logits.append(model(chunk).cpu())
        tr_probs = torch.sigmoid(torch.cat(tr_logits)).numpy()
        tr_preds = (tr_probs > 0.5).astype(int)
        cm_train = confusion_matrix(ytr.astype(int), tr_preds, labels=[0, 1])

    cm_val = None
    if len(Xva) > 0:
        with torch.no_grad():
            va_logits = []
            for ci in range(0, len(Xva), batch_size):
                chunk = torch.from_numpy(Xva[ci : ci + batch_size]).to(device)
                va_logits.append(model(chunk).cpu())
            va_probs = torch.sigmoid(torch.cat(va_logits)).numpy()
            va_preds = (va_probs > 0.5).astype(int)
            cm_val = confusion_matrix(yva.astype(int), va_preds, labels=[0, 1])

    # ── Image-Level Predictions Assembly ─────────────────────────────────────
    # Aggregate patch probabilities per test image to produce image-level predictions
    image_preds_rows = []
    # te_meta maps each patch to its image
    for p_idx, meta in enumerate(te_meta):
        meta["patch_prob"] = float(probs_te[p_idx])

    patch_df = pd.DataFrame(te_meta)
    grouped = patch_df.groupby(["pair_id", "image_name", "true_label", "true_class"])
    for (p_id, img_name, t_label, t_class), grp in grouped:
        avg_prob = float(grp["patch_prob"].mean())
        p_label  = 1 if avg_prob > 0.5 else 0
        p_class  = "Stego" if p_label == 1 else "Cover"
        image_preds_rows.append({
            "pair_id":                p_id,
            "image_name":             img_name,
            "true_label":             t_label,
            "true_class":             t_class,
            "predicted_label":        p_label,
            "predicted_class":        p_class,
            "prediction_probability": round(avg_prob, 4),
            "patches_evaluated":      len(grp),
        })

    predictions_df = pd.DataFrame(image_preds_rows)

    metrics = {
        "accuracy":           round(float(acc),  4),
        "precision":          round(float(prec), 4),
        "recall":             round(float(rec),  4),
        "f1_score":           round(float(f1),   4),
        "roc_auc":            round(float(roc_auc), 4) if not np.isnan(roc_auc) else "N/A",
        "is_collapsed":       is_collapsed,
        "collapse_warning":   collapse_warning,
        "actual_cover_count": act_cover,
        "actual_stego_count": act_stego,
        "pred_cover_count":   pred_cover,
        "pred_stego_count":   pred_stego,
        "device":             str(device),
        "classifier":         "SteganalysisNet (SRM-Inspired CNN, PyTorch)",
    }

    hist_df = pd.DataFrame(history)
    cm_fig  = _plot_cm(cm_test, title="Test Set Confusion Matrix")
    roc_fig = _plot_roc(fpr, tpr, roc_auc) if not np.isnan(roc_auc) else None
    his_fig = _plot_history(hist_df)

    # ── Markdown Report Generation ───────────────────────────────────────────
    report_text = _generate_markdown_report(
        split_info=split_info,
        metrics=metrics,
        cm=cm_test,
        predictions_df=predictions_df,
    )

    return {
        "metrics":                 metrics,
        "confusion_matrix":        cm_test,
        "confusion_matrix_train":  cm_train,
        "confusion_matrix_val":    cm_val,
        "confusion_matrix_figure": cm_fig,
        "roc_curve_figure":        roc_fig,
        "training_history":        hist_df,
        "training_history_figure": his_fig,
        "predictions_df":          predictions_df,
        "split_info":              split_info,
        "report_md":               report_text,
        "test_pairs":              te_pairs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_cm(cm: np.ndarray, title: str = "Confusion Matrix — Steganalysis") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Cover (0)", "Stego (1)"], fontsize=11, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Cover (0)", "Stego (1)"], fontsize=11, fontweight="bold")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    labels_grid = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            tag = labels_grid[i][j]
            color = "white" if val > thresh else "#0f172a"
            ax.text(
                j, i, f"{val}\n({tag})",
                ha="center", va="center",
                color=color, fontsize=13, fontweight="bold"
            )

    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def _plot_roc(fpr, tpr, auc) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    ax.plot(fpr, tpr, color="#8b5cf6", lw=2.5, label=f"SRM-CNN ROC (AUC = {auc_str})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title("ROC Curve — Steganalysis", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _plot_history(hist_df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], color="#ef4444", lw=2.2,
             marker="o", ms=4, label="Train BCE Loss")
    val_loss_series = pd.to_numeric(hist_df["val_loss"], errors="coerce")
    if not val_loss_series.isna().all():
        ax1.plot(hist_df["epoch"], val_loss_series, color="#f59e0b", lw=2.2,
                 marker="s", ms=4, label="Val BCE Loss")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("BCE Loss", fontsize=10, fontweight="bold")
    ax1.set_title("Training & Validation Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.4)
    ax1.legend(fontsize=9)

    # Accuracy curve
    ax2.plot(hist_df["epoch"], hist_df["train_acc"] * 100, color="#3b82f6", lw=2.2,
             marker="o", ms=4, label="Train Accuracy")
    val_acc_series = pd.to_numeric(hist_df["val_acc"], errors="coerce")
    if not val_acc_series.isna().all():
        ax2.plot(hist_df["epoch"], val_acc_series * 100, color="#10b981", lw=2.2,
                 marker="s", ms=4, label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Training & Validation Accuracy", fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE & REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as a Markdown table without requiring tabulate."""
    if df.empty:
        return "No data."
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    return "\n".join(lines)


def _generate_markdown_report(
    split_info: Dict[str, Any],
    metrics: Dict[str, Any],
    cm: np.ndarray,
    predictions_df: pd.DataFrame,
) -> str:
    """Generate structured markdown report of steganalysis results."""
    is_collapsed = metrics.get("is_collapsed", False)
    acc = metrics.get("accuracy", 0.0)

    if is_collapsed:
        interpretation = (
            "> [!WARNING]\n"
            "> **Classifier Collapse Detected:** The network predicted all test samples as a single class. "
            "This indicates that the classifier failed to find separable statistical boundaries between cover "
            "and stego patches under the current hyperparameters or sample size. This outcome does NOT prove "
            "undetectability; rather, it reflects an inconclusive or collapsed optimization state."
        )
    elif acc > 0.65:
        interpretation = (
            "> [!NOTE]\n"
            "> **Detectable Statistical Residuals:** The classifier performed substantially above chance level "
            f"({acc*100:.1f}%) on the completely unseen test pairs. This indicates that the embedding procedure "
            "leaves detectable statistical modifications in spatial correlation or residual statistics."
        )
    else:
        interpretation = (
            "> [!TIP]\n"
            "> **Low Detectability under Tested Conditions:** The classifier performed near chance level "
            f"({acc*100:.1f}%) while predicting both classes. Under this specific classifier architecture, "
            "patch resolution, and dataset size, CNN-DA-EMD-OLSB demonstrated strong resistance to spatial steganalysis. "
            "*(Note: 50% accuracy under a specific CNN does not imply universal security against all possible attacks).*"
        )

    return f"""# 🕵️ Steganalysis Research Report: Cover vs. Stego Binary Classification

**Experiment Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Method Evaluated:** CNN-DA-EMD-OLSB Dual-Steganography Model  
**Classifier:** SteganalysisNet (SRM-Inspired High-Pass CNN, PyTorch)  

---

## 1. Executive Summary & Interpretation

{interpretation}

---

## 2. Dataset & Pair-Based Split Architecture

Strict pair-based splitting was enforced to eliminate data leakage. No original cover and its corresponding stego counterpart were ever separated into different splits.

| Split Phase | Image Pairs | Extracted Patches (64×64) | Allocation Ratio |
|:------------|:------------|:--------------------------|:-----------------|
| **Training** | {split_info.get('train_pairs', '?')} | {split_info.get('train_patches', '?')} | {split_info.get('train_ratio_actual', 0.0)*100:.1f}% |
| **Validation** | {split_info.get('val_pairs', '?')} | {split_info.get('val_patches', '?')} | {split_info.get('val_ratio_actual', 0.0)*100:.1f}% |
| **Unseen Testing** | {split_info.get('test_pairs', '?')} | {split_info.get('test_patches', '?')} | {split_info.get('test_ratio_actual', 0.0)*100:.1f}% |
| **Total** | **{split_info.get('total_pairs', '?')}** | **{split_info.get('train_patches', 0) + split_info.get('val_patches', 0) + split_info.get('test_patches', 0)}** | **100.0%** |

---

## 3. Unseen Test Set Performance Metrics

| Metric | Score | Benchmark Target |
|:-------|:------|:-----------------|
| **Accuracy** | `{metrics.get('accuracy', 0.0):.4f}` | 0.5000 (Ideal undetectable = 0.50) |
| **Precision** | `{metrics.get('precision', 0.0):.4f}` | 0.5000 |
| **Recall (Sensitivity)** | `{metrics.get('recall', 0.0):.4f}` | 0.5000 |
| **F1-Score** | `{metrics.get('f1_score', 0.0):.4f}` | 0.5000 |
| **ROC-AUC** | `{metrics.get('roc_auc', 'N/A')}` | 0.5000 (Random guess) |

---

## 4. Class Distribution & Collapse Verification

| Class | Actual Test Patches | Predicted Test Patches |
|:------|:--------------------|:-----------------------|
| **Cover (Class 0)** | {metrics.get('actual_cover_count', 0)} | {metrics.get('pred_cover_count', 0)} |
| **Stego (Class 1)** | {metrics.get('actual_stego_count', 0)} | {metrics.get('pred_stego_count', 0)} |

**Collapse State:** `{'⚠️ COLLAPSED' if is_collapsed else '✅ BALANCED PREDICTION'}`

---

## 5. Confusion Matrix (Unseen Test Set)

```
                    Predicted Cover (0)     Predicted Stego (1)
True Cover (0)            {cm[0, 0]:<15}         {cm[0, 1]:<15}
True Stego (1)            {cm[1, 0]:<15}         {cm[1, 1]:<15}
```

- **True Negatives (TN):** {cm[0, 0]} (Clean covers correctly identified)
- **False Positives (FP):** {cm[0, 1]} (Clean covers misclassified as stego)
- **False Negatives (FN):** {cm[1, 0]} (Stego images undetected / misclassified as cover)
- **True Positives (TP):** {cm[1, 1]} (Stego images correctly detected)

---

## 6. Image-Level Prediction Summary

{_df_to_markdown(predictions_df)}

---
*Report generated automatically by CNN-DA-EMD-OLSB Research Suite.*
"""


def save_steganalysis_results(result: Dict[str, Any], base_dir: str = "results/steganalysis") -> str:
    """Save all experiment artifacts into timestamped directory."""
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)

    # 1. Metrics CSV
    if "metrics" in result:
        pd.DataFrame([result["metrics"]]).to_csv(os.path.join(out, "metrics.csv"), index=False)

    # 2. Predictions CSV
    if "predictions_df" in result and not result["predictions_df"].empty:
        result["predictions_df"].to_csv(os.path.join(out, "predictions.csv"), index=False)

    # 3. Training history CSV
    if "training_history" in result:
        result["training_history"].to_csv(os.path.join(out, "training_history.csv"), index=False)

    # 4. Split info CSV
    if "split_info" in result:
        pd.DataFrame([result["split_info"]]).to_csv(os.path.join(out, "split_info.csv"), index=False)

    # 5. Figures
    for key, fname in [
        ("confusion_matrix_figure", "confusion_matrix.png"),
        ("roc_curve_figure",        "roc_curve.png"),
        ("training_history_figure", "training_curves.png"),
    ]:
        fig = result.get(key)
        if fig:
            fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")

    # 6. Full Markdown Report
    if "report_md" in result:
        with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
            f.write(result["report_md"])

    return out


def get_steganalysis_zip(result: Dict[str, Any]) -> bytes:
    """Create in-memory zip containing all experiment artifacts."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if "metrics" in result:
            zf.writestr("metrics.csv", pd.DataFrame([result["metrics"]]).to_csv(index=False))
        if "predictions_df" in result and not result["predictions_df"].empty:
            zf.writestr("predictions.csv", result["predictions_df"].to_csv(index=False))
        if "training_history" in result:
            zf.writestr("training_history.csv", result["training_history"].to_csv(index=False))
        if "split_info" in result:
            zf.writestr("split_info.csv", pd.DataFrame([result["split_info"]]).to_csv(index=False))
        if "report_md" in result:
            zf.writestr("report.md", result["report_md"])

        for key, fname in [
            ("confusion_matrix_figure", "confusion_matrix.png"),
            ("roc_curve_figure",        "roc_curve.png"),
            ("training_history_figure", "training_curves.png"),
        ]:
            fig = result.get(key)
            if fig:
                fb = io.BytesIO()
                fig.savefig(fb, format="png", dpi=150, bbox_inches="tight")
                zf.writestr(fname, fb.getvalue())

    return buf.getvalue()
