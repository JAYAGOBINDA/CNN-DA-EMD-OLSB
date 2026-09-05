# -*- coding: utf-8 -*-
"""
Steganalysis Module — Cover vs Stego Binary Classification.

Architecture: Patch-based CNN (SRM-inspired preprocessing) trained with PyTorch.

Data Leakage Prevention
-----------------------
The dataset is split by ORIGINAL COVER IMAGE index before patch extraction.
A cover image and its corresponding stego image always land in the same split.
This prevents the model from learning cover-specific artefacts that would
inflate test accuracy.

Split: 70 % Train / 15 % Validation / 15 % Test  (document if adjusted).

CNN Architecture
----------------
  Preprocessing : 1 × Conv2d(3→16, k=5) + |·| activation  (SRM-like residual)
  Block 1       : Conv2d(16→32, k=3) → BN → ReLU → MaxPool(2)
  Block 2       : Conv2d(32→64,  k=3) → BN → ReLU → MaxPool(2)
  Block 3       : Conv2d(64→128, k=3) → BN → ReLU → AdaptiveAvgPool(4×4)
  Head          : FC(2048→256) → ReLU → Dropout(0.5) → FC(256→1) → Sigmoid
"""

import os
import io
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
# CNN MODEL
# ══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class SteganalysisNet(nn.Module):
        """Lightweight SRM-inspired binary classifier for cover/stego patches."""

        def __init__(self):
            super().__init__()
            torch.manual_seed(42)

            # SRM-like high-pass preprocessing
            self.prep = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
            nn.init.xavier_uniform_(self.prep.weight)

            self.features = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = torch.abs(self.prep(x))   # SRM-like high-pass residual
            x = self.features(x)
            return self.classifier(x).squeeze(1)


    class _PatchDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.y = torch.from_numpy(y.astype(np.float32))

        def __len__(self) -> int:
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# PATCH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def extract_patches(
    img: np.ndarray,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
) -> np.ndarray:
    """
    Extract (N, 3, P, P) float32 patches from an RGB uint8 image.
    Pixels are normalised to [0, 1].
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
# STEGO GENERATION
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
# MAIN TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_steganalysis(
    cover_images: List[np.ndarray],
    stego_images: List[Optional[np.ndarray]],
    cover_names: List[str],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    n_epochs:    int   = 15,
    batch_size:  int   = 32,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate the steganalysis CNN.

    Splitting is strictly grouped by original cover image identity to prevent
    data leakage. If multiple stego variants exist for a cover, all of them
    remain together in either train, val, or test.
    Patches are extracted AFTER split assignment.
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch is not installed. Cannot run steganalysis."}

    # Filter pairs where stego generation succeeded
    pairs = [
        (c, s, n)
        for c, s, n in zip(cover_images, stego_images, cover_names)
        if s is not None
    ]
    if len(pairs) < 2:
        return {"error": f"Need ≥ 2 valid image pairs. Only {len(pairs)} available."}

    n = len(pairs)

    # Group by unique base cover identity to strictly prevent data leakage
    cover_groups: Dict[str, List[int]] = {}
    for i, (c, s, n_img) in enumerate(pairs):
        base_key = n_img.split('_bpp')[0].split('_s1')[0].split('_s2')[0].split('_stego')[0].strip()
        cover_groups.setdefault(base_key, []).append(i)

    unique_covers = list(cover_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(unique_covers)

    n_uc = len(unique_covers)
    n_train_uc = max(1, int(round(n_uc * train_ratio)))
    n_val_uc   = max(0, int(round(n_uc * val_ratio))) if n_uc > 2 else 0
    n_test_uc  = max(1, n_uc - n_train_uc - n_val_uc)
    while n_train_uc + n_val_uc + n_test_uc > n_uc:
        if n_val_uc > 0:
            n_val_uc -= 1
        elif n_train_uc > 1:
            n_train_uc -= 1
        else:
            break

    train_covers = unique_covers[:n_train_uc]
    val_covers   = unique_covers[n_train_uc : n_train_uc + n_val_uc]
    test_covers  = unique_covers[n_train_uc + n_val_uc :]

    tr_idx = [idx for k in train_covers for idx in cover_groups[k]]
    va_idx = [idx for k in val_covers for idx in cover_groups[k]]
    te_idx = [idx for k in test_covers for idx in cover_groups[k]]

    split_info = {
        "total_samples":       n,
        "total_image_pairs":   n,
        "train_samples":       len(tr_idx),
        "train_pairs":         len(tr_idx),
        "val_samples":         len(va_idx),
        "val_pairs":           len(va_idx),
        "test_samples":        len(te_idx),
        "test_pairs":          len(te_idx),
        "total_unique_covers": n_uc,
        "train_unique_covers": len(train_covers),
        "val_unique_covers":   len(val_covers),
        "test_unique_covers":  len(test_covers),
        "train_ratio_actual":  round(len(tr_idx) / n, 3),
        "val_ratio_actual":    round(len(va_idx) / n, 3),
        "test_ratio_actual":   round(len(te_idx) / n, 3),
        "split_method":        "Grouped by original cover image — all variants of same cover isolated in same split (zero data leakage)",
        "patch_size":          PATCH_SIZE,
        "patch_stride":        PATCH_STRIDE,
        "n_epochs":            n_epochs,
        "batch_size":          batch_size,
    }

    def _collect(idx_list):
        Xp, yp = [], []
        for i in idx_list:
            cov, stg, _ = pairs[i]
            pc = extract_patches(cov)
            ps = extract_patches(stg)
            if len(pc):
                Xp.append(pc); yp.append(np.zeros(len(pc), np.float32))
            if len(ps):
                Xp.append(ps); yp.append(np.ones(len(ps),  np.float32))
        if Xp:
            return np.concatenate(Xp), np.concatenate(yp)
        return (np.zeros((0, 3, PATCH_SIZE, PATCH_SIZE), np.float32),
                np.zeros(0, np.float32))

    if progress_callback:
        progress_callback(0, n_epochs, "Extracting patches …")

    Xtr, ytr = _collect(tr_idx)
    Xva, yva = _collect(va_idx)
    Xte, yte = _collect(te_idx)

    split_info["train_patches"] = len(Xtr)
    split_info["val_patches"]   = len(Xva)
    split_info["test_patches"]  = len(Xte)
    split_info["total_patches"] = len(Xtr) + len(Xva) + len(Xte)

    if len(Xtr) == 0:
        return {"error": "No training patches extracted (images too small?).",
                "split_info": split_info}
    if len(Xte) == 0:
        return {"error": "No test patches extracted. Use more images or larger images.",
                "split_info": split_info}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SteganalysisNet().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit   = nn.BCEWithLogitsLoss()

    loader = DataLoader(_PatchDataset(Xtr, ytr), batch_size=batch_size,
                        shuffle=True, num_workers=0)
    history = []

    for ep in range(n_epochs):
        if progress_callback:
            progress_callback(ep + 1, n_epochs, f"Training epoch {ep+1}/{n_epochs} on {device}")

        model.train()
        ep_loss = ep_correct = ep_total = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss   = crit(logits, yb)
            loss.backward()
            opt.step()
            ep_loss    += loss.item() * len(Xb)
            ep_correct += ((torch.sigmoid(logits) > 0.5).float() == yb).sum().item()
            ep_total   += len(Xb)
        sched.step()

        tr_loss = ep_loss / ep_total if ep_total else 0.0
        tr_acc  = ep_correct / ep_total if ep_total else 0.0

        va_acc = 0.0
        if len(Xva) > 0:
            model.eval()
            with torch.no_grad():
                va_preds = []
                for ci in range(0, len(Xva), batch_size):
                    chunk = torch.from_numpy(Xva[ci:ci+batch_size]).to(device)
                    va_preds.append((torch.sigmoid(model(chunk)) > 0.5).float().cpu().numpy())
                va_preds_all = np.concatenate(va_preds)
                va_acc = (va_preds_all.flatten() == yva).mean()

        history.append({
            "epoch": ep + 1,
            "train_loss": round(tr_loss, 5),
            "train_acc":  round(tr_acc,  4),
            "val_acc":    round(float(va_acc), 4),
        })

    # ── Evaluate on test set ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits_chunks = []
        for ci in range(0, len(Xte), batch_size):
            chunk = torch.from_numpy(Xte[ci:ci+batch_size]).to(device)
            logits_chunks.append(model(chunk).cpu())
        logits_te = torch.cat(logits_chunks)
        probs_te  = torch.sigmoid(logits_te).numpy()
        preds_te  = (probs_te > 0.5).astype(int)

    labels_te = yte.astype(int)

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

    cm = confusion_matrix(labels_te, preds_te, labels=[0, 1])

    metrics = {
        "accuracy":   round(float(acc),  4),
        "precision":  round(float(prec), 4),
        "recall":     round(float(rec),  4),
        "f1_score":   round(float(f1),   4),
        "roc_auc":    round(float(roc_auc), 4) if not np.isnan(roc_auc) else "N/A",
        "device":     str(device),
        "classifier": "SteganalysisNet (SRM-inspired CNN, PyTorch)",
    }

    hist_df = pd.DataFrame(history)
    cm_fig  = _plot_cm(cm)
    roc_fig = _plot_roc(fpr, tpr, roc_auc) if not np.isnan(roc_auc) else None
    his_fig = _plot_history(hist_df)

    return {
        "metrics":                 metrics,
        "confusion_matrix":        cm,
        "confusion_matrix_figure": cm_fig,
        "roc_curve_figure":        roc_fig,
        "training_history":        hist_df,
        "training_history_figure": his_fig,
        "split_info":              split_info,
        "test_probs":              probs_te.tolist(),
        "test_preds":              preds_te.tolist(),
        "test_labels":             labels_te.tolist(),
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _plot_cm(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Cover (0)", "Stego (1)"], rotation=20)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Cover (0)", "Stego (1)"])
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("Confusion Matrix — Steganalysis", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def _plot_roc(fpr, tpr, auc) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color="#8b5cf6", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curve — Steganalysis", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


def _plot_history(hist_df: pd.DataFrame) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], color="#ef4444", lw=2,
             marker="o", ms=4, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss", fontsize=10, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.4); ax1.legend(fontsize=9)

    ax2.plot(hist_df["epoch"], hist_df["train_acc"] * 100, color="#3b82f6", lw=2,
             marker="o", ms=4, label="Train Acc")
    if "val_acc" in hist_df:
        ax2.plot(hist_df["epoch"], hist_df["val_acc"] * 100, color="#10b981", lw=2,
                 marker="s", ms=4, label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy", fontsize=10, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.4); ax2.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ── Persistence ───────────────────────────────────────────────────────────────

def save_steganalysis_results(result: Dict[str, Any], base_dir: str = "results/steganalysis") -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)

    if "metrics" in result:
        pd.DataFrame([result["metrics"]]).to_csv(
            os.path.join(out, "classification_metrics.csv"), index=False)
    if "training_history" in result:
        result["training_history"].to_csv(
            os.path.join(out, "training_history.csv"), index=False)
    if "split_info" in result:
        pd.DataFrame([result["split_info"]]).to_csv(
            os.path.join(out, "split_info.csv"), index=False)
    for key, fname in [
        ("confusion_matrix_figure", "confusion_matrix.png"),
        ("roc_curve_figure",        "roc_curve.png"),
        ("training_history_figure", "training_history.png"),
    ]:
        fig = result.get(key)
        if fig:
            fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")
    return out


def get_steganalysis_zip(result: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if "metrics" in result:
            zf.writestr("classification_metrics.csv",
                        pd.DataFrame([result["metrics"]]).to_csv(index=False))
        if "training_history" in result:
            zf.writestr("training_history.csv",
                        result["training_history"].to_csv(index=False))
        if "split_info" in result:
            zf.writestr("split_info.csv",
                        pd.DataFrame([result["split_info"]]).to_csv(index=False))
        for key, fname in [
            ("confusion_matrix_figure", "confusion_matrix.png"),
            ("roc_curve_figure",        "roc_curve.png"),
            ("training_history_figure", "training_history.png"),
        ]:
            fig = result.get(key)
            if fig:
                fb = io.BytesIO()
                fig.savefig(fb, format="png", dpi=150, bbox_inches="tight")
                zf.writestr(fname, fb.getvalue())
    return buf.getvalue()
