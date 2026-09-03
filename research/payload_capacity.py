"""
Payload Capacity Experiment Module.

Tests the CNN-DA-EMD-OLSB model across multiple BPP levels and images.
Calls the real embed_cnn_da_emd_olsb / extract_cnn_da_emd_olsb functions —
no dummy or fabricated results.

Computed metrics per (image, BPP) pair:
  - target_bpp, actual_bpp, payload_bits, payload_bytes
  - PSNR (dB), SSIM, MSE
  - BER, payload_recovery_%, cover_recovery_%
  - embed_time_s, extract_time_s
"""

import os
import io
import time
import zipfile
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Callable, Optional

from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb
from benchmark.metrics import calculate_psnr, calculate_ssim, compute_mse
from cnn.distortion_cnn import load_trained_distortion_cnn

# Standard target BPP levels required for comprehensive capacity analysis
DEFAULT_BPP_LEVELS: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

# Header + AES overhead so we don't fabricate BPP; subtract from target
_HEADER_OVERHEAD_BYTES = 64 + 28  # 64-byte header + ~28 bytes AES overhead (nonce+tag)


def _make_payload(n_bytes: int, seed: int = 42) -> bytes:
    """Return deterministic random payload of exactly n_bytes."""
    rng = np.random.default_rng(seed)
    return bytes(rng.integers(0, 256, n_bytes, dtype=np.uint8))


def run_payload_capacity_experiment(
    images: List[np.ndarray],
    image_names: List[str],
    bpp_levels: Optional[List[float]] = None,
    password: str = "Pass123!",
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.6,
    t1: float = 0.33,
    t2: float = 0.66,
    model=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[plt.Figure]]:
    """
    Run payload capacity sweep using trained CNN-DA-EMD-OLSB.

    Parameters
    ----------
    images       : List of RGB uint8 numpy arrays (cover images).
    image_names  : Corresponding display names.
    bpp_levels   : Target BPP values to test (defaults to [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]).
    model        : Optional trained DistortionCNN instance.
    progress_callback : Called as (step, total, status_string).

    Returns
    -------
    results_df : Per-(image, BPP) measurement table.
    stats_df   : Summary statistics (mean, std, min, max) per BPP level.
    figures    : [fig_psnr, fig_ssim, fig_mse, fig_ber] matplotlib Figures.
    """
    if bpp_levels is None or len(bpp_levels) == 0:
        bpp_levels = list(DEFAULT_BPP_LEVELS)

    if model is None and gamma > 0.0:
        model = load_trained_distortion_cnn()

    rows = []
    total = len(images) * len(bpp_levels)
    step = 0

    for img_idx, (cover, name) in enumerate(zip(images, image_names)):
        h, w, c = cover.shape

        for bpp_target in bpp_levels:
            step += 1
            if progress_callback:
                progress_callback(
                    step, total,
                    f"Image {img_idx + 1}/{len(images)} ({name}) | BPP = {bpp_target}"
                )

            # target_bits uses standard per-pixel definition: BPP × H × W
            target_bits = int(bpp_target * h * w)
            target_bytes = max(1, target_bits // 8)
            raw_bytes = max(1, target_bytes - _HEADER_OVERHEAD_BYTES)

            row: Dict[str, Any] = {
                "image_id":                 name,
                "image_hw":                 f"{h}×{w}",
                "requested_bpp":            bpp_target,
                "target_bpp":               bpp_target,  # compatibility alias
                "target_bits":              target_bits,
                "target_bytes":             target_bytes,
                "payload_size_bytes":       raw_bytes,
                "payload_size_bits":        raw_bytes * 8,
                "payload_bytes":            raw_bytes,   # compatibility alias
                "payload_bits":             raw_bytes * 8,
            }

            try:
                secret = _make_payload(raw_bytes, seed=img_idx * 1000 + int(bpp_target * 10000))

                t0 = time.time()
                stego_dual, stats = embed_cnn_da_emd_olsb(
                    cover_rgb=cover,
                    secret_data=secret,
                    password=password,
                    alpha=alpha, beta=beta, gamma=gamma,
                    t1=t1, t2=t2,
                    payload_type=0,
                    model=model,
                )
                embed_time = time.time() - t0
                s1, s2 = stego_dual

                psnr_val = calculate_psnr(cover, s1)
                ssim_val = calculate_ssim(cover, s1)
                mse_val  = compute_mse(cover, s1)

                actual_raw_bpp = stats.get('raw_bpp', round((raw_bytes * 8) / (h * w), 6))
                actual_embedded_bpp = stats.get('embedded_bpp', round(stats.get('internal_bits_embedded', 0) / (h * w), 6))
                usable_cap = stats.get('usable_capacity_bits', stats.get('max_capacity_bits', 0))
                theo_cap = stats.get('theoretical_capacity_bits', usable_cap)
                cap_util = stats.get('capacity_utilization_%', round((stats.get('internal_bits_embedded', 0) / usable_cap * 100), 2) if usable_cap > 0 else 0.0)

                t1_ex = time.time()
                extracted, recovered, meta = extract_cnn_da_emd_olsb(
                    stego_dual=stego_dual,
                    password=password,
                    alpha=alpha, beta=beta, gamma=gamma,
                    t1=t1, t2=t2,
                    model=model,
                )
                extract_time = time.time() - t1_ex

                # BER & payload recovery
                if extracted == secret:
                    ber = 0.0
                    rec_acc = 100.0
                    extr_ok = True
                else:
                    extr_ok = False
                    orig_arr = np.frombuffer(secret, dtype=np.uint8)
                    extr_arr = np.frombuffer(extracted, dtype=np.uint8)
                    min_l = min(len(orig_arr), len(extr_arr))
                    if min_l > 0:
                        ob = np.unpackbits(orig_arr[:min_l])
                        eb = np.unpackbits(extr_arr[:min_l])
                        ber = float(np.sum(ob != eb) / len(ob))
                        rec_acc = float(
                            np.sum(orig_arr[:min_l] == extr_arr[:min_l])
                            / max(len(orig_arr), len(extr_arr)) * 100
                        )
                    else:
                        ber, rec_acc = 1.0, 0.0

                # Cover recovery accuracy
                if recovered is not None and recovered.shape == cover.shape:
                    cov_acc = float(np.sum(cover == recovered) / cover.size * 100)
                    rec_ok = bool(cov_acc == 100.0)
                else:
                    cov_acc = float("nan")
                    rec_ok = False

                row.update({
                    "actual_bpp":                 actual_raw_bpp,
                    "actual_raw_bpp":             actual_raw_bpp,
                    "actual_embedded_bpp":        actual_embedded_bpp,
                    "psnr":                       round(psnr_val, 2),
                    "ssim":                       round(ssim_val, 4),
                    "mse":                        round(mse_val, 4),
                    "ber":                        round(ber, 6),
                    "extraction_success":         extr_ok,
                    "recovery_success":           rec_ok,
                    "payload_recovery_%":         round(rec_acc, 2),
                    "cover_recovery_%":           round(cov_acc, 2),
                    "usable_capacity_bits":       int(usable_cap),
                    "theoretical_capacity_bits":  int(theo_cap),
                    "capacity_utilization_%":     float(cap_util),
                    "embed_time_s":               round(embed_time, 4),
                    "extract_time_s":             round(extract_time, 4),
                    "max_capacity_bits":          int(usable_cap),
                    "status":                     "success",
                })
            except Exception as exc:
                row.update({
                    "actual_bpp":                 float("nan"),
                    "actual_raw_bpp":             float("nan"),
                    "actual_embedded_bpp":        float("nan"),
                    "psnr":                       float("nan"),
                    "ssim":                       float("nan"),
                    "mse":                        float("nan"),
                    "ber":                        float("nan"),
                    "extraction_success":         False,
                    "recovery_success":           False,
                    "payload_recovery_%":         float("nan"),
                    "cover_recovery_%":           float("nan"),
                    "usable_capacity_bits":       0,
                    "theoretical_capacity_bits":  0,
                    "capacity_utilization_%":     0.0,
                    "embed_time_s":               float("nan"),
                    "extract_time_s":             float("nan"),
                    "max_capacity_bits":          0,
                    "status":                     f"FAILED: {str(exc)[:120]}",
                })

            rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── Summary statistics per BPP level ─────────────────────────────────────
    stat_rows = []
    for bpp in bpp_levels:
        sub = results_df[results_df["target_bpp"] == bpp]
        sr: Dict[str, Any] = {
            "target_bpp": bpp,
            "n_images":   len(sub),
            "n_success":  int((sub["status"] == "success").sum()),
        }
        for col in ("psnr", "ssim", "mse", "ber", "payload_recovery_%"):
            vals = sub[col].dropna()
            if len(vals):
                sr[f"{col}_mean"] = round(float(vals.mean()), 4)
                sr[f"{col}_std"]  = round(float(vals.std()),  4)
                sr[f"{col}_min"]  = round(float(vals.min()),  4)
                sr[f"{col}_max"]  = round(float(vals.max()),  4)
            else:
                sr[f"{col}_mean"] = sr[f"{col}_std"] = sr[f"{col}_min"] = sr[f"{col}_max"] = float("nan")
        stat_rows.append(sr)

    stats_df = pd.DataFrame(stat_rows)
    figures  = _make_capacity_figures(results_df, bpp_levels)
    return results_df, stats_df, figures


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _make_capacity_figures(df: pd.DataFrame, bpp_levels: List[float]) -> List[plt.Figure]:
    """Return 4 matplotlib figures: BPP vs PSNR, SSIM, MSE, BER."""
    ok = df[df["status"] == "success"]
    agg_rows = []
    for bpp in bpp_levels:
        sub = ok[ok["target_bpp"] == bpp]
        if not sub.empty:
            agg_rows.append({
                "bpp":  bpp,
                "psnr": sub["psnr"].mean(),
                "ssim": sub["ssim"].mean(),
                "mse":  sub["mse"].mean(),
                "ber":  sub["ber"].mean(),
            })
    if not agg_rows:
        return []

    agg = pd.DataFrame(agg_rows)
    spec = [
        ("psnr", "PSNR (dB)",   "BPP vs PSNR",   "#818cf8"),
        ("ssim", "SSIM",        "BPP vs SSIM",   "#c084fc"),
        ("mse",  "MSE",         "BPP vs MSE",    "#f472b6"),
        ("ber",  "BER",         "BPP vs BER",    "#fb923c"),
    ]
    figs = []
    for col, ylabel, title, color in spec:
        if col not in agg or agg[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(agg["bpp"], agg[col], marker="o", color=color,
                lw=2.5, ms=8, label="CNN-DA-EMD-OLSB (mean)")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("BPP (Bits Per Pixel)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, ls="--", alpha=0.45)
        ax.legend(fontsize=9)
        plt.tight_layout()
        figs.append(fig)
    return figs


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_capacity_results(
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    figures: List[plt.Figure],
    base_dir: str = "results/payload",
) -> str:
    """Save to timestamped sub-directory; returns path."""
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)
    results_df.to_csv(os.path.join(out, "payload_capacity_results.csv"), index=False)
    stats_df.to_csv(os.path.join(out, "payload_capacity_stats.csv"),   index=False)
    names = ["BPP_vs_PSNR", "BPP_vs_SSIM", "BPP_vs_MSE", "BPP_vs_BER"]
    for i, fig in enumerate(figures):
        nm = names[i] if i < len(names) else f"figure_{i+1}"
        fig.savefig(os.path.join(out, f"{nm}.png"), dpi=150, bbox_inches="tight")
    return out


def get_capacity_zip(
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    figures: List[plt.Figure],
) -> bytes:
    """Return all results as a ZIP archive in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("payload_capacity_results.csv", results_df.to_csv(index=False))
        zf.writestr("payload_capacity_stats.csv",   stats_df.to_csv(index=False))
        names = ["BPP_vs_PSNR", "BPP_vs_SSIM", "BPP_vs_MSE", "BPP_vs_BER"]
        for i, fig in enumerate(figures):
            fb = io.BytesIO()
            fig.savefig(fb, format="png", dpi=150, bbox_inches="tight")
            nm = names[i] if i < len(names) else f"figure_{i+1}"
            zf.writestr(f"{nm}.png", fb.getvalue())
    return buf.getvalue()
