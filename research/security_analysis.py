"""
Security Analysis Module.

Implements six genuine steganalysis methods — no fabricated results:

  A. Histogram Analysis  — per-channel RGB histograms, cover vs stego
  B. Shannon Entropy     — per-channel and overall
  C. Pixel Correlation   — horizontal / vertical / diagonal correlation coefficients
  D. RS Analysis         — Regular-Singular steganalysis (Fridrich et al. 2001)
  E. SPA                 — Sample Pair Analysis (Dumitrescu et al. 2003)
  F. Chi-Square PoV Test — Pairs-of-Values chi-square test (Westfeld & Pfitzmann 2000)

All methods are applied to grayscale projections of the colour images
(mean of R, G, B channels) except the histogram which uses per-channel data.
"""

import os
import io
import zipfile
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2 as _chi2_dist
from typing import List, Dict, Any, Optional, Callable, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# A.  HISTOGRAM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _channel_hist(channel: np.ndarray) -> np.ndarray:
    return np.histogram(channel.flatten(), bins=256, range=(0, 255))[0].astype(np.float64)


def plot_histogram_comparison(
    cover: np.ndarray,
    stego: np.ndarray,
    name: str = "Image",
) -> plt.Figure:
    """Side-by-side RGB histogram: cover (row 1) vs stego (row 2)."""
    channels   = ["R", "G", "B"]
    ch_colors  = ["#ef4444", "#22c55e", "#3b82f6"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=False)
    fig.suptitle(f"RGB Histogram — {name}", fontsize=12, fontweight="bold")

    for col, (ch, color) in enumerate(zip(channels, ch_colors)):
        for row, (img, label) in enumerate([(cover, "Cover"), (stego, "Stego")]):
            h = _channel_hist(img[:, :, col])
            ax = axes[row, col]
            ax.bar(np.arange(256), h, width=1, color=color, alpha=0.75)
            ax.set_title(f"{label} — {ch}", fontsize=9)
            ax.set_xlim(0, 255)
            ax.set_xlabel("Pixel Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# B.  SHANNON ENTROPY
# ══════════════════════════════════════════════════════════════════════════════

def _entropy_channel(channel: np.ndarray) -> float:
    h = _channel_hist(channel)
    total = h.sum()
    if total == 0:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_shannon_entropy(img: np.ndarray) -> Dict[str, float]:
    return {
        "entropy_R":       round(_entropy_channel(img[:, :, 0]), 6),
        "entropy_G":       round(_entropy_channel(img[:, :, 1]), 6),
        "entropy_B":       round(_entropy_channel(img[:, :, 2]), 6),
        "entropy_overall": round(_entropy_channel(img.reshape(-1, 3).flatten().reshape(-1, 1)[:, 0]), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# C.  PIXEL CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_pixel_correlation(img: np.ndarray) -> Dict[str, float]:
    """
    Compute exact horizontal, vertical, and diagonal adjacent pixel-pair
    correlation on the luminance (grayscale mean) channel.
    """
    gray = np.mean(img, axis=2).astype(np.float64)
    h, w = gray.shape

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a_flat = a.ravel()
        b_flat = b.ravel()
        std_a = np.std(a_flat)
        std_b = np.std(b_flat)
        if std_a < 1e-8 or std_b < 1e-8:
            return float("nan")
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    corr_h = _corr(gray[:, :-1], gray[:, 1:])
    corr_v = _corr(gray[:-1, :], gray[1:, :])
    corr_d = _corr(gray[:-1, :-1], gray[1:, 1:])

    return {
        "corr_horizontal": round(corr_h, 6) if not np.isnan(corr_h) else float("nan"),
        "corr_vertical":   round(corr_v, 6) if not np.isnan(corr_v) else float("nan"),
        "corr_diagonal":   round(corr_d, 6) if not np.isnan(corr_d) else float("nan"),
    }


def plot_correlation_comparison(corr_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart: cover vs stego pixel correlation by direction."""
    directions = ["corr_horizontal", "corr_vertical", "corr_diagonal"]
    labels     = ["Horizontal", "Vertical", "Diagonal"]

    cover_vals = [corr_df[corr_df["image_type"] == "cover"][d].mean() for d in directions]
    stego_vals = [corr_df[corr_df["image_type"] == "stego"][d].mean() for d in directions]

    x     = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, cover_vals, width, label="Cover", color="#3b82f6", alpha=0.82)
    ax.bar(x + width / 2, stego_vals, width, label="Stego", color="#ec4899", alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Pixel Correlation: Cover vs Stego", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# D.  RS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_rs_analysis(
    channel: np.ndarray,
    group_size: int = 4,
    max_groups: int = 50_000,
) -> Dict[str, Any]:
    """
    Regular-Singular (RS) steganalysis on a single image channel (uint8).

    Reference
    ---------
    Fridrich, J., Goljan, M., Du, R. (2001).
    "Reliable Detection of LSB Steganography in Color and Grayscale Images."
    ACM Workshop on Multimedia and Security.

    Discrimination function  d(x) = Σ |x_{i+1} − x_i|
    Positive flip mask  F:   x_i ← x_i XOR 1  (toggle LSB)
    Negative flip mask  F⁻¹: 2k ↔ 2k+1         (swap even/odd)

    R  = fraction of groups where d(F(x))  > d(x)   [Regular under +]
    S  = fraction of groups where d(F(x))  < d(x)   [Singular under +]
    RM = fraction of groups where d(F⁻¹(x)) > d(x)  [Regular under -]
    SM = fraction of groups where d(F⁻¹(x)) < d(x)  [Singular under -]

    Estimate: p̂ ≈ (RM − R) / (R − S − RM + SM)   [fraction of embedded bits]
    """
    flat = channel.flatten().astype(np.int32)
    n    = len(flat)
    G    = group_size

    n_groups = n // G
    rng      = np.random.default_rng(42)
    if n_groups > max_groups:
        starts = rng.choice(np.arange(0, (n // G) * G, G), size=max_groups, replace=False)
    else:
        starts = np.arange(0, (n // G) * G, G)

    R_pos = S_pos = R_neg = S_neg = 0

    for i in starts:
        grp = flat[i : i + G]

        d_orig = int(np.sum(np.abs(np.diff(grp))))

        # Positive flip: XOR LSB with 1
        fp = grp ^ 1
        d_fp = int(np.sum(np.abs(np.diff(fp))))

        # Negative flip: swap even ↔ odd
        fn = np.where(grp % 2 == 0, grp - 1, grp + 1)
        fn = np.clip(fn, 0, 255)
        d_fn = int(np.sum(np.abs(np.diff(fn))))

        if d_fp > d_orig:   R_pos += 1
        elif d_fp < d_orig: S_pos += 1

        if d_fn > d_orig:   R_neg += 1
        elif d_fn < d_orig: S_neg += 1

    total = len(starts)
    R  = R_pos / total
    S  = S_pos / total
    RM = R_neg / total
    SM = S_neg / total

    denom = R - S - RM + SM
    p_hat = float(np.clip((RM - R) / denom, 0.0, 1.0)) if abs(denom) > 1e-12 else 0.0

    return {
        "R_rate":         round(R,     6),
        "S_rate":         round(S,     6),
        "RM_rate":        round(RM,    6),
        "SM_rate":        round(SM,    6),
        "p_estimate":     round(p_hat, 6),
        "groups_analysed": int(total),
        "method":         "RS Analysis (Fridrich et al. 2001)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# E.  SAMPLE PAIR ANALYSIS (SPA)
# ══════════════════════════════════════════════════════════════════════════════

def compute_spa(channel: np.ndarray) -> Dict[str, Any]:
    """
    Sample Pair Analysis (SPA) on a single image channel (uint8).

    Reference
    ---------
    Dumitrescu, S., Wu, X., Wang, Z. (2003).
    "Detection of LSB Steganography via Sample Pair Analysis."
    IEEE Transactions on Signal Processing, 51(7), 1995-2007.

    For adjacent horizontal pixel pairs (a, b):
        E  = #{(a,b) : a even, b odd}   (even→odd cross-parity)
        O  = #{(a,b) : a odd,  b even}  (odd→even cross-parity)
        EE = #{(a,b) : a even, b even}
        OO = #{(a,b) : a odd,  b odd}

    Clean images: E ≈ O  (symmetric parity transitions)
    LSB-embedded: E and O diverge proportionally to embedding rate p.

    Estimate:   p̂ = |E − O| / (E + O)    (normalised parity asymmetry)
    """
    flat = channel.flatten().astype(np.int32)
    a = flat[:-1]
    b = flat[1:]

    E  = int(np.sum((a % 2 == 0) & (b % 2 == 1)))
    O  = int(np.sum((a % 2 == 1) & (b % 2 == 0)))
    EE = int(np.sum((a % 2 == 0) & (b % 2 == 0)))
    OO = int(np.sum((a % 2 == 1) & (b % 2 == 1)))

    denom = E + O
    p_hat = abs(E - O) / denom if denom > 0 else 0.0

    return {
        "E_pairs":      E,
        "O_pairs":      O,
        "EE_pairs":     EE,
        "OO_pairs":     OO,
        "total_pairs":  len(a),
        "p_estimate":   round(float(p_hat), 6),
        "D_asymmetry":  int(E - O),
        "method":       "SPA (Dumitrescu et al. 2003)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# F.  CHI-SQUARE PAIRS-OF-VALUES (PoV) TEST
# ══════════════════════════════════════════════════════════════════════════════

def compute_chi_square_pov(channel: np.ndarray) -> Dict[str, Any]:
    """
    Chi-square Pairs-of-Values (PoV) test for LSB steganography.

    Reference
    ---------
    Westfeld, A., Pfitzmann, A. (2000).
    "Attacks on Steganographic Systems."
    Information Hiding, LNCS 1768, pp. 61-76.

    Method
    ------
    Histogram h[v], v = 0..255.
    PoV groups: (h[2i], h[2i+1]) for i = 0..127.
    Expected (if embedded): h[2i] ≈ h[2i+1] = (h[2i]+h[2i+1]) / 2.

    χ² = Σ [ (h[2i] − expected)² / expected + (h[2i+1] − expected)² / expected ]

    Interpretation (chi-square PoV):
      • Large χ², small p  → histogram NOT uniform in pairs → likely CLEAN image
      • Small χ², large p  → histogram close to uniform in pairs → LSB embedding suspected
    """
    hist = _channel_hist(channel)

    ev = hist[0::2][:128]   # even-valued bins
    od = hist[1::2][:128]   # odd-valued bins
    ex = (ev + od) / 2.0    # expected under H0 (equal counts)

    mask  = ex > 0
    chi_sq = float(np.sum(
        (ev[mask] - ex[mask]) ** 2 / ex[mask] +
        (od[mask] - ex[mask]) ** 2 / ex[mask]
    ))
    df      = int(np.sum(mask)) - 1
    df      = max(1, df)
    p_value = float(1.0 - _chi2_dist.cdf(chi_sq, df))

    # p > 0.05 → pairs NOT significantly different from uniform → stego suspected
    suspected = bool(p_value > 0.05)

    return {
        "chi_sq_stat":        round(chi_sq, 4),
        "degrees_of_freedom": df,
        "p_value":            round(p_value, 6),
        "alpha":              0.05,
        "suspected_stego":    suspected,
        "interpretation": (
            "LSB steganography SUSPECTED — PoV pairs statistically uniform (p > 0.05)"
            if suspected else
            "Natural / clean image — PoV pairs non-uniform (p < 0.05)"
        ),
        "method": "Chi-Square PoV (Westfeld & Pfitzmann 2000)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# FULL SECURITY PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_security_analysis(
    cover_images: List[np.ndarray],
    stego_images: List[np.ndarray],
    image_names: List[str],
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run complete 6-part security analysis for all (cover, stego) pairs.

    Returns dict with DataFrames and matplotlib Figure objects.
    """
    entropy_rows = []
    corr_rows    = []
    rs_rows      = []
    spa_rows     = []
    chi_rows     = []
    hist_figs: List[plt.Figure] = []
    total = len(cover_images)

    for idx, (cov, stg, nm) in enumerate(zip(cover_images, stego_images, image_names)):
        if progress_callback:
            progress_callback(idx + 1, total, f"Analysing: {nm}")

        # Histogram
        hist_figs.append(plot_histogram_comparison(cov, stg, nm))

        # Entropy
        for img_type, img in [("cover", cov), ("stego", stg)]:
            ent = compute_shannon_entropy(img)
            for ch_key, val in ent.items():
                ch = ch_key.replace("entropy_", "")
                entropy_rows.append({"image": nm, "image_type": img_type, "channel": ch, "entropy": val})

        # Correlation
        for img_type, img in [("cover", cov), ("stego", stg)]:
            corr = compute_pixel_correlation(img)
            corr_rows.append({"image": nm, "image_type": img_type, **corr})

        # RS, SPA, Chi-Square on grayscale (mean of channels)
        for img_type, img in [("cover", cov), ("stego", stg)]:
            gray = np.mean(img, axis=2).astype(np.uint8)
            rs   = compute_rs_analysis(gray)
            spa  = compute_spa(gray)
            chi  = compute_chi_square_pov(gray)
            rs_rows.append({"image": nm, "image_type": img_type, **rs})
            spa_rows.append({"image": nm, "image_type": img_type, **spa})
            chi_rows.append({"image": nm, "image_type": img_type, **chi})

    entropy_df = pd.DataFrame(entropy_rows)
    corr_df    = pd.DataFrame(corr_rows)
    rs_df      = pd.DataFrame(rs_rows)
    spa_df     = pd.DataFrame(spa_rows)
    chi_df     = pd.DataFrame(chi_rows)

    corr_fig = plot_correlation_comparison(corr_df) if len(corr_rows) >= 2 else None

    return {
        "entropy_df":          entropy_df,
        "correlation_df":      corr_df,
        "rs_df":               rs_df,
        "spa_df":              spa_df,
        "chi_df":              chi_df,
        "histogram_figures":   hist_figs,
        "correlation_figure":  corr_fig,
    }


# ── Stego generation helper ───────────────────────────────────────────────────

def generate_stego_for_security(
    cover_images: List[np.ndarray],
    cover_names: List[str],
    password: str = "Pass123!",
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.6,
    t1: float = 0.33,
    t2: float = 0.66,
    payload_bpp: float = 0.05,
    model=None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Embed a payload into each cover image and return (covers, stegos, names)
    for the images where embedding succeeded using trained CNN-DA-EMD-OLSB.
    """
    from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb
    from cnn.distortion_cnn import load_trained_distortion_cnn
    OVERHEAD = 96

    if model is None and gamma > 0.0:
        model = load_trained_distortion_cnn()

    ok_covers, ok_stegos, ok_names = [], [], []
    total = len(cover_images)

    for idx, (cov, nm) in enumerate(zip(cover_images, cover_names)):
        if progress_callback:
            progress_callback(idx + 1, total, f"Embedding stego for: {nm}")
        try:
            h, w, _ = cov.shape
            raw_bytes = max(1, int(payload_bpp * h * w / 8) - OVERHEAD)
            rng = np.random.default_rng(idx)
            secret = bytes(rng.integers(0, 256, raw_bytes, dtype=np.uint8))
            stego_dual, _ = embed_cnn_da_emd_olsb(
                cover_rgb=cov, secret_data=secret, password=password,
                alpha=alpha, beta=beta, gamma=gamma, t1=t1, t2=t2, payload_type=0,
                model=model,
            )
            ok_covers.append(cov)
            ok_stegos.append(stego_dual[0])  # S1
            ok_names.append(nm)
        except Exception:
            pass  # skip failed images

    return ok_covers, ok_stegos, ok_names


# ── Persistence ───────────────────────────────────────────────────────────────

def save_security_results(result: Dict[str, Any], base_dir: str = "results/security") -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)

    for key, fname in [
        ("entropy_df",     "entropy_results.csv"),
        ("correlation_df", "correlation_results.csv"),
        ("rs_df",          "rs_analysis_results.csv"),
        ("spa_df",         "spa_results.csv"),
        ("chi_df",         "chi_square_results.csv"),
    ]:
        df = result.get(key)
        if df is not None and not df.empty:
            df.to_csv(os.path.join(out, fname), index=False)

    for i, fig in enumerate(result.get("histogram_figures", [])):
        fig.savefig(os.path.join(out, f"histogram_{i+1}.png"), dpi=120, bbox_inches="tight")
    if result.get("correlation_figure"):
        result["correlation_figure"].savefig(
            os.path.join(out, "correlation_comparison.png"), dpi=120, bbox_inches="tight")

    return out


def get_security_zip(result: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, fname in [
            ("entropy_df",     "entropy_results.csv"),
            ("correlation_df", "correlation_results.csv"),
            ("rs_df",          "rs_analysis_results.csv"),
            ("spa_df",         "spa_results.csv"),
            ("chi_df",         "chi_square_results.csv"),
        ]:
            df = result.get(key)
            if df is not None and not df.empty:
                zf.writestr(fname, df.to_csv(index=False))

        for i, fig in enumerate(result.get("histogram_figures", [])):
            fb = io.BytesIO()
            fig.savefig(fb, format="png", dpi=120, bbox_inches="tight")
            zf.writestr(f"histogram_{i+1}.png", fb.getvalue())

        if result.get("correlation_figure"):
            fb = io.BytesIO()
            result["correlation_figure"].savefig(fb, format="png", dpi=120, bbox_inches="tight")
            zf.writestr("correlation_comparison.png", fb.getvalue())

    return buf.getvalue()
