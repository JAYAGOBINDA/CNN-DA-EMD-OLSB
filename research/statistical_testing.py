"""
Statistical Testing Module — Friedman Test + Nemenyi Post-Hoc + Kendall's W.

Compares the 6 benchmark models on a chosen quality metric (PSNR / SSIM / MSE)
across a set of test images.  All observations come from actually running the
BenchmarkRunner — no fabricated values.

Statistical procedure
---------------------
1. Friedman χ² test (non-parametric, suitable for k≥2 treatments, n≥3 blocks)
2. Nemenyi post-hoc pairwise comparison (if Friedman p < alpha)
3. Kendall's W effect size (W = χ² / (n·(k−1)))
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
from scipy.stats import friedmanchisquare, norm
from typing import List, Dict, Any, Optional, Callable

from benchmark.runner import BenchmarkRunner

# All 6 model names as registered in the runner
ALL_MODEL_NAMES: List[str] = [
    "MPEH-RDH",
    "MCSH-RDH",
    "CNN-RDH Predictor",
    "SRDNN-Stego",
    "EMD-OLSB RDH",
    "CNN-DA-EMD-OLSB",
]

SUPPORTED_METRICS: List[str] = ["PSNR_dB", "SSIM", "MSE", "BER", "BPP"]


def run_statistical_experiment(
    images: List[np.ndarray],
    image_names: List[str],
    model_names: List[str],
    metric: str = "PSNR_dB",
    password: str = "Pass123!",
    payload_str: str = "A" * 512,
    alpha: float = 0.05,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Run statistical comparison across selected models and images.

    Rows = test images (blocks in the Friedman sense).
    Columns = models (treatments).

    Returns
    -------
    dict with keys:
        observation_matrix  : pd.DataFrame  (images × models, raw metric values)
        friedman_result     : dict | None
        nemenyi_result      : pd.DataFrame | None
        effect_size         : float | None  (Kendall's W)
        effect_size_label   : str
        ranks_df            : pd.DataFrame  (average ranks per model)
        warning             : str | None
        metric              : str
        n_complete_rows     : int
    """
    runner = BenchmarkRunner()
    total = len(model_names) * len(images)
    step  = 0
    obs: Dict[str, List[float]] = {m: [] for m in model_names}

    for model in model_names:
        for img_idx, (cover, name) in enumerate(zip(images, image_names)):
            step += 1
            if progress_callback:
                progress_callback(step, total, f"Model: {model} | Image: {name}")
            try:
                res = runner.run_single_model(
                    model_name=model,
                    cover_rgb=cover,
                    payload_data=payload_str,
                    password=password,
                )
                val = res.get(metric, float("nan"))
                if isinstance(val, str):
                    val = float("nan")
                obs[model].append(float(val))
            except Exception:
                obs[model].append(float("nan"))

    obs_df = pd.DataFrame(obs, index=image_names)
    obs_df.index.name = "image"

    clean = obs_df.dropna()
    n, k  = len(clean), len(model_names)

    result: Dict[str, Any] = {
        "observation_matrix": obs_df,
        "n_complete_rows":    n,
        "metric":             metric,
        "friedman_result":    None,
        "nemenyi_result":     None,
        "effect_size":        None,
        "effect_size_label":  "Kendall's W",
        "ranks_df":           None,
        "warning":            None,
    }

    if n < 3:
        result["warning"] = (
            f"Only {n} complete observation rows (need ≥ 3 for Friedman test). "
            "Upload more images."
        )
        return result
    if k < 3:
        result["warning"] = (
            f"The Friedman test requires at least 3 models to compare (got {k}). "
            "Please select at least 3 models."
        )
        return result

    # ── Friedman Test ─────────────────────────────────────────────────────────
    groups = [clean[m].values for m in model_names]
    stat, p_value = friedmanchisquare(*groups)

    significant = bool(p_value < alpha)
    result["friedman_result"] = {
        "chi2_statistic":     round(float(stat), 4),
        "degrees_of_freedom": int(k - 1),
        "p_value":            round(float(p_value), 6),
        "alpha":              alpha,
        "significant":        significant,
        "decision": (
            f"SIGNIFICANT (p = {p_value:.4f} < {alpha}) — at least one model differs"
            if significant else
            f"NOT SIGNIFICANT (p = {p_value:.4f} ≥ {alpha}) — no evidence of difference"
        ),
    }

    # ── Average ranks ─────────────────────────────────────────────────────────
    # Lower rank = better for MSE/BER; higher rank = better for PSNR/SSIM.
    ascending = metric in ("MSE", "BER")
    ranks = clean.rank(axis=1, ascending=ascending)
    avg_ranks = ranks.mean()
    result["ranks_df"] = pd.DataFrame({
        "Model":        list(avg_ranks.index),
        "Average Rank": [round(v, 4) for v in avg_ranks.values],
    }).sort_values("Average Rank").reset_index(drop=True)

    # ── Kendall's W ───────────────────────────────────────────────────────────
    W = float(stat) / (n * (k - 1))
    result["effect_size"] = round(max(0.0, min(1.0, W)), 4)

    # ── Nemenyi Post-Hoc ──────────────────────────────────────────────────────
    if significant:
        result["nemenyi_result"] = _nemenyi_posthoc(
            model_names, avg_ranks, n, k, alpha
        )

    return result


# ── Nemenyi post-hoc (Dunn's z-based critical difference) ────────────────────

def _nemenyi_posthoc(
    model_names: List[str],
    avg_ranks: "pd.Series",
    n: int,
    k: int,
    alpha: float,
) -> pd.DataFrame:
    """
    Nemenyi pairwise post-hoc test.

    Critical Difference:
        CD = q_α · √(k(k+1) / (6n))
    where q_α is the Studentized range statistic approximated via Bonferroni-
    corrected normal quantile (conservative, standard in literature).
    """
    q_alpha = norm.ppf(1.0 - alpha / (k * (k - 1)))  # two-sided Bonferroni
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            diff = abs(float(avg_ranks[m1]) - float(avg_ranks[m2]))
            sig  = diff > CD
            rows.append({
                "Model A":              m1,
                "Model B":              m2,
                "|Avg Rank Diff|":      round(diff, 4),
                f"Critical Diff (CD)":  round(float(CD), 4),
                "Significant?":         "Yes" if sig else "No",
            })
    return pd.DataFrame(rows)


# ── Visualisation ─────────────────────────────────────────────────────────────

def generate_statistical_figures(result: Dict[str, Any]) -> List[plt.Figure]:
    """Return list of matplotlib figures for the statistical test results."""
    figs = []

    if result.get("ranks_df") is not None:
        ranks_df = result["ranks_df"]
        fig, ax = plt.subplots(figsize=(9, max(3, len(ranks_df) * 0.55)))
        colors = [
            "#8b5cf6" if "CNN-DA-EMD-OLSB" in str(m) else "#3b82f6"
            for m in ranks_df["Model"]
        ]
        bars = ax.barh(ranks_df["Model"], ranks_df["Average Rank"],
                       color=colors, height=0.55)
        ax.set_xlabel("Average Rank (lower = better for PSNR/SSIM)", fontsize=10)
        ax.set_title(
            f"Average Ranks — Friedman Test ({result['metric']})",
            fontsize=11, fontweight="bold",
        )
        ax.grid(True, axis="x", ls="--", alpha=0.4)
        for bar, val in zip(bars, ranks_df["Average Rank"]):
            ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        figs.append(fig)

    return figs


# ── Persistence ───────────────────────────────────────────────────────────────

def save_statistical_results(
    result: Dict[str, Any],
    figs: List[plt.Figure],
    base_dir: str = "results/statistics",
) -> str:
    ts  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(base_dir, ts)
    os.makedirs(out, exist_ok=True)

    result["observation_matrix"].to_csv(os.path.join(out, "observation_matrix.csv"))
    if result["ranks_df"] is not None:
        result["ranks_df"].to_csv(os.path.join(out, "average_ranks.csv"), index=False)
    if result["friedman_result"]:
        pd.DataFrame([result["friedman_result"]]).to_csv(
            os.path.join(out, "friedman_result.csv"), index=False)
    if result["nemenyi_result"] is not None:
        result["nemenyi_result"].to_csv(
            os.path.join(out, "nemenyi_posthoc.csv"), index=False)
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(out, f"figure_{i+1}.png"), dpi=150, bbox_inches="tight")
    return out


def get_statistical_zip(result: Dict[str, Any], figs: List[plt.Figure]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("observation_matrix.csv", result["observation_matrix"].to_csv())
        if result["ranks_df"] is not None:
            zf.writestr("average_ranks.csv", result["ranks_df"].to_csv(index=False))
        if result["friedman_result"]:
            zf.writestr("friedman_result.csv",
                        pd.DataFrame([result["friedman_result"]]).to_csv(index=False))
        if result["nemenyi_result"] is not None:
            zf.writestr("nemenyi_posthoc.csv",
                        result["nemenyi_result"].to_csv(index=False))
        for i, fig in enumerate(figs):
            fb = io.BytesIO()
            fig.savefig(fb, format="png", dpi=150, bbox_inches="tight")
            zf.writestr(f"figure_{i+1}.png", fb.getvalue())
    return buf.getvalue()
