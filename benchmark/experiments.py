"""
Payload-Level Comparative Experiments & Research Plot Generator.
Runs benchmarks across Low (256B), Medium (1KB), and High (4KB) payloads.
Generates research publication quality graphs: PSNR vs Payload, SSIM vs Payload, BER vs Payload, Time vs Payload.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from benchmark.runner import BenchmarkRunner


def run_payload_experiments(cover_rgb: np.ndarray, password: str = "Pass123!") -> Dict[str, pd.DataFrame]:
    """
    Executes 6-model benchmarks across Low, Medium, and High payload sizes.
    """
    runner = BenchmarkRunner()

    payload_levels = {
        'Low Payload (256 B)': "A" * 256,
        'Medium Payload (1 KB)': "B" * 1024,
        'High Payload (4 KB)': "C" * 4096
    }

    results_by_level = {}

    for level_name, payload_str in payload_levels.items():
        df = runner.run_all_models(cover_rgb, payload_str, password=password)
        df['Payload_Level'] = level_name
        results_by_level[level_name] = df

    return results_by_level


def generate_research_graphs(results_by_level: Dict[str, pd.DataFrame], output_dir: str = "results/graphs"):
    """
    Generates 4 research publication plots saved to results/graphs/:
    1. PSNR vs Payload Level
    2. SSIM vs Payload Level
    3. Embedding Time vs Payload Level
    4. BER vs Payload Level
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_df = pd.concat(results_by_level.values(), ignore_index=True)

    levels = list(results_by_level.keys())
    models = combined_df['Model'].unique()

    # Colors: Proposed model highlighted in Violet (#8b5cf6), baselines in distinct hues
    color_map = {
        'MPEH-RDH': '#3b82f6',
        'MCSH-RDH': '#06b6d4',
        'CNN-RDH Predictor': '#10b981',
        'SRDNN-Stego': '#f59e0b',
        'EMD-OLSB RDH': '#ec4899',
        'CNN-DA-EMD-OLSB': '#8b5cf6'
    }

    # 1. PSNR vs Payload Level
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in models:
        sub = combined_df[combined_df['Model'] == model]
        ax.plot(sub['Payload_Level'], sub['PSNR_dB'], marker='o', label=model, color=color_map.get(model, '#333'), linewidth=2.5)

    ax.set_title("PSNR (dB) vs. Payload Level Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("Payload Level")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psnr_vs_payload.png"), dpi=300)
    plt.close()

    # 2. SSIM vs Payload Level
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in models:
        sub = combined_df[combined_df['Model'] == model]
        ax.plot(sub['Payload_Level'], sub['SSIM'], marker='s', label=model, color=color_map.get(model, '#333'), linewidth=2.5)

    ax.set_title("SSIM vs. Payload Level Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel("SSIM Index")
    ax.set_xlabel("Payload Level")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ssim_vs_payload.png"), dpi=300)
    plt.close()

    # 3. Embedding Time vs Payload Level
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in models:
        sub = combined_df[combined_df['Model'] == model]
        ax.plot(sub['Payload_Level'], sub['Embed_Time_s'], marker='^', label=model, color=color_map.get(model, '#333'), linewidth=2.5)

    ax.set_title("Embedding Time (s) vs. Payload Level Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel("Embedding Time (Seconds)")
    ax.set_xlabel("Payload Level")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embed_time_vs_payload.png"), dpi=300)
    plt.close()

    # 4. BER vs Payload Level
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in models:
        sub = combined_df[combined_df['Model'] == model]
        ax.plot(sub['Payload_Level'], sub['BER'], marker='D', label=model, color=color_map.get(model, '#333'), linewidth=2.5)

    ax.set_title("Bit Error Rate (BER) vs. Payload Level Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel("BER (lower is better)")
    ax.set_xlabel("Payload Level")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ber_vs_payload.png"), dpi=300)
    plt.close()

    print(f"✅ Research Graphs saved successfully to {output_dir}/")
