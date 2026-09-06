"""
Unified Benchmark Runner Engine for the 6-Model Comparative Research Framework.
Runs MPEH-RDH, MCSH-RDH, CNN-RDH Predictor, SRDNN-Stego, EMD-OLSB RDH, and
Proposed Model: CNN-DA-EMD-OLSB (CNN-Guided Distortion-Aware Adaptive EMD-OLSB).
"""

import time
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union

from models.adapters import (
    MPEHAdapter,
    MCSHAdapter,
    CNNRDHAdapter,
    SRDNNAdapter,
    EMDOLSBAdapter,
    CNNDAEMDOLSBAdapter
)
from benchmark.metrics import evaluate_model_performance
from utils.image_utils import load_image_rgb, resize_image


class BenchmarkRunner:
    """
    Executes 6-Model Comparative Research Benchmark.
    """
    def __init__(self):
        self.adapters = {
            'MPEH-RDH':           MPEHAdapter(),
            'MCSH-RDH':           MCSHAdapter(),
            'CNN-RDH Predictor':  CNNRDHAdapter(),
            'SRDNN-Stego':        SRDNNAdapter(),
            'EMD-OLSB RDH':       EMDOLSBAdapter(),
            'CNN-DA-EMD-OLSB':    CNNDAEMDOLSBAdapter()
        }

    def run_single_model(
        self,
        model_name: str,
        cover_rgb: np.ndarray,
        payload_data: Union[bytes, str],
        password: str = "Pass123!",
        payload_type: int = 0
    ) -> Dict[str, Any]:
        """
        Runs embedding and extraction pipeline for a single model.
        """
        if model_name not in self.adapters:
            raise ValueError(f"Unknown model name: {model_name}. Choose from {list(self.adapters.keys())}")

        adapter = self.adapters[model_name]
        
        # Support high-capacity cover images (up to 2048x2048)
        h, w = cover_rgb.shape[:2]
        if max(h, w) > 2048:
            cover_rgb = resize_image(cover_rgb, (2048, 2048))

        payload_bytes = payload_data.encode('utf-8') if isinstance(payload_data, str) else payload_data

        # 1. Embed Phase
        t0 = time.time()
        if model_name == "CNN-DA-EMD-OLSB":
            stego_output, stats = adapter.embed(cover_rgb, payload_bytes, password=password, payload_type=payload_type)
        else:
            stego_output, stats = adapter.embed(cover_rgb, payload_bytes, password=password)
        embed_time = time.time() - t0

        # 2. Extract Phase
        t1 = time.time()
        extracted_bytes, recovered_cover = adapter.extract(stego_output, stats, password=password)
        extract_time = time.time() - t1

        total_bits = stats.get('actual_embedded_bits', stats.get('total_bits_embedded', len(payload_bytes) * 8))
        is_dual = stats.get('dual_images', False)

        metrics = evaluate_model_performance(
            cover=cover_rgb,
            stego_output=stego_output,
            total_embedded_bits=total_bits,
            original_payload=payload_bytes,
            extracted_payload=extracted_bytes,
            recovered_cover=recovered_cover,
            embed_time=embed_time,
            extract_time=extract_time,
            is_dual_stego=is_dual
        )

        metrics['Model'] = model_name
        metrics['Cover_RGB'] = cover_rgb
        metrics['Stego_Output'] = stego_output
        metrics['Recovered_Cover'] = recovered_cover
        metrics['Extracted_Payload'] = extracted_bytes
        metrics['Stats'] = stats
        return metrics

    def run_all_models(
        self,
        cover_rgb: np.ndarray,
        payload_data: Union[bytes, str],
        password: str = "Pass123!"
    ) -> pd.DataFrame:
        """
        Runs comparative benchmark across ALL 6 research models.
        """
        results = []
        for model_name in self.adapters.keys():
            try:
                res = self.run_single_model(model_name, cover_rgb, payload_data, password=password)
                results.append(res)
            except Exception as e:
                print(f"Error running {model_name}: {str(e)}")

        df = pd.DataFrame(results)
        
        # Save CSV output
        os.makedirs("results/csv", exist_ok=True)
        csv_path = "results/csv/6_model_benchmark_results.csv"
        export_df = df.drop(columns=['Stego_Output', 'Recovered_Cover', 'Extracted_Payload', 'Stats', 'Cover_RGB'], errors='ignore')
        export_df.to_csv(csv_path, index=False)
        
        return df

    def get_compatibility_protocol_table(self) -> pd.DataFrame:
        """
        Returns Compatibility / Experimental Protocol Summary Table across all 6 models.
        """
        protocols = [adapter.get_compatibility_protocol() for adapter in self.adapters.values()]
        return pd.DataFrame(protocols)
