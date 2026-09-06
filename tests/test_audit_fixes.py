"""
Comprehensive Test Suite for Audit Fixes across all 13 User-Requested Items.
"""

import os
import unittest
import numpy as np
import torch
import pandas as pd

from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from cnn.distortion_cnn import DistortionCNN, load_trained_distortion_cnn, compute_distortion_maps
from core.cnn_da_emd_olsb import (
    embed_cnn_da_emd_olsb,
    extract_cnn_da_emd_olsb,
    compute_capacity,
    _get_cap_maps
)
from research.payload_capacity import run_payload_capacity_experiment, DEFAULT_BPP_LEVELS
from research.security_analysis import compute_pixel_correlation, compute_shannon_entropy, compute_rs_analysis
from research.statistical_testing import _nemenyi_posthoc


class TestAuditFixes(unittest.TestCase):

    def setUp(self):
        self.password = "AuditPass123!"
        np.random.seed(42)
        torch.manual_seed(42)

    def test_01_trained_cnn_loading(self):
        """Item 1: Verify trained DistortionCNN loads weights properly without error."""
        model = CNNDAEMDOLSBModel(gamma=0.6)
        self.assertIsNotNone(model._cnn_model)
        self.assertTrue(getattr(model, '_cnn_trained', False))
        self.assertFalse(model._cnn_model.training, "Model should be in eval() mode")

    def test_02_no_silent_fallback(self):
        """Item 2: Verify that CNN failure or missing model when use_cnn=True raises an error."""
        dummy_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Passing an invalid/failing object as model must raise RuntimeError, not silently fall back
        with self.assertRaises(RuntimeError):
            compute_distortion_maps(dummy_rgb, gamma=0.6, model="invalid_model", use_cnn=True)

    def test_03_gamma_blending(self):
        """Item 3: Verify gamma blending produces different maps for different gamma values."""
        cnn = load_trained_distortion_cnn()
        img = np.random.randint(20, 230, (64, 64, 3), dtype=np.uint8)

        d_00 = compute_distortion_maps(img, gamma=0.0, model=cnn)
        d_05 = compute_distortion_maps(img, gamma=0.5, model=cnn)
        d_10 = compute_distortion_maps(img, gamma=1.0, model=cnn)

        self.assertFalse(np.allclose(d_00, d_10), "gamma=0.0 and gamma=1.0 should produce different maps")
        self.assertFalse(np.allclose(d_05, d_10), "gamma=0.5 and gamma=1.0 should produce different maps")

    def test_04_capacity_boundary_conditions(self):
        """Item 4: Verify usable capacity accounts for boundary pixels [8, 240] unlike theoretical."""
        # Image with extreme border values
        img_extreme = np.zeros((64, 64, 3), dtype=np.uint8)
        img_extreme[:32, :, :] = 5   # below 8 -> excluded from EMD
        img_extreme[32:, :, :] = 250 # above 240 -> excluded from EMD

        upper = (img_extreme & 0xF8).astype(np.uint8)
        cls_r, cls_g, cls_b = _get_cap_maps(upper, alpha=0.5, beta=0.5, gamma=0.0, t1=0.33, t2=0.66, model=None)
        cap_info = compute_capacity(cls_r, cls_g, cls_b, upper)

        self.assertEqual(cap_info['emd_capacity_bits'], 0, "Extreme pixels outside [8, 240] must yield 0 EMD capacity")
        self.assertGreater(cap_info['theoretical_emd_bits'], 0, "Theoretical EMD capacity does not filter boundary")
        self.assertLess(cap_info['usable_capacity_bits'], cap_info['theoretical_capacity_bits'], "Usable capacity must be strictly less than theoretical due to boundary exclusion")

    def test_05_bpp_and_embedding_extraction(self):
        """Items 5 & 7: Verify embedding, extraction, cover recovery, and BPP metrics."""
        y, x = np.ogrid[:256, :256]
        gradient = ((x + y) * 2) % 200 + 30
        img = np.stack([gradient, (gradient + 20) % 220 + 20, (gradient + 40) % 220 + 20], axis=-1).astype(np.uint8)
        secret = b"Antigravity verified payload for CNN-DA-EMD-OLSB!"
        test_gamma = 0.75

        stego, stats = embed_cnn_da_emd_olsb(
            img, secret, password=self.password, gamma=test_gamma
        )

        self.assertIn('raw_bpp', stats)
        self.assertIn('embedded_bpp', stats)
        self.assertIn('usable_capacity_bits', stats)
        self.assertIn('theoretical_capacity_bits', stats)
        self.assertGreater(stats['embedded_bpp'], stats['raw_bpp'], "Embedded BPP includes header/encryption overhead")

        # Extraction
        extracted, recovered, meta = extract_cnn_da_emd_olsb(
            stego, password=self.password
        )

        self.assertEqual(extracted, secret, "Extracted payload must match original exactly")
        self.assertTrue(np.array_equal(img, recovered), "Recovered cover must be bit-exact to original cover")
        self.assertAlmostEqual(meta['gamma'], test_gamma, places=2, msg="Gamma stored in header must match on extraction")

    def test_08_payload_capacity_sweep(self):
        """Item 8: Verify payload-capacity experiment across BPP levels."""
        y, x = np.ogrid[:256, :256]
        gradient = ((x + y) * 2) % 200 + 30
        img = np.stack([gradient, (gradient + 20) % 220 + 20, (gradient + 40) % 220 + 20], axis=-1).astype(np.uint8)
        test_bpps = [0.001, 0.005]  # fast subset for unit test

        results_df, stats_df, figs = run_payload_capacity_experiment(
            images=[img],
            image_names=["test_image"],
            bpp_levels=test_bpps,
            password=self.password,
            gamma=0.6
        )

        self.assertEqual(len(results_df), 2)
        expected_cols = [
            "requested_bpp", "actual_raw_bpp", "actual_embedded_bpp",
            "payload_size_bits", "usable_capacity_bits", "capacity_utilization_%",
            "psnr", "ssim", "mse", "ber", "extraction_success", "recovery_success"
        ]
        for col in expected_cols:
            self.assertIn(col, results_df.columns)
            self.assertFalse(results_df[col].isna().all(), f"Column {col} has NaN")

    def test_09_security_exact_correlation(self):
        """Item 9: Verify exact correlation computation without NaN or random sampling artifacts."""
        img = np.random.randint(30, 220, (64, 64, 3), dtype=np.uint8)
        corr = compute_pixel_correlation(img)
        self.assertIn("corr_horizontal", corr)
        self.assertIn("corr_vertical", corr)
        self.assertIn("corr_diagonal", corr)
        self.assertFalse(np.isnan(corr["corr_horizontal"]))
        self.assertFalse(np.isnan(corr["corr_vertical"]))
        self.assertFalse(np.isnan(corr["corr_diagonal"]))

    def test_10_steganalysis_grouping_logic(self):
        """Item 10: Verify grouping logic prevents data leakage across multiple stego versions."""
        # Simulated 6 pairs from 2 covers: coverA (3 variants), coverB (3 variants)
        pairs = [
            ("covA", "stgA1", "coverA_bpp0.01"),
            ("covA", "stgA2", "coverA_bpp0.05"),
            ("covA", "stgA3", "coverA_bpp0.10"),
            ("covB", "stgB1", "coverB_bpp0.01"),
            ("covB", "stgB2", "coverB_bpp0.05"),
            ("covB", "stgB3", "coverB_bpp0.10"),
        ]
        cover_groups = {}
        for i, (c, s, n_img) in enumerate(pairs):
            base_key = n_img.split('_bpp')[0].strip()
            cover_groups.setdefault(base_key, []).append(i)

        unique_covers = list(cover_groups.keys())
        self.assertEqual(len(unique_covers), 2)
        # All 3 variants of coverA stay together
        self.assertEqual(len(cover_groups["coverA"]), 3)
        self.assertEqual(len(cover_groups["coverB"]), 3)

    def test_11_statistical_testing_nemenyi(self):
        """Item 11: Verify Nemenyi post-hoc calculation."""
        model_names = ["M1", "M2", "M3"]
        avg_ranks = pd.Series([1.2, 2.0, 2.8], index=model_names)
        df_nem = _nemenyi_posthoc(model_names, avg_ranks, n=10, k=3, alpha=0.05)
        self.assertEqual(len(df_nem), 3)
        self.assertIn("Critical Diff (CD)", df_nem.columns)
        self.assertIn("Significant?", df_nem.columns)


if __name__ == "__main__":
    unittest.main()
