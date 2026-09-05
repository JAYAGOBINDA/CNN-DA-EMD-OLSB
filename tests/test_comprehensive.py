"""
Comprehensive Test Suite for CNN-DA-EMD-OLSB Single-Stego Reversible Data Hiding.

Validates:
1. Exact cover recovery (max_diff == 0, num_diff == 0, np.array_equal == True).
2. EMD mod-5 correctness across all d in {0, 1, 2, 3, 4} and boundary pixels.
3. Gamma fusion: gamma=0.0 vs 0.5 vs 1.0 produce genuine differences.
4. Single-stego image output (shape HxWx3 uint8, no dual tuples).
5. Lossless extraction with 0.0 BER under clean conditions.
6. Real execution metrics without hardcoded numbers.
"""

import unittest
import numpy as np
import torch

from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from cnn.distortion_cnn import (
    DistortionCNN,
    load_trained_distortion_cnn,
    compute_distortion_maps
)
from core.cnn_da_emd_olsb import (
    embed_cnn_da_emd_olsb,
    extract_cnn_da_emd_olsb,
    compute_capacity,
    _get_cap_maps,
    bytes_to_base5_digits,
    base5_digits_to_bytes
)
from benchmark.runner import BenchmarkRunner


class TestComprehensiveSingleStego(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.password = "Research2026Pass!"
        # Create a realistic test image with smooth gradient features
        y, x = np.ogrid[:256, :256]
        g1 = ((x + y) * 2) % 200 + 30
        g2 = (g1 + 20) % 220 + 20
        g3 = (g1 + 40) % 220 + 20
        cls.test_cover = np.stack([g1, g2, g3], axis=-1).astype(np.uint8)

    def test_01_single_stego_exact_recovery(self):
        """Mandate: Single stego image output, extraction success, exact cover recovery (max_diff=0)."""
        model = CNNDAEMDOLSBModel(gamma=0.6, use_cnn=True)
        secret = b"Real verified payload for CNN-DA-EMD-OLSB single-stego research test."

        stego, stats = model.embed(self.test_cover, secret, password=self.password)

        # 1. Output must be a single image, NOT a tuple
        self.assertIsInstance(stego, np.ndarray, "Stego output must be a single np.ndarray")
        self.assertEqual(stego.ndim, 3, "Stego must have 3 dimensions (H, W, C)")
        self.assertEqual(stego.shape, self.test_cover.shape)
        self.assertEqual(stego.dtype, np.uint8)

        # 2. Stats must reflect single-stego
        self.assertTrue(stats.get('single_stego', False))
        self.assertFalse(stats.get('dual_images', True))
        self.assertGreater(stats['embedded_bpp'], 0.0)

        # 3. Extraction
        extracted_secret, recovered_cover, meta = model.extract(stego, password=self.password)

        # 4. Payload match
        self.assertEqual(extracted_secret, secret, "Extracted payload must match original exactly")

        # 5. Exact cover recovery
        self.assertTrue(np.array_equal(self.test_cover, recovered_cover),
                        "Recovered cover must be bit-exact to original cover")
        max_diff = int(np.max(np.abs(self.test_cover.astype(int) - recovered_cover.astype(int))))
        num_diff = int(np.sum(self.test_cover != recovered_cover))
        self.assertEqual(max_diff, 0, f"Max difference must be 0, got {max_diff}")
        self.assertEqual(num_diff, 0, f"Number of differing pixels must be 0, got {num_diff}")

    def test_02_emd_mod5_all_symbols(self):
        """Verify EMD mod-5 extraction function f(p1, p2) = (p1 + 2*p2) mod 5 across all secret digits."""
        # Test base-5 digit conversions
        sample_bytes = b"Hello EMD!"
        digits = bytes_to_base5_digits(sample_bytes)
        recovered_bytes = base5_digits_to_bytes(digits)
        self.assertEqual(sample_bytes, recovered_bytes)

        # Test all d in {0, 1, 2, 3, 4}
        for d in range(5):
            p1, p2 = 120, 120
            r1, r2 = p1 & 7, p2 & 7
            if d == 0:
                p1_new, p2_new = p1, p2
            elif d == 1:
                p1_new = p1 + 1 if r1 < 7 else p1 - 4
                p2_new = p2
            elif d == 2:
                p1_new = p1
                p2_new = p2 + 1 if r2 < 7 else p2 - 4
            elif d == 3:
                p1_new = p1
                p2_new = p2 - 1 if r2 > 0 else p2 + 4
            else:  # d == 4
                p1_new = p1 - 1 if r1 > 0 else p1 + 4
                p2_new = p2

            # Extraction: (p1_new + 2*p2_new) % 5
            f_orig = (p1 * 1 + p2 * 2) % 5
            f_new = (p1_new * 1 + p2_new * 2) % 5
            extracted_d = (f_new - f_orig) % 5
            self.assertEqual(extracted_d, d, f"EMD mod-5 failed for d={d}")

            # 8-block bitplane immutability check
            self.assertEqual(p1_new & 0xF8, p1 & 0xF8, "Upper bits of p1 must not change")
            self.assertEqual(p2_new & 0xF8, p2 & 0xF8, "Upper bits of p2 must not change")

    def test_03_gamma_fusion_differences(self):
        """Item 4: Verify gamma=0.0 vs 0.5 vs 1.0 produces genuine differences on real images."""
        cnn = load_trained_distortion_cnn()
        d_00 = compute_distortion_maps(self.test_cover, gamma=0.0, model=cnn)
        d_05 = compute_distortion_maps(self.test_cover, gamma=0.5, model=cnn)
        d_10 = compute_distortion_maps(self.test_cover, gamma=1.0, model=cnn)

        diff_0_05 = float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(d_00, d_05)]))
        diff_05_10 = float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(d_05, d_10)]))
        diff_0_10 = float(np.mean([np.mean(np.abs(a - b)) for a, b in zip(d_00, d_10)]))

        print(f"\n[Gamma Fusion Verification]")
        print(f"  Gamma 0.0 vs 0.5 mean abs diff: {diff_0_05:.6f}")
        print(f"  Gamma 0.5 vs 1.0 mean abs diff: {diff_05_10:.6f}")
        print(f"  Gamma 0.0 vs 1.0 mean abs diff: {diff_0_10:.6f}")

        self.assertGreater(diff_0_05, 1e-4, "Gamma 0.0 and 0.5 maps must differ")
        self.assertGreater(diff_05_10, 1e-4, "Gamma 0.5 and 1.0 maps must differ")
        self.assertGreater(diff_0_10, 1e-4, "Gamma 0.0 and 1.0 maps must differ")

    def test_04_benchmark_all_six_models(self):
        """Run all 6 models through BenchmarkRunner to confirm slot 6 is single-stego and achieves 100% recovery."""
        runner = BenchmarkRunner()
        payload = "A" * 128
        df = runner.run_all_models(self.test_cover, payload, password=self.password)

        self.assertEqual(len(df), 6, "Must benchmark all 6 models")

        # Find Proposed Model
        proposed = df[df['Model'] == 'CNN-DA-EMD-OLSB'].iloc[0]
        self.assertEqual(proposed['Carrier_Recovery_Acc_%'], 100.0,
                         "Proposed model must achieve 100.0% carrier recovery")
        self.assertEqual(proposed['Payload_Recovery_Acc_%'], 100.0,
                         "Proposed model must achieve 100.0% payload recovery")
        self.assertEqual(proposed['BER'], 0.0, "Clean extraction BER must be 0.0")
        self.assertGreater(proposed['PSNR_dB'], 38.0, "Proposed model PSNR must exceed 38 dB")

        # Stego output of proposed model must be a single image
        stego_out = proposed['Stego_Output']
        self.assertIsInstance(stego_out, np.ndarray, "Proposed stego output must be single np.ndarray")


if __name__ == '__main__':
    unittest.main()
