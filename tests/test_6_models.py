"""
Automated Test Suite for the 6-Model Steganography Research Framework.
Tests embedding, extraction, carrier photo recovery, and metrics for all 6 models.
"""

import unittest
import numpy as np

from benchmark.runner import BenchmarkRunner


class Test6ModelFramework(unittest.TestCase):

    def setUp(self):
        self.runner = BenchmarkRunner()
        self.password = "Pass123!"
        self.secret_text = "Research steganography payload verification test 2026"
        
        # Create realistic test image with smooth and textured natural photo patterns
        y, x = np.ogrid[:128, :128]
        gradient = ((x + y) * 2) % 200 + 30
        cover = np.zeros((128, 128, 3), dtype=np.uint8)
        cover[:, :, 0] = gradient
        cover[:, :, 1] = (gradient + 20) % 220 + 20
        cover[:, :, 2] = (gradient + 40) % 220 + 20
        self.cover_rgb = cover

    def test_01_mpeh_rdh(self):
        res = self.runner.run_single_model("MPEH-RDH", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)
        self.assertEqual(res['Carrier_Recovery_Acc_%'], 100.0)

    def test_02_mcsh_rdh(self):
        res = self.runner.run_single_model("MCSH-RDH", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)
        self.assertEqual(res['Carrier_Recovery_Acc_%'], 100.0)

    def test_03_cnn_rdh(self):
        res = self.runner.run_single_model("CNN-RDH Predictor", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)
        self.assertEqual(res['Carrier_Recovery_Acc_%'], 100.0)

    def test_04_srdnn_stego(self):
        res = self.runner.run_single_model("SRDNN-Stego", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)

    def test_05_emd_olsb(self):
        res = self.runner.run_single_model("EMD-OLSB RDH", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)
        self.assertEqual(res['Carrier_Recovery_Acc_%'], 100.0)

    def test_06_cnn_da_emd_olsb(self):
        """Test CNN-DA-EMD-OLSB (Proposed): must achieve 100% payload AND carrier recovery."""
        res = self.runner.run_single_model("CNN-DA-EMD-OLSB", self.cover_rgb, self.secret_text)
        self.assertGreater(res['PSNR_dB'], 35.0)
        self.assertEqual(res['Payload_Recovery_Acc_%'], 100.0)
        self.assertEqual(res['Carrier_Recovery_Acc_%'], 100.0,
                         "CNN-DA-EMD-OLSB must achieve exact 100.0% carrier recovery "
                         "via dual-image averaging, not 99.99%.")

    def test_07_run_all_models(self):
        df = self.runner.run_all_models(self.cover_rgb, self.secret_text)
        self.assertEqual(len(df), 6)


if __name__ == '__main__':
    unittest.main()
