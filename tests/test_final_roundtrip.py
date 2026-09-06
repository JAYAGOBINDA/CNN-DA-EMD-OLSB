"""
Dedicated Final Roundtrip Test for CNN-DA-EMD-OLSB Single-Stego RDH.

Executes Model 6 end-to-end on a 512x512 synthetic RGB image and verifies:
1. Single-stego property (single HxWx3 uint8 array, no dual stego).
2. Exact cover recovery (max_diff == 0, num_diff == 0, np.array_equal == True).
3. Zero BER (100% payload recovery).
4. Deterministic gamma recovery (gamma = 0.61 embedded, recovered without brute-force).
5. CNN inference execution truthfully reported.
6. Accurate BPP and payload accounting (raw_secret_bits vs actual_embedded_bits).
7. Complete quality metrics (PSNR, SSIM, wPSNR, MSE).
"""

import unittest
import numpy as np

from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from core.metrics import compute_ber, evaluate_quality, compute_bpp


class TestFinalRoundtrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.password = "FinalRoundtrip2026!"
        cls.gamma_test = 0.61

        # Create a realistic 512x512 synthetic RGB image with smooth gradient features
        y, x = np.ogrid[:512, :512]
        r_chan = ((x + y) * 2) % 200 + 30
        g_chan = (r_chan + 20) % 220 + 20
        b_chan = (r_chan + 40) % 220 + 20
        cls.cover_512 = np.stack([r_chan, g_chan, b_chan], axis=-1).astype(np.uint8)

        # Secret payload
        cls.secret_data = (
            b"Antigravity CNN-DA-EMD-OLSB Final Validation Payload: "
            b"Deterministic gamma=0.61 bootstrap, reserved bootstrap region, "
            b"authenticated recovery side information, exact cover recovery!"
        )

    def test_end_to_end_model6_gamma_061(self):
        """Execute Model 6 with gamma=0.61 on 512x512 and verify all research requirements."""
        model = CNNDAEMDOLSBModel(gamma=self.gamma_test, use_cnn=True)

        # ── 1. Embed ──────────────────────────────────────────────────────────
        stego, stats = model.embed(self.cover_512, self.secret_data, password=self.password)

        # Single-stego assertions
        self.assertIsInstance(stego, np.ndarray, "Stego output must be a single np.ndarray")
        self.assertEqual(stego.ndim, 3, "Stego must be 3-dimensional (H, W, C)")
        self.assertEqual(stego.shape, (512, 512, 3), "Stego shape must match 512x512x3")
        self.assertEqual(stego.dtype, np.uint8, "Stego dtype must be uint8")
        self.assertTrue(stats.get('single_stego', False), "Stats must report single_stego=True")
        self.assertFalse(stats.get('dual_images', True), "Stats must report dual_images=False")

        # CNN inference tracking
        self.assertTrue(stats.get('cnn_inference_executed', False),
                        "CNN inference must be genuinely executed and truthfully reported")

        # Payload & BPP accounting
        raw_bits = len(self.secret_data) * 8
        self.assertEqual(stats['raw_secret_bits'], raw_bits)
        actual_bits = stats['actual_embedded_bits']
        self.assertGreaterEqual(actual_bits, raw_bits,
                                "Actual embedded bits must include bootstrap header, side info, and ciphertext")
        self.assertIn('recovery_side_info_bytes', stats)
        self.assertGreater(stats['embedded_bpp'], 0.0)

        # ── 2. Extract ────────────────────────────────────────────────────────
        # Extract without passing gamma=0.61 — extractor must deterministically recover gamma
        extracted_secret, recovered_cover, meta = model.extract(stego, password=self.password)

        # Deterministic gamma recovery check
        recovered_gamma = meta.get('gamma')
        self.assertIsNotNone(recovered_gamma, "Extractor must return recovered gamma in metadata")
        self.assertAlmostEqual(recovered_gamma, self.gamma_test, places=2,
                              msg=f"Expected gamma {self.gamma_test}, got {recovered_gamma}")

        # Payload recovery & zero BER
        self.assertEqual(extracted_secret, self.secret_data, "Extracted payload must match original exactly")
        orig_bits = np.unpackbits(np.frombuffer(self.secret_data, dtype=np.uint8))
        extr_bits = np.unpackbits(np.frombuffer(extracted_secret, dtype=np.uint8))
        ber = compute_ber(orig_bits, extr_bits)
        self.assertEqual(ber, 0.0, f"Bit Error Rate must be 0.0, got {ber}")

        # Exact cover recovery
        self.assertTrue(np.array_equal(self.cover_512, recovered_cover),
                        "Recovered cover must be bit-exact to original cover")
        max_diff = int(np.max(np.abs(self.cover_512.astype(int) - recovered_cover.astype(int))))
        num_diff = int(np.sum(self.cover_512 != recovered_cover))
        self.assertEqual(max_diff, 0, f"Max difference must be 0, got {max_diff}")
        self.assertEqual(num_diff, 0, f"Differing pixel count must be 0, got {num_diff}")

        # Stego quality evaluation
        quality = evaluate_quality(self.cover_512, stego, total_embedded_bits=actual_bits)
        print("\n" + "=" * 60)
        print("  CNN-DA-EMD-OLSB 512x512 FINAL ROUNDTRIP TEST RESULTS")
        print("=" * 60)
        print(f"  Cover Image Shape:       {self.cover_512.shape}")
        print(f"  Single-Stego Output:     {stego.shape} (ndim={stego.ndim})")
        print(f"  Embedded Gamma:          {self.gamma_test}")
        print(f"  Recovered Gamma:         {recovered_gamma}")
        print(f"  CNN Inference Executed:  {stats['cnn_inference_executed']}")
        print(f"  Raw Secret Bits:         {stats['raw_secret_bits']} ({stats['raw_secret_bytes']} bytes)")
        print(f"  Side Info Bytes:         {stats['recovery_side_info_bytes']}")
        print(f"  Actual Embedded Bits:    {stats['actual_embedded_bits']}")
        print(f"  Raw BPP:                 {stats['raw_bpp']:.4f}")
        print(f"  Embedded BPP:            {stats['embedded_bpp']:.4f}")
        print(f"  Stego PSNR (dB):         {quality['PSNR_dB']:.2f}")
        print(f"  Stego SSIM:              {quality['SSIM']:.4f}")
        print(f"  Stego wPSNR (dB):        {quality['wPSNR_dB']:.2f}")
        print(f"  Stego MSE:               {quality['MSE']:.4f}")
        print(f"  BER:                     {ber:.6f}")
        print(f"  Cover Max Diff:          {max_diff}")
        print(f"  Cover Differing Pixels:  {num_diff} / {512*512*3}")
        print(f"  Exact Cover Reversible:  {np.array_equal(self.cover_512, recovered_cover)}")
        print("=" * 60)

        self.assertGreater(quality['PSNR_dB'], 38.0, "PSNR should exceed 38 dB")
        self.assertGreater(quality['SSIM'], 0.95, "SSIM should exceed 0.95")


if __name__ == '__main__':
    unittest.main()
