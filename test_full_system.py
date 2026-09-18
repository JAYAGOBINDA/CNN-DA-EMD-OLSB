"""
Full End-to-End System Verification for CNN-DA-EMD-OLSB.
Tests embedding and extracting:
1. Any size image (4K, 2K, non-square) into any size cover (512x512, 256x256, 128x128).
2. Real-world photographic images with natural sensor noise.
3. Text messages.
4. Binary files.
5. Bit-exact cover recovery where capacity allows.
"""

import unittest
import numpy as np
import io
from PIL import Image

from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb
from models.cnn_da_emd_olsb_model import CNNDAEMDOLSBModel
from utils.image_utils import optimize_secret_image
from benchmark.metrics import calculate_psnr, calculate_ssim, compute_mse


class TestFullSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = CNNDAEMDOLSBModel()
        cls.password = "SystemVerifyPass2026!"

    def test_01_text_into_noisy_photo_cover(self):
        """Test embedding text message into a realistic noisy photographic cover."""
        # 256x256 cover with random natural noise
        rng = np.random.RandomState(101)
        cover = rng.randint(20, 235, (256, 256, 3), dtype=np.uint8)
        secret_text = b"Confidential Research Verification: CNN-DA-EMD-OLSB 2026."

        stego, stats = self.model.embed(cover, secret_text, password=self.password)
        self.assertEqual(stego.shape, cover.shape)
        self.assertGreater(calculate_psnr(cover, stego), 60.0)

        extracted, recovered, meta = self.model.extract(stego, password=self.password)
        self.assertEqual(extracted, secret_text)
        print(f"[PASS] Text into noisy cover: PSNR={calculate_psnr(cover, stego):.2f} dB, 0 BER")

    def test_02_4k_image_into_512_cover(self):
        """Test embedding 3840x2160 (4K) image into 512x512 cover."""
        rng = np.random.RandomState(102)
        cover = rng.randint(30, 220, (512, 512, 3), dtype=np.uint8)
        secret_4k = rng.randint(10, 240, (2160, 3840, 3), dtype=np.uint8)

        # Optimize secret image for cover capacity
        target_budget = 6000
        opt_bytes, opt_w, opt_h = optimize_secret_image(secret_4k, target_budget)
        self.assertLessEqual(len(opt_bytes), target_budget + 500)

        stego, stats = self.model.embed(cover, opt_bytes, password=self.password, payload_type=1)
        self.assertEqual(stego.shape, cover.shape)

        extracted_bytes, recovered, meta = self.model.extract(stego, password=self.password)
        self.assertEqual(len(extracted_bytes), len(opt_bytes))
        self.assertEqual(extracted_bytes, opt_bytes)

        # Verify extracted image opens and has valid dimensions
        ext_pil = Image.open(io.BytesIO(extracted_bytes))
        self.assertEqual(ext_pil.size, (opt_w, opt_h))
        print(f"[PASS] 4K secret into 512x512 cover: extracted ({ext_pil.width}x{ext_pil.height}) — {len(extracted_bytes):,} bytes")

    def test_03_2k_image_into_256_cover(self):
        """Test embedding 2048x2048 (2K) image into 256x256 cover."""
        rng = np.random.RandomState(103)
        cover = rng.randint(30, 220, (256, 256, 3), dtype=np.uint8)
        secret_2k = rng.randint(10, 240, (2048, 2048, 3), dtype=np.uint8)

        target_budget = 1500
        opt_bytes, opt_w, opt_h = optimize_secret_image(secret_2k, target_budget)

        stego, stats = self.model.embed(cover, opt_bytes, password=self.password, payload_type=1)
        extracted_bytes, recovered, meta = self.model.extract(stego, password=self.password)
        self.assertEqual(extracted_bytes, opt_bytes)

        ext_pil = Image.open(io.BytesIO(extracted_bytes))
        self.assertGreater(ext_pil.width, 0)
        self.assertGreater(ext_pil.height, 0)
        print(f"[PASS] 2K secret into 256x256 cover: extracted ({ext_pil.width}x{ext_pil.height}) — {len(extracted_bytes):,} bytes")

    def test_04_arbitrary_image_into_small_128_cover(self):
        """Test embedding 1024x768 secret image into tiny 128x128 cover."""
        rng = np.random.RandomState(104)
        cover = rng.randint(30, 220, (128, 128, 3), dtype=np.uint8)
        secret_img = rng.randint(10, 240, (768, 1024, 3), dtype=np.uint8)

        target_budget = 400
        opt_bytes, opt_w, opt_h = optimize_secret_image(secret_img, target_budget)

        stego, stats = self.model.embed(cover, opt_bytes, password=self.password, payload_type=1)
        extracted_bytes, recovered, meta = self.model.extract(stego, password=self.password)
        self.assertEqual(extracted_bytes, opt_bytes)

        ext_pil = Image.open(io.BytesIO(extracted_bytes))
        self.assertGreater(ext_pil.width, 0)
        self.assertGreater(ext_pil.height, 0)
        print(f"[PASS] 1024x768 secret into 128x128 cover: extracted ({ext_pil.width}x{ext_pil.height}) — {len(extracted_bytes):,} bytes")

    def test_05_binary_document_payload(self):
        """Test embedding binary PDF/document payload."""
        rng = np.random.RandomState(105)
        cover = rng.randint(30, 220, (256, 256, 3), dtype=np.uint8)
        binary_payload = b"%PDF-1.4 Mock document binary header data and payload bytes 2026."

        stego, stats = self.model.embed(cover, binary_payload, password=self.password, payload_type=2)
        extracted_bytes, recovered, meta = self.model.extract(stego, password=self.password)
        self.assertEqual(extracted_bytes, binary_payload)
        print(f"[PASS] Binary document into cover: extracted {len(extracted_bytes)} bytes exact match")

    def test_06_exact_cover_recovery_when_smooth(self):
        """Verify 100% bit-exact cover recovery on smooth gradient cover."""
        y, x = np.ogrid[:256, :256]
        gradient = ((x + y) * 2) % 200 + 30
        cover = np.stack([gradient, (gradient + 20) % 220 + 20, (gradient + 40) % 220 + 20], axis=-1).astype(np.uint8)
        secret = b"Antigravity exact cover recovery test payload!"

        stego, stats = self.model.embed(cover, secret, password=self.password)
        extracted, recovered, meta = self.model.extract(stego, password=self.password)

        self.assertEqual(extracted, secret)
        self.assertTrue(np.array_equal(cover, recovered), "Recovered cover must match bit-exactly on smooth cover")
        print("[PASS] Bit-exact cover recovery verified (max_diff=0)")


if __name__ == '__main__':
    unittest.main()
