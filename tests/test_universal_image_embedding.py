"""
Test Suite: Universal Image-into-Image Steganography.
Validates embedding arbitrary sized secret images (e.g. 4K, 2K, non-square) into arbitrary sized covers
(512x512, 256x256, 128x128, non-square) with 100% bit-exact cover recovery and secret image extraction.
"""

import unittest
import numpy as np
import io
from PIL import Image

from core.cnn_da_emd_olsb import embed_cnn_da_emd_olsb, extract_cnn_da_emd_olsb
from utils.image_utils import optimize_secret_image
from core.metrics import calculate_psnr, compute_mse


class TestUniversalImageEmbedding(unittest.TestCase):

    def _create_synthetic_image(self, width: int, height: int, pattern: str = "gradient") -> np.ndarray:
        """Generates realistic synthetic RGB image with smooth and textured areas."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if pattern == "gradient":
            x = np.linspace(0, 255, width, dtype=np.uint8)
            y = np.linspace(0, 255, height, dtype=np.uint8)
            xx, yy = np.meshgrid(x, y)
            img[:, :, 0] = xx
            img[:, :, 1] = yy
            img[:, :, 2] = ((xx.astype(int) + yy.astype(int)) // 2).astype(np.uint8)
        else:
            rng = np.random.RandomState(42)
            img = rng.randint(40, 220, (height, width, 3), dtype=np.uint8)
        return img

    def test_4k_secret_into_512_cover(self):
        """Case 1: Extreme 3840x2160 (4K) secret image into 512x512 cover."""
        cover = self._create_synthetic_image(512, 512, "gradient")
        secret_4k = self._create_synthetic_image(3840, 2160, "gradient")

        # Encode 4K to bytes
        buf = io.BytesIO()
        Image.fromarray(secret_4k).save(buf, format="JPEG", quality=85)
        raw_secret_bytes = buf.getvalue()

        # Embed with payload_type=1 (secret image)
        stego, stats = embed_cnn_da_emd_olsb(
            cover_rgb=cover,
            secret_data=raw_secret_bytes,
            password="TestPassword4K!",
            payload_type=1,
            gamma=0.0
        )
        self.assertEqual(stego.shape, cover.shape)

        # Extract
        extracted_bytes, recovered_cover, meta = extract_cnn_da_emd_olsb(
            stego_input=stego,
            password="TestPassword4K!",
            gamma=0.0
        )

        # Bit-exact cover recovery
        self.assertTrue(np.array_equal(cover, recovered_cover), "Cover must be recovered bit-exactly!")
        self.assertEqual(compute_mse(cover, recovered_cover), 0.0)
        self.assertEqual(calculate_psnr(cover, recovered_cover), float('inf'))

        # Secret image successfully decoded
        extracted_img = Image.open(io.BytesIO(extracted_bytes))
        self.assertGreater(extracted_img.width, 0)
        self.assertGreater(extracted_img.height, 0)
        print(f"[PASS] 4K (3840x2160) into 512x512 cover: extracted image {extracted_img.size} ({len(extracted_bytes):,} B)")

    def test_2k_secret_into_256_cover(self):
        """Case 2: 2048x2048 secret image into 256x256 cover."""
        cover = self._create_synthetic_image(256, 256, "gradient")
        secret_2k = self._create_synthetic_image(2048, 2048, "gradient")

        buf = io.BytesIO()
        Image.fromarray(secret_2k).save(buf, format="JPEG", quality=80)
        raw_secret_bytes = buf.getvalue()

        stego, stats = embed_cnn_da_emd_olsb(
            cover_rgb=cover,
            secret_data=raw_secret_bytes,
            password="Secret2K!",
            payload_type=1,
            gamma=0.0
        )

        extracted_bytes, recovered_cover, meta = extract_cnn_da_emd_olsb(
            stego_input=stego,
            password="Secret2K!",
            gamma=0.0
        )

        self.assertTrue(np.array_equal(cover, recovered_cover))
        extracted_img = Image.open(io.BytesIO(extracted_bytes))
        self.assertGreater(extracted_img.width, 0)
        self.assertGreater(extracted_img.height, 0)
        print(f"[PASS] 2048x2048 into 256x256 cover: extracted image {extracted_img.size} ({len(extracted_bytes):,} B)")

    def test_800x600_secret_into_128_cover(self):
        """Case 3: 800x600 secret image into small 128x128 cover."""
        cover = self._create_synthetic_image(128, 128, "gradient")
        secret_img = self._create_synthetic_image(800, 600, "gradient")

        buf = io.BytesIO()
        Image.fromarray(secret_img).save(buf, format="PNG")
        raw_secret_bytes = buf.getvalue()

        stego, stats = embed_cnn_da_emd_olsb(
            cover_rgb=cover,
            secret_data=raw_secret_bytes,
            password="Pass128Cover!",
            payload_type=1,
            gamma=0.0
        )

        extracted_bytes, recovered_cover, meta = extract_cnn_da_emd_olsb(
            stego_input=stego,
            password="Pass128Cover!",
            gamma=0.0
        )

        self.assertTrue(np.array_equal(cover, recovered_cover))
        extracted_img = Image.open(io.BytesIO(extracted_bytes))
        self.assertGreater(extracted_img.width, 0)
        self.assertGreater(extracted_img.height, 0)
        print(f"[PASS] 800x600 into 128x128 cover: extracted image {extracted_img.size} ({len(extracted_bytes):,} B)")


if __name__ == '__main__':
    unittest.main()
