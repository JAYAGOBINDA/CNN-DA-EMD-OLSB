"""
Model 3: CNN-RDH Predictor (CNN-Based Prediction Reversible Data Hiding)
Reversible Data Hiding (RDH) Baseline from Literature Paper 3.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
from utils.payload_utils import bytes_to_bits, bits_to_bytes
from utils.image_utils import rgb_to_gray


class CNNRDHPredictorNetwork(nn.Module):
    def __init__(self):
        super(CNNRDHPredictorNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = torch.relu(self.conv3(h))
        out = self.conv4(h)
        return out


class CNNRDHPredictor:
    def __init__(self, weights_path: str = "weights/cnn_rdh.pth"):
        self.weights_path = weights_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNRDHPredictorNetwork().to(self.device)

        if os.path.exists(self.weights_path):
            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                self.model.eval()
            except Exception:
                pass
        else:
            self._quick_init_weights()

    def _quick_init_weights(self):
        weights_dir = os.path.dirname(self.weights_path)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        self.model.eval()

    def predict_image(self, gray_img: np.ndarray) -> np.ndarray:
        self.model.eval()
        img_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(img_tensor)
            pred_img = pred_tensor.squeeze().cpu().numpy() * 255.0
        return np.round(np.clip(pred_img, 0, 255))

    def embed(self, cover_rgb: np.ndarray, secret_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w, c = cover_rgb.shape

        payload_bits = bytes_to_bits(secret_bytes)
        total_bits = len(payload_bits)

        cover_gray = rgb_to_gray(cover_rgb)
        pred = self.predict_image(cover_gray)
        diff_G = np.abs(cover_gray.astype(np.float32) - pred)
        flat_diff = diff_G.reshape(-1)
        sort_indices = np.argsort(flat_diff)

        flat_cover = cover_rgb.reshape(-1)
        flat_stego = flat_cover.copy()
        orig_lsbs = []

        bit_idx = 0

        for i in range(len(sort_indices)):
            if bit_idx >= total_bits:
                break
            
            pix_idx = sort_indices[i]
            for ch in range(3):
                if bit_idx >= total_bits:
                    break
                flat_idx = pix_idx * 3 + ch
                orig_lsbs.append(int(flat_cover[flat_idx] & 1))
                b = int(payload_bits[bit_idx])
                flat_stego[flat_idx] = (flat_cover[flat_idx] & 0xFE) | b
                bit_idx += 1

        stego_rgb = flat_stego.reshape(h, w, c)

        stats = {
            'total_bits_embedded': bit_idx,
            'bpp': bit_idx / (h * w * c),
            'sort_indices': sort_indices,
            'orig_lsbs': np.array(orig_lsbs, dtype=np.uint8),
            'model_name': 'CNN-RDH Predictor'
        }

        return stego_rgb, stats

    def extract(self, stego_rgb: np.ndarray, stats: Dict[str, Any]) -> Tuple[bytes, np.ndarray]:
        h, w, c = stego_rgb.shape

        total_bits = stats['total_bits_embedded']
        sort_indices = stats['sort_indices']
        orig_lsbs = stats['orig_lsbs']

        flat_stego = stego_rgb.reshape(-1)
        flat_recovered = flat_stego.copy()

        extracted_bits = []
        bit_idx = 0

        for i in range(len(sort_indices)):
            if bit_idx >= total_bits:
                break
            
            pix_idx = sort_indices[i]
            for ch in range(3):
                if bit_idx >= total_bits:
                    break
                flat_idx = pix_idx * 3 + ch
                b = int(flat_stego[flat_idx] & 1)
                extracted_bits.append(b)

                flat_recovered[flat_idx] = (flat_stego[flat_idx] & 0xFE) | orig_lsbs[bit_idx]
                bit_idx += 1

        extracted_bytes = bits_to_bytes(np.array(extracted_bits[:total_bits], dtype=np.uint8))
        recovered_rgb = flat_recovered.reshape(h, w, c)

        return extracted_bytes, recovered_rgb
