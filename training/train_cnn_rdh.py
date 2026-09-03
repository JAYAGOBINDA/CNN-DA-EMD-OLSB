"""
Standalone Training Pipeline for Model 3: CNN-RDH Predictor.
Uses PyTorch and Adam optimizer to train CNN pixel predictor weights and saves checkpoint to weights/cnn_rdh.pth.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.cnn_rdh import CNNRDHPredictorNetwork


def train_cnn_rdh(epochs: int = 5, lr: float = 0.001, output_path: str = "weights/cnn_rdh.pth"):
    """
    Trains CNNRDHPredictorNetwork on synthetic/dataset image patches using Adam optimizer.
    """
    print("=" * 60)
    print("🚀 Starting Training Pipeline for Model 3: CNN-RDH Predictor")
    print(f"Optimizer: Adam | Epochs: {epochs} | Output Checkpoint: {output_path}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNRDHPredictorNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Generate synthetic image dataset patches (100 patches of size 64x64)
    np.random.seed(42)
    synthetic_patches = np.random.randint(0, 256, (100, 1, 64, 64), dtype=np.uint8).astype(np.float32) / 255.0
    dataset_tensor = torch.from_numpy(synthetic_patches).to(device)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for i in range(len(dataset_tensor)):
            patch = dataset_tensor[i:i+1]
            
            optimizer.zero_grad()
            pred = model(patch)
            loss = criterion(pred, patch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset_tensor)
        print(f"Epoch [{epoch}/{epochs}] - Loss (MSE): {avg_loss:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print("=" * 60)
    print(f"✅ Training Complete! Weights saved to {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    train_cnn_rdh(epochs=5)
