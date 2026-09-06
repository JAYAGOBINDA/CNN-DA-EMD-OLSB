import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cnn.distortion_cnn import DistortionCNN
from training.distortion_dataset import find_images, DistortionMapDataset


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_paths(paths, val_ratio, seed):
    paths = list(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(round(len(paths) * val_ratio)))
    if len(paths) - n_val < 1:
        raise ValueError("Need at least 2 images so train and validation are non-empty.")
    return paths[n_val:], paths[:n_val]


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        bs = x.size(0)
        total += loss.item() * bs
        count += bs

    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder containing RGB cover images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="models/distortion_cnn.pth")
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paths = find_images(args.data)
    train_paths, val_paths = split_paths(paths, args.val_ratio, args.seed)
    print(f"Images: {len(paths)} | Train: {len(train_paths)} | Validation: {len(val_paths)}")

    train_ds = DistortionMapDataset(train_paths, args.crop_size, train=True)
    val_ds = DistortionMapDataset(val_paths, args.crop_size, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    model = DistortionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = out_path.parent / "distortion_cnn_training_history.csv"

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, criterion, optimizer, device, False)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_mse": train_loss,
            "val_mse": val_loss,
            "learning_rate": lr_now,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train MSE={train_loss:.6f} | "
            f"val MSE={val_loss:.6f} | lr={lr_now:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  Saved best model -> {out_path}")

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\nTraining complete.")
    print(f"Best validation MSE: {best_val:.6f}")
    print(f"Model: {out_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
