"""
Sensor Model Training Script
==============================
Trains the small feedforward SensorBehaviorModel on UAH-DriveSet-style
manifests (numeric features, no images). Separate from train.py since
the data pipeline and model are fundamentally different (sensor
features vs. images).

Usage:
    python src/train_sensor.py \
        --manifests manifests/uah_driveset_manifest.csv \
        --task aggression \
        --epochs 30 --out checkpoints/aggression_uah.pt
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import set_seed, load_taxonomy, get_device, ensure_dir
from src.datasets.sensor_dataset import SensorBehaviorDataset
from src.models.sensor_model import SensorBehaviorModel


def run_epoch(model, loader, optimizer, device, criterion, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            out = model(features)
            loss = criterion(out, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = out.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_n += len(labels)

    return total_loss / total_n, total_correct / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--task", default="aggression")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[device] {device}")

    taxonomy = load_taxonomy(args.taxonomy)
    task_spec = taxonomy["sensor_tasks"][args.task]
    classes = task_spec["classes"]
    feature_columns = task_spec["feature_columns"]
    label_encoder = {c: i for i, c in enumerate(classes)}
    print(f"[task] {args.task}: classes={classes}, features={feature_columns}")

    train_ds = SensorBehaviorDataset(args.manifests, "train", args.task, feature_columns, label_encoder)
    val_ds = SensorBehaviorDataset(args.manifests, "val", args.task, feature_columns, label_encoder)
    print(f"[data] {len(train_ds)} train / {len(val_ds)} val samples")

    # class weighting, same principle as the image pipeline
    counts = torch.zeros(len(classes))
    for cls_name, cls_idx in label_encoder.items():
        counts[cls_idx] = (train_ds.df["label"] == cls_name).sum()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (len(classes) * counts)
    print(f"[class weights] {dict(zip(classes, weights.tolist()))}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SensorBehaviorModel(num_features=len(feature_columns), num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_val_acc = 0.0
    ensure_dir(os.path.dirname(args.out) or ".")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, criterion, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, criterion, train=False)

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_features": len(feature_columns),
                "num_classes": len(classes),
                "classes": classes,
                "feature_columns": feature_columns,
                "task": args.task,
                "val_acc": val_acc,
            }, args.out)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f}) to {args.out}")

    print(f"[done] best val_acc={best_val_acc:.4f}, checkpoint at {args.out}")


if __name__ == "__main__":
    main()