"""
Temporal Model Training Script
==================================
Trains TemporalMultiTaskModel on video-clip manifests (DMD only, for
now -- State Farm has no video, YawDD's drowsiness clips would need
their own extraction, see notes at bottom).

Given 4GB VRAM, batch_size is intentionally small (4), since each
"sample" is now 16 frames rather than 1 -- effectively a 16x larger
per-sample memory footprint.

Usage:
    python src/train_temporal.py \
        --manifests manifests/dmd_video_manifest.csv \
        --task distraction --epochs 15 --out checkpoints/temporal_distraction_dmd.pt
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score

from src.utils import set_seed, load_taxonomy, build_label_encoders, get_device, ensure_dir
from src.datasets.video_clip_dataset import VideoClipDataset, collate_clips
from src.models.temporal_model import TemporalMultiTaskModel
from src.models.multitask_model import make_domain


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run_epoch(model, loader, optimizer, device, criterion, dataset_name, task, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            clips = batch["clip"].to(device)
            labels = batch["label"].to(device)

            if train:
                optimizer.zero_grad()

            out = model(clips, task=task, dataset=dataset_name)[task]
            loss = criterion(out, labels)

            if train:
                loss.backward()
                optimizer.step()

            preds = out.argmax(dim=1)
            total_loss += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total_n += len(labels)

            if not train:
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if not train else None
    return total_loss / total_n, total_correct / total_n, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--task", default="distraction")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[device] {device}", flush=True)

    taxonomy = load_taxonomy(args.taxonomy)
    label_encoders = build_label_encoders(taxonomy)
    num_classes = {args.task: len(taxonomy["tasks"][args.task]["classes"])}

    train_ds = VideoClipDataset(args.manifests, "train", label_encoders[args.task], get_transforms(True))
    val_ds = VideoClipDataset(args.manifests, "val", label_encoders[args.task], get_transforms(False))
    print(f"[data] {len(train_ds)} train / {len(val_ds)} val clips", flush=True)

    dataset_name = train_ds.df["source_dataset"].iloc[0]
    domains = [make_domain(args.task, dataset_name)]
    print(f"[domains] {domains}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_clips, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_clips, num_workers=args.num_workers)

    model = TemporalMultiTaskModel(num_classes, domains, backbone="resnet18").to(device)
    if args.freeze_backbone:
        model.freeze_early_layers()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    counts = torch.zeros(num_classes[args.task])
    for cls_name, cls_idx in label_encoders[args.task].items():
        counts[cls_idx] = (train_ds.df["label"] == cls_name).sum()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes[args.task] * counts)
    print(f"[class weights] {dict(zip(label_encoders[args.task].keys(), weights.tolist()))}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    best_macro_f1 = 0.0
    ensure_dir(os.path.dirname(args.out) or ".")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, optimizer, device,
                                               criterion, dataset_name, args.task, train=True)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, optimizer, device,
                                                       criterion, dataset_name, args.task, train=False)

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}", flush=True)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "domains": domains,
                "task": args.task,
                "dataset_name": dataset_name,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }, args.out)
            print(f"  -> saved new best checkpoint (val_macro_f1={val_macro_f1:.4f}) to {args.out}", flush=True)

    print(f"[done] best val_macro_f1={best_macro_f1:.4f}, checkpoint at {args.out}", flush=True)


if __name__ == "__main__":
    main()