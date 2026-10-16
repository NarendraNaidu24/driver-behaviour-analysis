"""
Training Script (Domain-Specific BatchNorm version)
=======================================================
Groups each batch by (task, source_dataset) domain, since the model
now needs both to select the right BatchNorm statistics.

Usage examples:

  # Joint model across multiple datasets/tasks
  python src/train.py --mode joint \
      --manifests manifests/state_farm_manifest.csv manifests/dmd_manifest.csv manifests/yawdd_manifest.csv \
      --backbone resnet18 --epochs 15 --out checkpoints/joint_model_domainbn.pt

  # Single-task baseline (still domain-aware internally, just one task active)
  python src/train.py --mode single --task distraction \
      --manifests manifests/state_farm_manifest.csv manifests/dmd_manifest.csv \
      --backbone resnet18 --epochs 15 --out checkpoints/distraction_multi.pt
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

from src.utils import set_seed, load_taxonomy, build_label_encoders, get_device, ensure_dir
from src.datasets.unified_dataset import UnifiedDriverBehaviorDataset, collate_fn
from src.models.multitask_model import MultiTaskDriverModel, make_domain
from sklearn.metrics import f1_score


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run_epoch(model, loader, optimizer, device, criterion_by_task: dict,
              task_loss_weights: dict, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds_by_task = {}
    all_labels_by_task = {}

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            tasks = batch["task"]
            sources = batch["source_dataset"]

            if train:
                optimizer.zero_grad()

            batch_loss = 0.0
            preds = torch.zeros_like(labels)
            unique_domains = set(zip(tasks, sources))
            for t, ds in unique_domains:
                idx = [i for i in range(len(tasks)) if tasks[i] == t and sources[i] == ds]
                idx_t = torch.tensor(idx, device=device)
                out = model(images[idx_t], task=t, dataset=ds)[t]
                loss_t = criterion_by_task[t](out, labels[idx_t])
                task_weight = task_loss_weights.get(t, 1.0)
                batch_loss = batch_loss + loss_t * task_weight * len(idx)
                preds[idx_t] = out.argmax(dim=1)

                if not train:
                    all_preds_by_task.setdefault(t, []).extend(out.argmax(dim=1).cpu().tolist())
                    all_labels_by_task.setdefault(t, []).extend(labels[idx_t].cpu().tolist())

            batch_loss = batch_loss / len(tasks)

            if train:
                batch_loss.backward()
                optimizer.step()

            total_loss += batch_loss.item() * len(tasks)
            total_correct += (preds == labels).sum().item()
            total_n += len(tasks)

    macro_f1 = None
    if not train and all_preds_by_task:
        f1_per_task = []
        for t in all_preds_by_task:
            f1 = f1_score(all_labels_by_task[t], all_preds_by_task[t], average="macro", zero_division=0)
            f1_per_task.append(f1)
        macro_f1 = sum(f1_per_task) / len(f1_per_task)

    return total_loss / total_n, total_correct / total_n, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["joint", "single"], required=True)
    parser.add_argument("--task", choices=["distraction", "drowsiness", "aggression"],
                         help="Required when --mode single")
    parser.add_argument("--manifests", nargs="+", required=True,
                         help="One or more manifest CSV paths")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", required=True, help="Path to save the trained checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="Freeze early backbone layers to reduce overfitting on small/noisy data")
    parser.add_argument("--task_weights", type=str, default="",
                         help="Comma-separated per-task loss weights for joint mode, "
                              "e.g. 'distraction=1.5,drowsiness=1.0'. Tasks not listed default to 1.0.")
    args = parser.parse_args()

    if args.mode == "single" and args.task is None:
        parser.error("--task is required when --mode single")

    set_seed(args.seed)
    device = get_device()
    print(f"[device] {device}")

    taxonomy = load_taxonomy(args.taxonomy)
    label_encoders = build_label_encoders(taxonomy)
    num_classes = {t: len(spec["classes"]) for t, spec in taxonomy["tasks"].items()}

    train_ds = UnifiedDriverBehaviorDataset(args.manifests, "train", label_encoders, get_transforms(True))
    val_ds = UnifiedDriverBehaviorDataset(args.manifests, "val", label_encoders, get_transforms(False))

    if args.mode == "single":
        train_ds.df = train_ds.df[train_ds.df["task"] == args.task].reset_index(drop=True)
        val_ds.df = val_ds.df[val_ds.df["task"] == args.task].reset_index(drop=True)
        print(f"[single-task] restricted to task='{args.task}': "
              f"{len(train_ds)} train / {len(val_ds)} val samples")
    else:
        print(f"[joint] {len(train_ds)} train / {len(val_ds)} val samples "
              f"across tasks: {sorted(train_ds.df['task'].unique())}")

    domain_pairs = sorted(set(zip(train_ds.df["task"], train_ds.df["source_dataset"])))
    domains = [make_domain(t, ds) for t, ds in domain_pairs]
    print(f"[domains] {domains}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)

    model = MultiTaskDriverModel(num_classes, domains, backbone=args.backbone).to(device)
    if args.freeze_backbone:
        model.freeze_early_layers()

    task_loss_weights = {}
    if args.task_weights:
        for pair in args.task_weights.split(","):
            k, v = pair.split("=")
            task_loss_weights[k.strip()] = float(v)
    print(f"[task loss weights] {task_loss_weights if task_loss_weights else '(all default 1.0)'}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    WEIGHT_CAPS = {"other_distraction": 1.0}

    criterion_by_task = {}
    for task in num_classes:
        task_train_df = train_ds.df[train_ds.df["task"] == task]
        n_classes = num_classes[task]
        counts = torch.zeros(n_classes)
        for cls_name, cls_idx in label_encoders[task].items():
            counts[cls_idx] = (task_train_df["label"] == cls_name).sum()
        counts = counts.clamp(min=1)
        weights = counts.sum() / (n_classes * counts)
        for cls_name, cap in WEIGHT_CAPS.items():
            if cls_name in label_encoders[task]:
                idx = label_encoders[task][cls_name]
                weights[idx] = min(weights[idx].item(), cap)
        print(f"[class weights] task={task}: "
              f"{dict(zip(label_encoders[task].keys(), weights.tolist()))}")
        criterion_by_task[task] = nn.CrossEntropyLoss(weight=weights.to(device))

    best_val_acc = 0.0
    ensure_dir(os.path.dirname(args.out) or ".")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, optimizer, device,
                                               criterion_by_task, task_loss_weights, train=True)
        val_loss, val_acc, val_macro_f1 = run_epoch(model, val_loader, optimizer, device,
                                                       criterion_by_task, task_loss_weights, train=False)

        print(f"[epoch {epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val_acc:
            best_val_acc = val_macro_f1
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "domains": domains,
                "backbone": args.backbone,
                "mode": args.mode,
                "task": args.task if args.mode == "single" else None,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
            }, args.out)
            print(f"  -> saved new best checkpoint (val_macro_f1={val_macro_f1:.4f}, val_acc={val_acc:.4f}) to {args.out}")

    print(f"[done] best val_macro_f1={best_val_acc:.4f}, checkpoint at {args.out}")

if __name__ == "__main__":
    main()