"""
Cross-Dataset Generalization Evaluation (Domain-Specific BatchNorm version)
==============================================================================
Same purpose as before: evaluate a checkpoint on a manifest, optionally
building the full train-X/test-Y generalization matrix.

Now domain-aware: when evaluating on a manifest, we use that manifest's
own source_dataset as the BatchNorm domain IF the checkpoint was trained
on that exact dataset (a true "in-distribution" domain match). If the
test manifest's dataset was NEVER seen during training (a genuine
cross-dataset generalization test), there is no matching domain -- the
model falls back to its first registered domain for that task, which is
an imperfect but reasonable default for testing zero-shot generalization.
This limitation is itself worth reporting: domain-specific BN improves
performance on datasets seen during training, at some cost to true
zero-shot transfer to entirely new datasets, since there's no
domain-specific normalization available for them.
"""

import argparse
import os
import sys
import glob
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

from src.utils import load_taxonomy, build_label_encoders, get_device
from src.datasets.unified_dataset import UnifiedDriverBehaviorDataset, collate_fn
from src.models.multitask_model import MultiTaskDriverModel, make_domain


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = MultiTaskDriverModel(ckpt["num_classes"], ckpt["domains"],
                                   backbone=ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def evaluate(model, loader, task, dataset, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            out = model(images, task=task, dataset=dataset)[task]
            preds = out.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, macro_f1, len(all_labels)


def run_single_eval(checkpoint_path, test_manifest, task, split, taxonomy_path, batch_size=32):
    device = get_device()
    taxonomy = load_taxonomy(taxonomy_path)
    label_encoders = build_label_encoders(taxonomy)

    model, ckpt = load_model(checkpoint_path, device)

    ds = UnifiedDriverBehaviorDataset([test_manifest], split, label_encoders, get_eval_transform())
    ds.df = ds.df[ds.df["task"] == task].reset_index(drop=True)
    if len(ds) == 0:
        raise ValueError(f"No rows for task='{task}', split='{split}' in {test_manifest}")

    test_source_dataset = ds.df["source_dataset"].iloc[0]
    checkpoint_domains = ckpt.get("domains", [])
    domain_seen_in_training = make_domain(task, test_source_dataset) in checkpoint_domains

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    acc, macro_f1, n = evaluate(model, loader, task, test_source_dataset, device)

    result = {
        "checkpoint": os.path.basename(checkpoint_path),
        "trained_mode": ckpt.get("mode"),
        "trained_task": ckpt.get("task"),
        "test_manifest": os.path.basename(test_manifest),
        "test_source_dataset": test_source_dataset,
        "task": task,
        "split": split,
        "n_samples": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "domain_seen_in_training": domain_seen_in_training,
    }
    print(json.dumps(result, indent=2))
    return result


def build_matrix(checkpoints_dir, manifests_dir, taxonomy_path, out_path, split="test"):
    checkpoint_paths = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
    manifest_paths = glob.glob(os.path.join(manifests_dir, "*.csv"))

    rows = []
    for ckpt_path in checkpoint_paths:
        device = get_device()
        _, ckpt = load_model(ckpt_path, device)
        tasks_to_eval = (
            [ckpt["task"]] if ckpt.get("mode") == "single" and ckpt.get("task")
            else list(ckpt["num_classes"].keys())
        )
        for manifest_path in manifest_paths:
            for task in tasks_to_eval:
                try:
                    result = run_single_eval(ckpt_path, manifest_path, task, split, taxonomy_path)
                    rows.append(result)
                except ValueError as e:
                    print(f"  [skip] {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[done] wrote {len(df)} rows -> {out_path}")

    if len(df) > 0:
        pivot = df.pivot_table(index=["checkpoint", "task"], columns="test_source_dataset",
                                values="accuracy")
        pivot_path = out_path.replace(".csv", "_pivot.csv")
        pivot.to_csv(pivot_path)
        print(f"[done] wrote pivoted generalization matrix -> {pivot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_matrix", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--test_manifest")
    parser.add_argument("--task", choices=["distraction", "drowsiness", "aggression"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--manifests_dir", default="manifests")
    parser.add_argument("--out", default="results/generalization_matrix.csv")
    parser.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    args = parser.parse_args()

    if args.build_matrix:
        build_matrix(args.checkpoints_dir, args.manifests_dir, args.taxonomy, args.out, args.split)
    else:
        if not (args.checkpoint and args.test_manifest and args.task):
            parser.error("--checkpoint, --test_manifest, and --task are required "
                          "unless using --build_matrix")
        run_single_eval(args.checkpoint, args.test_manifest, args.task, args.split, args.taxonomy)


if __name__ == "__main__":
    main()