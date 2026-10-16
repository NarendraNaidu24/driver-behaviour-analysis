"""
Unified Dataset
================
Reads one or more manifest CSVs and serves them through a single,
consistent PyTorch Dataset interface.

Each sample carries a `task` field (distraction/drowsiness/aggression)
since different datasets cover different tasks. The training loop uses
this to route each sample to the correct model head.
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class UnifiedDriverBehaviorDataset(Dataset):
    def __init__(self, manifest_paths, split, label_encoders, transform=None):
        """
        manifest_paths : list[str] - one or more manifest CSV paths to combine
        split          : "train" | "val" | "test"
        label_encoders : dict from utils.build_label_encoders()
        transform      : torchvision transform applied to the image
        """
        dfs = [pd.read_csv(p) for p in manifest_paths]
        df = pd.concat(dfs, ignore_index=True)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_encoders = label_encoders
        self.transform = transform

        if len(self.df) == 0:
            raise ValueError(
                f"No rows found for split='{split}' across manifests: {manifest_paths}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        task = row["task"]
        label_str = row["label"]
        label_idx = self.label_encoders[task][label_str]

        return {
            "image": image,
            "task": task,
            "label": label_idx,
            "source_dataset": row["source_dataset"],
        }


def collate_fn(batch):
    """
    Custom collate needed because samples in a batch can belong to
    different tasks when training jointly on combined manifests.
    """
    import torch
    images = torch.stack([b["image"] for b in batch])
    tasks = [b["task"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sources = [b["source_dataset"] for b in batch]
    return {"image": images, "task": tasks, "label": labels, "source_dataset": sources}