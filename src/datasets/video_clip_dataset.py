"""
Video Clip Dataset
====================
Reads clip manifests (one row per clip, filepath points to a FOLDER of
16 numbered frame images, not a single image) and serves them as
(time, channels, height, width) tensors for the temporal model.
"""

import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class VideoClipDataset(Dataset):
    def __init__(self, manifest_paths, split, label_encoder, transform=None, clip_len=16):
        dfs = [pd.read_csv(p) for p in manifest_paths]
        df = pd.concat(dfs, ignore_index=True)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_encoder = label_encoder
        self.transform = transform
        self.clip_len = clip_len

        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}' across manifests: {manifest_paths}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clip_dir = row["filepath"]

        frames = []
        for i in range(self.clip_len):
            frame_path = os.path.join(clip_dir, f"frame_{i:02d}.jpg")
            image = Image.open(frame_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        clip_tensor = torch.stack(frames)  # (time, channels, height, width)

        task = row["task"]
        label_str = row["label"]
        label_idx = self.label_encoder[label_str]

        return {
            "clip": clip_tensor,
            "task": task,
            "label": label_idx,
            "source_dataset": row["source_dataset"],
        }


def collate_clips(batch):
    clips = torch.stack([b["clip"] for b in batch])  # (batch, time, C, H, W)
    tasks = [b["task"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sources = [b["source_dataset"] for b in batch]
    return {"clip": clips, "task": tasks, "label": labels, "source_dataset": sources}