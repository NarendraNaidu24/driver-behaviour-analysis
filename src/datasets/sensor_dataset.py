"""
Sensor Dataset
===============
Reads UAH-DriveSet-style manifests (numeric feature columns stored
directly in the CSV, no image file) for the aggression task. This is
a separate, simpler pipeline from unified_dataset.py, since sensor
features don't need image loading/transforms.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class SensorBehaviorDataset(Dataset):
    def __init__(self, manifest_paths, split, task, feature_columns, label_encoder):
        """
        manifest_paths  : list[str] - manifest CSV paths
        split           : "train" | "val" | "test"
        task            : e.g. "aggression" -- filters rows to this task only
        feature_columns : list[str] - which CSV columns are model inputs
        label_encoder   : dict {class_name: index}
        """
        dfs = [pd.read_csv(p) for p in manifest_paths]
        df = pd.concat(dfs, ignore_index=True)
        df = df[(df["split"] == split) & (df["task"] == task)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows for task='{task}', split='{split}' in {manifest_paths}")

        self.df = df
        self.feature_columns = feature_columns
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(
            [row[c] for c in self.feature_columns], dtype=torch.float32
        )
        label = self.label_encoder[row["label"]]
        return {"features": features, "label": label}