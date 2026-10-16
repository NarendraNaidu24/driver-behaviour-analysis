"""
Fits Mahalanobis OOD detectors for distraction and drowsiness, now
domain-aware: for tasks with multiple datasets (distraction has
state_farm + dmd), we fit ONE detector per domain, since each dataset
has its own feature distribution under domain-specific BatchNorm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import load_taxonomy, build_label_encoders, get_device
from src.datasets.unified_dataset import UnifiedDriverBehaviorDataset, collate_fn
from src.models.multitask_model import MultiTaskDriverModel
from src.ood_detector import MahalanobisOODDetector

DEVICE = get_device()
TAXONOMY = load_taxonomy()
ENCODERS = build_label_encoders(TAXONOMY)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

ckpt = torch.load("checkpoints/joint_model_domainbn_unfrozen.pt", map_location=DEVICE)
model = MultiTaskDriverModel(ckpt["num_classes"], ckpt["domains"],
                               backbone=ckpt["backbone"], pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

domain_manifests = {
    "distraction": [
        ("state_farm", "manifests/state_farm_manifest.csv"),
        ("dmd", "manifests/dmd_manifest.csv"),
    ],
    "drowsiness": [
        ("yawdd", "manifests/yawdd_manifest.csv"),
    ],
}

for task, dataset_list in domain_manifests.items():
    num_classes = len(ENCODERS[task])
    for dataset_name, manifest_path in dataset_list:
        print(f"\n=== Fitting OOD detector for domain: {task}::{dataset_name} ===")

        train_ds = UnifiedDriverBehaviorDataset([manifest_path], "train", ENCODERS, EVAL_TRANSFORM)
        train_ds.df = train_ds.df[(train_ds.df["task"] == task) &
                                    (train_ds.df["source_dataset"] == dataset_name)].reset_index(drop=True)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        val_ds = UnifiedDriverBehaviorDataset([manifest_path], "val", ENCODERS, EVAL_TRANSFORM)
        val_ds.df = val_ds.df[(val_ds.df["task"] == task) &
                                (val_ds.df["source_dataset"] == dataset_name)].reset_index(drop=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        class DomainBoundModel:
            def __init__(self, base_model, dataset_name):
                self.base_model = base_model
                self.dataset_name = dataset_name
            def eval(self):
                self.base_model.eval()
            def get_features(self, x, task):
                return self.base_model.get_features(x, task=task, dataset=self.dataset_name)

        bound_model = DomainBoundModel(model, dataset_name)

        detector = MahalanobisOODDetector()
        detector.fit(bound_model, train_loader, task, DEVICE, num_classes)
        detector.calibrate_threshold(bound_model, val_loader, task, DEVICE, percentile=99)

        out_path = f"checkpoints/ood_{task}_{dataset_name}.npz"
        detector.save(out_path)
        print(f"saved -> {out_path}")

print("\ndone")