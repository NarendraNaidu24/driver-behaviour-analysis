"""
Fits a Mahalanobis OOD detector on the TEMPORAL model's clip-level
features (the LSTM's aggregated output), same principle as
fit_ood_detector.py but operating on 16-frame clip features instead of
single-frame features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import load_taxonomy, build_label_encoders, get_device
from src.datasets.video_clip_dataset import VideoClipDataset, collate_clips
from src.models.temporal_model import TemporalMultiTaskModel
from src.ood_detector import MahalanobisOODDetector

DEVICE = get_device()
TAXONOMY = load_taxonomy()
ENCODERS = build_label_encoders(TAXONOMY)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

import argparse

parser = argparse.ArgumentParser(description="Fit temporal OOD detector")
parser.add_argument("--checkpoint", default=None, help="Path to temporal model checkpoint")
parser.add_argument("--manifest", default="manifests/dmd_video_manifest.csv", help="Path to clip manifest")
parser.add_argument("--out", default=None, help="Output npz path")
args = parser.parse_args()

ckpt_path = args.checkpoint
if ckpt_path is None:
    if os.path.exists("checkpoints/temporal_distraction_dmd_FINAL.pt"):
        ckpt_path = "checkpoints/temporal_distraction_dmd_FINAL.pt"
    else:
        ckpt_path = "checkpoints/temporal_distraction_dmd_full14.pt"

print(f"[ood] loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model = TemporalMultiTaskModel(ckpt["num_classes"], ckpt["domains"], backbone="resnet18")
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

task = "distraction"
dataset_name = ckpt["dataset_name"]
num_classes = ckpt["num_classes"][task]

manifest_path = args.manifest
train_ds = VideoClipDataset([manifest_path], "train", ENCODERS[task], EVAL_TRANSFORM)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, collate_fn=collate_clips)

val_ds = VideoClipDataset([manifest_path], "val", ENCODERS[task], EVAL_TRANSFORM)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_clips)


class ClipFeatureModel:
    def __init__(self, base_model, dataset_name):
        self.base_model = base_model
        self.dataset_name = dataset_name
    def eval(self):
        self.base_model.eval()
    def get_features(self, x, task):
        return self.base_model.get_clip_features(x, task=task, dataset=self.dataset_name)


wrapped = ClipFeatureModel(model, dataset_name)

detector = MahalanobisOODDetector()
detector.fit(wrapped, train_loader, task, DEVICE, num_classes, input_key="clip")
detector.calibrate_threshold(wrapped, val_loader, task, DEVICE, percentile=99, input_key="clip")

out_path = args.out or f"checkpoints/ood_temporal_{task}_{dataset_name}.npz"
detector.save(out_path)
print(f"saved -> {out_path}")