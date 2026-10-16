"""
audit_persistence_test.py
=========================
Evaluates the temporal distraction model on held-out test sessions
BOTH with and without the Euro NCAP 3-of-5 persistence filter.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import deque, Counter
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_taxonomy, build_label_encoders, get_device
from src.datasets.video_clip_dataset import VideoClipDataset, collate_clips
from src.models.temporal_model import TemporalMultiTaskModel

DEVICE = get_device()
TAXONOMY = load_taxonomy()
ENCODERS = build_label_encoders(TAXONOMY)
TASK = "distraction"
CLASS_NAMES = TAXONOMY["tasks"][TASK]["classes"]
NORM_IDX = CLASS_NAMES.index("normal_driving")

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def run_persistence_audit():
    ckpt_path = "checkpoints/temporal_distraction_dmd_FINAL.pt"
    manifest_path = "manifests/dmd_video_manifest.csv"

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TemporalMultiTaskModel(ckpt["num_classes"], ckpt["domains"], backbone="resnet18")
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    dataset_name = ckpt["dataset_name"]

    df = pd.read_csv(manifest_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    # Extract session prefix to group clips chronologically
    test_df["session"] = test_df["filepath"].apply(lambda p: "_".join(os.path.basename(p).split("_")[:7]))
    
    # We evaluate sequentially session by session
    all_raw_preds = []
    all_filtered_preds = []
    all_ground_truths = []

    print(f"Loaded test split with {len(test_df)} clips across {test_df['session'].nunique()} continuous sessions.")

    # To be fast, let's load with batch_size=32, compute logits for all test clips, then apply filter per session
    test_ds = VideoClipDataset([manifest_path], "test", ENCODERS[TASK], EVAL_TRANSFORM)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_clips, num_workers=2)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            clips = batch["clip"].to(DEVICE)
            out = model(clips, task=TASK, dataset=dataset_name)[TASK]
            all_logits.append(out.cpu())
            all_labels.extend(batch["label"].tolist())

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.softmax(all_logits, dim=1).numpy()
    test_df["probs"] = list(all_probs)
    test_df["label_idx"] = all_labels

    for session_id, group in test_df.groupby("session", sort=False):
        history = deque(maxlen=5)
        for _, row in group.iterrows():
            probs = row["probs"]
            gt = row["label_idx"]
            raw_idx = probs.argmax()
            conf = probs[raw_idx]
            norm_conf = probs[NORM_IDX]

            # Raw prediction
            all_raw_preds.append(raw_idx)
            all_ground_truths.append(gt)

            # Euro NCAP persistence filter (3-of-5 window requirement)
            if raw_idx != NORM_IDX:
                if conf < 0.65 or (conf - norm_conf) < 0.15:
                    cand = NORM_IDX
                else:
                    cand = raw_idx
            else:
                cand = NORM_IDX

            history.append(cand)
            counts = Counter(history)
            top_cand, top_cnt = counts.most_common(1)[0]
            if top_cand != NORM_IDX and top_cnt < 3:
                filtered_idx = NORM_IDX
            else:
                filtered_idx = top_cand

            all_filtered_preds.append(filtered_idx)

    raw_acc = accuracy_score(all_ground_truths, all_raw_preds)
    filt_acc = accuracy_score(all_ground_truths, all_filtered_preds)

    safe_mask = [gt == NORM_IDX for gt in all_ground_truths]
    total_safe = sum(safe_mask)
    raw_fps = sum(1 for p, s in zip(all_raw_preds, safe_mask) if s and p != NORM_IDX)
    filt_fps = sum(1 for p, s in zip(all_filtered_preds, safe_mask) if s and p != NORM_IDX)

    print("\n" + "=" * 75)
    print(f"PERSISTENCE FILTER EVALUATION ON TEST SET ({len(all_ground_truths)} clips)")
    print("=" * 75)
    print(f"\n1. WITHOUT Euro NCAP Filter (Raw per-clip argmax):")
    print(f"   Overall Accuracy:            {raw_acc:.4f} ({raw_acc*100:.2f}%)")
    print(f"   False Alarms on Safe Driving: {raw_fps}/{total_safe} ({raw_fps/total_safe*100:.2f}%)")
    print("\nClassification Report (WITHOUT Filter):")
    print(classification_report(all_ground_truths, all_raw_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\n" + "-" * 75)
    print(f"\n2. WITH Euro NCAP Filter (3-of-5 window persistence):")
    print(f"   Overall Accuracy:            {filt_acc:.4f} ({filt_acc*100:.2f}%)")
    print(f"   False Alarms on Safe Driving: {filt_fps}/{total_safe} ({filt_fps/total_safe*100:.2f}%)")
    print("\nClassification Report (WITH Filter):")
    print(classification_report(all_ground_truths, all_filtered_preds, target_names=CLASS_NAMES, digits=4, zero_division=0))

if __name__ == "__main__":
    run_persistence_audit()
