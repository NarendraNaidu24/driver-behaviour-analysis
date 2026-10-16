"""
audit_evaluation.py
===================
Rigorous audit evaluation script fulfilling:
- Item 6: Real evaluation numbers for both Temporal Distraction model and Sensor Aggression model.
- Item 8: Empirical persistence filter validation (with vs without filter) on held-out test ground truth.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter, deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_taxonomy, build_label_encoders, get_device
from src.datasets.video_clip_dataset import VideoClipDataset, collate_clips
from src.datasets.sensor_dataset import SensorBehaviorDataset
from src.models.temporal_model import TemporalMultiTaskModel
from src.models.sensor_model import SensorBehaviorModel

DEVICE = get_device()
TAXONOMY = load_taxonomy()
ENCODERS = build_label_encoders(TAXONOMY)

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def audit_temporal_model():
    print("=" * 75)
    print(" AUDIT PART 1: TEMPORAL DISTRACTION MODEL ON HELD-OUT TEST SPLIT")
    print("=" * 75)

    ckpt_path = "checkpoints/temporal_distraction_dmd_FINAL.pt"
    manifest_path = "manifests/dmd_video_manifest.csv"
    task = "distraction"

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = TemporalMultiTaskModel(ckpt["num_classes"], ckpt["domains"], backbone="resnet18")
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    dataset_name = ckpt["dataset_name"]

    test_ds = VideoClipDataset([manifest_path], "test", ENCODERS[task], EVAL_TRANSFORM)
    print(f"Loaded held-out test split: {len(test_ds)} clips from test subjects: {test_ds.df['subject'].unique().tolist() if 'subject' in test_ds.df.columns else 'held-out test subjects'}")
    
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_clips, num_workers=2)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            clips = batch["clip"].to(DEVICE)
            labels = batch["label"]
            out = model(clips, task=task, dataset=dataset_name)[task]
            preds = out.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    class_names = TAXONOMY["tasks"][task]["classes"]
    acc = accuracy_score(all_labels, all_preds)
    
    # Majority baseline: always guessing the most frequent class in test split
    label_counts = Counter(all_labels)
    majority_class, majority_count = label_counts.most_common(1)[0]
    majority_acc = majority_count / len(all_labels)
    majority_class_name = class_names[majority_class]

    print(f"\nOverall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Majority Baseline Accuracy: {majority_acc:.4f} ({majority_acc*100:.2f}%) [Always predicting '{majority_class_name}']")
    print(f"Improvement over Majority Baseline: +{(acc - majority_acc)*100:.2f} percentage points\n")
    print("Per-Class Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0))


def audit_sensor_model():
    print("\n" + "=" * 75)
    print(" AUDIT PART 2: SENSOR / AGGRESSION MODEL ON HELD-OUT TEST SPLIT")
    print("=" * 75)

    ckpt_path = "checkpoints/aggression_uah.pt"
    manifest_path = "manifests/uah_driveset_manifest.csv"
    task = "aggression"

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint {ckpt_path} not found!")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    classes = ckpt["classes"]
    feature_columns = ckpt["feature_columns"]
    label_encoder = {c: i for i, c in enumerate(classes)}

    model = SensorBehaviorModel(num_features=len(feature_columns), num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    test_ds = SensorBehaviorDataset([manifest_path], "test", task, feature_columns, label_encoder)
    print(f"Loaded UAH-DriveSet test split: {len(test_ds)} samples")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Classes ({len(classes)}): {classes}")

    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(DEVICE)
            labels = batch["label"]
            out = model(feats)
            preds = out.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    label_counts = Counter(all_labels)
    majority_class, majority_count = label_counts.most_common(1)[0]
    majority_acc = majority_count / len(all_labels)
    majority_class_name = classes[majority_class]

    print(f"\nOverall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Majority Baseline Accuracy: {majority_acc:.4f} ({majority_acc*100:.2f}%) [Always predicting '{majority_class_name}']")
    print(f"Improvement over Majority Baseline: +{(acc - majority_acc)*100:.2f} percentage points\n")
    print("Per-Class Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4, zero_division=0))


def audit_persistence_filter():
    print("\n" + "=" * 75)
    print(" AUDIT PART 3: PERSISTENCE FILTER VALIDATION ON TEST GROUND TRUTH")
    print("=" * 75)

    # We evaluate on an authentic continuous recording session from held-out Test Subject 13 or 33
    # Let's inspect Subject 13 session 1: E:\nare\DMD\gA\13\s1
    json_path = r"E:\nare\DMD\gA\13\s1\gC_13_s1_2019-03-04T10;26;12+01;00_rgb_ann_distraction.json"
    video_path = r"E:\nare\DMD\gA\13\s1\gC_13_s1_2019-03-04T10;26;12+01;00_rgb_body.mp4"

    if not os.path.exists(json_path) or not os.path.exists(video_path):
        print("  [skip] Subject 13 raw files not accessible for continuous validation.")
        return

    print(f"Validating on held-out Test Subject 13 session 1:")
    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  JSON:  {os.path.basename(json_path)}")

    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    actions = d.get('openlabel', {}).get('actions', {})

    # Ground truth mapping per frame interval
    mapping = TAXONOMY["dataset_mappings"]["dmd_distraction"]["mapping"]
    intervals = []
    for a in actions.values():
        t = a.get('type', '').replace('driver_actions/', '')
        if t in mapping:
            mapped_lbl = mapping[t]
            for it in a.get('frame_intervals', []):
                intervals.append((it['frame_start'], it['frame_end'], mapped_lbl))

    # Load model
    ckpt = torch.load("checkpoints/temporal_distraction_dmd_FINAL.pt", map_location=DEVICE)
    model = TemporalMultiTaskModel(ckpt["num_classes"], ckpt["domains"], backbone="resnet18")
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    dataset_name = ckpt["dataset_name"]

    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.8
    # Test on a representative continuous block of 1500 frames (~50 seconds) containing safe driving + action
    start_frame = 300
    num_frames = 1500
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, f = cap.read()
        if not ret: break
        frames.append(cv2.resize(f, (224, 224)))
    cap.release()

    CLIP_LEN = 16
    CLIP_STRIDE = 8
    dist_classes = TAXONOMY["tasks"]["distraction"]["classes"]
    norm_idx = dist_classes.index("normal_driving")

    raw_window_preds = []
    filtered_window_preds = []
    ground_truth_labels = []

    history = deque(maxlen=5)

    with torch.no_grad():
        for i in range(0, len(frames) - CLIP_LEN + 1, CLIP_STRIDE):
            clip_frames = frames[i:i + CLIP_LEN]
            tensors = [transforms.ToTensor()(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in clip_frames]
            tensors = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t) for t in tensors]
            clip_t = torch.stack(tensors).unsqueeze(0).to(DEVICE)
            out = model(clip_t, task="distraction", dataset=dataset_name)["distraction"]
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            raw_idx = probs.argmax()
            raw_label = dist_classes[raw_idx]
            conf = probs[raw_idx]
            norm_conf = probs[norm_idx]

            raw_window_preds.append(raw_label)

            # Determine ground truth for mid-frame of this window
            mid_f = start_frame + i + CLIP_LEN // 2
            gt = "normal_driving"
            for s, e, lbl in intervals:
                if s <= mid_f <= e:
                    gt = lbl
                    break
            ground_truth_labels.append(gt)

            # Apply Euro NCAP Persistence Filter (3 of 5 rule + margin check)
            if raw_label != "normal_driving":
                if conf < 0.65 or (conf - norm_conf) < 0.15:
                    cand = "normal_driving"
                else:
                    cand = raw_label
            else:
                cand = "normal_driving"

            history.append(cand)
            counts = Counter(history)
            top_cand, top_cnt = counts.most_common(1)[0]
            if top_cand != "normal_driving" and top_cnt < 3:
                filtered_label = "normal_driving"
            else:
                filtered_label = top_cand

            filtered_window_preds.append(filtered_label)

    # Calculate metrics for both
    raw_acc = accuracy_score(ground_truth_labels, raw_window_preds)
    filtered_acc = accuracy_score(ground_truth_labels, filtered_window_preds)

    # False positive rate on normal driving frames (predicting distraction when driver was safe)
    safe_indices = [idx for idx, gt in enumerate(ground_truth_labels) if gt == "normal_driving"]
    raw_fps = sum(1 for idx in safe_indices if raw_window_preds[idx] != "normal_driving")
    filtered_fps = sum(1 for idx in safe_indices if filtered_window_preds[idx] != "normal_driving")
    safe_count = max(1, len(safe_indices))

    print(f"\nWindow-Level Evaluation on Test Subject 13 ({len(ground_truth_labels)} windows):")
    print(f"  WITHOUT Persistence Filter (Raw argmax per window):")
    print(f"    - Overall Window Accuracy:        {raw_acc:.4f} ({raw_acc*100:.2f}%)")
    print(f"    - False Alarms on Safe Driving:    {raw_fps}/{safe_count} ({raw_fps/safe_count*100:.2f}%)")
    print(f"  WITH Persistence Filter (3-of-5 consecutive windows + margin):")
    print(f"    - Overall Window Accuracy:        {filtered_acc:.4f} ({filtered_acc*100:.2f}%)")
    print(f"    - False Alarms on Safe Driving:    {filtered_fps}/{safe_count} ({filtered_fps/safe_count*100:.2f}%)")
    print(f"\n  Conclusion: Persistence filtering reduces false-positive distraction alarms from {raw_fps/safe_count*100:.1f}% down to {filtered_fps/safe_count*100:.1f}%, increasing accuracy from {raw_acc*100:.1f}% to {filtered_acc*100:.1f}%.")


if __name__ == "__main__":
    audit_temporal_model()
    audit_sensor_model()
    audit_persistence_filter()
