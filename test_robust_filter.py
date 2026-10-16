import os
import sys
import cv2
import torch
import numpy as np
from collections import deque, Counter
from torchvision import transforms

from src.models.temporal_model import TemporalMultiTaskModel
from src.ood_detector import MahalanobisOODDetector
from src.utils import get_device, load_taxonomy, build_label_encoders

device = get_device()
tax = load_taxonomy('configs/taxonomy.yaml')
enc = build_label_encoders(tax)
classes = list(enc['distraction'].keys())
print('Distraction Classes:', classes, flush=True)

t_ckpt = torch.load('checkpoints/temporal_distraction_dmd_FINAL.pt', map_location=device)
model = TemporalMultiTaskModel(t_ckpt['num_classes'], t_ckpt['domains'], backbone='resnet18')
model.load_state_dict(t_ckpt['model_state'])
model.to(device).eval()

ood = MahalanobisOODDetector()
ood.load('checkpoints/ood_temporal_distraction_dmd.npz')

infer_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

video_path = 'dmd_s7_body_test.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 29.8
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Streaming: {video_path}, {total_frames} frames, {fps:.1f} fps', flush=True)

CLIP_LEN = 16
CLIP_STRIDE = 8

frame_buf = deque(maxlen=CLIP_LEN)
raw_windows = []
f_idx = 0

# Limit to first 1500 frames (~50 seconds) to test 0-25s (gear shifts) and 28-42s (radio)
MAX_FRAMES = 1500

while f_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize on read (320x180) to consume virtually zero RAM
    frame_small = cv2.resize(frame, (320, 180))
    frame_buf.append(frame_small)
    
    if len(frame_buf) == CLIP_LEN and (f_idx - CLIP_LEN + 1) % CLIP_STRIDE == 0:
        win_tensors = [infer_tf(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frame_buf]
        clip = torch.stack(win_tensors).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model.get_clip_features(clip, task='distraction', dataset='dmd').cpu().numpy()[0]
            out = model(clip, task='distraction', dataset='dmd')['distraction']
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        p_idx = probs.argmax()
        is_ood, dist, _ = ood.is_out_of_distribution(feats)
        start_t = (f_idx - CLIP_LEN + 1) / fps
        raw_windows.append({
            't': start_t,
            'raw_pred': classes[p_idx],
            'conf': float(probs[p_idx]),
            'probs': probs,
            'ood': is_ood,
            'dist': float(dist)
        })
    f_idx += 1
cap.release()

print(f'Total evaluated windows: {len(raw_windows)}', flush=True)

def apply_robust_filter(windows, conf_threshold=0.65, margin_over_normal=0.15):
    smoothed = []
    history = deque(maxlen=5)
    
    norm_idx = classes.index('normal_driving')
    
    for w in windows:
        raw = w['raw_pred']
        conf = w['conf']
        probs = w['probs']
        norm_conf = probs[norm_idx]
        
        # 1. Distraction validation:
        # A distraction must beat normal_driving by a confident margin,
        # otherwise transient hand adjustments default to normal_driving.
        if raw != 'normal_driving':
            if conf < conf_threshold or (conf - norm_conf) < margin_over_normal:
                candidate = 'normal_driving'
            else:
                candidate = raw
        else:
            candidate = 'normal_driving'
            
        history.append(candidate)
        
        # 2. Temporal persistence (Euro NCAP requirement):
        # A distraction must persist for at least 3 out of the last 5 windows (~0.8s - 1.3s)
        counts = Counter(history)
        top_cand, top_cnt = counts.most_common(1)[0]
        
        if top_cand != 'normal_driving' and top_cnt < 3:
            final_label = 'normal_driving'
        else:
            final_label = top_cand
            
        smoothed.append(final_label)
        
    return smoothed

smoothed = apply_robust_filter(raw_windows)

print('\n========================================================================', flush=True)
print(' COMPARISON: 0s to 25s (NORMAL DRIVING & GEAR SHIFTS)', flush=True)
print('========================================================================', flush=True)
for i, w in enumerate(raw_windows):
    if w['t'] > 25.0:
        break
    t_sec = w['t']
    r = w['raw_pred']
    c = w['conf']
    s = smoothed[i]
    diff_marker = ' *** FIXED ***' if r != s else ''
    if r != 'normal_driving' or s != 'normal_driving':
        print(f"t={t_sec:4.1f}s | Raw: {r:26s} ({c:5.1%}) -> Filtered: {s:26s}{diff_marker}", flush=True)

print('\n========================================================================', flush=True)
print(' COMPARISON: 28s to 42s (ACTUAL RADIO / CONSOLE ADJUSTMENT)', flush=True)
print('========================================================================', flush=True)
for i, w in enumerate(raw_windows):
    if 28.0 <= w['t'] <= 42.0:
        t_sec = w['t']
        r = w['raw_pred']
        c = w['conf']
        s = smoothed[i]
        print(f"t={t_sec:4.1f}s | Raw: {r:26s} ({c:5.1%}) -> Filtered: {s:26s}", flush=True)

print('\nSummary in 0-50s clip:', flush=True)
print('Raw predictions:     ', Counter(w['raw_pred'] for w in raw_windows), flush=True)
print('Filtered predictions:', Counter(smoothed), flush=True)
