"""
Manifest Builder Template
==========================
Converts each raw dataset's folder/label format into a common CSV manifest.

Expected manifest CSV columns:
    filepath        - path to the image
    task             - one of: distraction | drowsiness | aggression
    label            - the UNIFIED label (must match configs/taxonomy.yaml)
    source_dataset   - e.g. "state_farm", "yawdd"
    split            - "train" | "val" | "test"

Usage:
    python src/preprocessing/build_manifest_template.py --dataset state_farm \
        --raw_dir C:\\path\\to\\state-farm\\imgs\\train \
        --out manifests/state_farm_manifest.csv
"""

import argparse
import os
import csv
import random
import yaml


def load_mapping(dataset_name: str, taxonomy_path: str = "configs/taxonomy.yaml"):
    with open(taxonomy_path, "r") as f:
        taxonomy = yaml.safe_load(f)
    entry = taxonomy["dataset_mappings"][dataset_name]
    return entry["task"], entry["mapping"]


def train_val_test_split(items, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return train, val, test


def build_state_farm_manifest(raw_dir: str, driver_list_path: str, out_path: str,
                                val_frac=0.15, test_frac=0.15, seed=42):
    """
    Splits by DRIVER (subject), not by individual image, to avoid leakage
    from near-duplicate consecutive video frames of the same driver
    appearing in both train and val/test.

    raw_dir expected structure:
        raw_dir/c0/*.jpg ... raw_dir/c9/*.jpg
    driver_list_path: path to driver_imgs_list.csv (columns: subject, classname, img)
    """
    import pandas as pd

    task, mapping = load_mapping("state_farm")
    driver_df = pd.read_csv(driver_list_path)

    # Split DRIVERS (not images) into train/val/test
    drivers = sorted(driver_df["subject"].unique())
    rng = random.Random(seed)
    rng.shuffle(drivers)
    n = len(drivers)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    val_drivers = set(drivers[:n_val])
    test_drivers = set(drivers[n_val:n_val + n_test])
    train_drivers = set(drivers[n_val + n_test:])

    print(f"[state_farm] {len(train_drivers)} train drivers, "
          f"{len(val_drivers)} val drivers, {len(test_drivers)} test drivers")
    print(f"[state_farm] val drivers: {sorted(val_drivers)}")
    print(f"[state_farm] test drivers: {sorted(test_drivers)}")

    def split_for(subject):
        if subject in val_drivers:
            return "val"
        if subject in test_drivers:
            return "test"
        return "train"

    rows = []
    missing = 0
    for _, r in driver_df.iterrows():
        class_folder = r["classname"]
        if class_folder not in mapping:
            continue
        filepath = os.path.join(raw_dir, class_folder, r["img"])
        if not os.path.isfile(filepath):
            missing += 1
            continue
        rows.append({
            "filepath": filepath,
            "task": task,
            "label": mapping[class_folder],
            "source_dataset": "state_farm",
            "split": split_for(r["subject"]),
        })

    if missing:
        print(f"[warn] {missing} images listed in driver CSV were not found on disk")

    _write_manifest(rows, out_path)
    print(f"[state_farm] wrote {len(rows)} rows -> {out_path}")

def build_yawdd_manifest(mirror_root: str, out_path: str, frames_out_dir: str,
                           yunet_model_path: str = "face_detection_yunet_2026may.onnx",
                           landmarker_model_path: str = "face_landmarker.task",
                           every_n_frames=15, yawn_scan_every_n=3, mar_threshold=0.5,
                           max_yawn_frames_per_video=8, face_margin=0.6,
                           val_frac=0.15, test_frac=0.15, seed=42):
    """
    Processes YawDD's Mirror videos ONLY.

    Key fix: naively sampling every N frames from a "Yawning" video labels
    mostly-closed-mouth frames as yawning (a real yawn is ~1-2s within a
    15-25s clip). Instead, for Yawning/TalkingYawning videos, we densely
    scan for the actual yawn peak using Mouth Aspect Ratio (MAR) from
    MediaPipe face landmarks, and only keep frames where MAR exceeds a
    calibrated threshold -- i.e. frames that actually show a yawn.

    Normal/Talking videos (-> alert) don't have this problem (the mouth
    stays roughly closed/normal throughout) so they keep the simpler
    fixed-interval sampling.
    """
    import cv2
    import re
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker, RunningMode
    from mediapipe.tasks import python as mp_tasks

    behavior_map = {
        "normal": "alert",
        "talking": "alert",
        "yawning": "microsleep_yawning",
        "talkingyawning": "microsleep_yawning",
    }

    detector = cv2.FaceDetectorYN.create(yunet_model_path, "", (320, 320), 0.7, 0.3, 5000)
    base_options = mp_tasks.BaseOptions(model_asset_path=landmarker_model_path)
    landmarker_options = FaceLandmarkerOptions(base_options=base_options,
                                                running_mode=RunningMode.IMAGE, num_faces=1)
    landmarker = FaceLandmarker.create_from_options(landmarker_options)

    def crop_face(frame, margin=face_margin):
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])
        x, y, fw, fh = best[0], best[1], best[2], best[3]
        mx, my = fw * margin, fh * margin
        x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
        x1, y1 = min(w, int(x + fw + mx)), min(h, int(y + fh + my))
        crop = frame[y0:y1, x0:x1]
        return crop if crop.size > 0 else None

    def compute_mar(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if len(result.face_landmarks) == 0:
            return None
        lm = result.face_landmarks[0]
        vertical = abs(lm[14].y - lm[13].y)
        horizontal = abs(lm[308].x - lm[78].x)
        return vertical / horizontal if horizontal > 0 else 0

    subfolders = ["Female_mirror", "Male_mirror Avi Videos"]
    video_files = []
    for sub in subfolders:
        folder = os.path.join(mirror_root, sub)
        if not os.path.isdir(folder):
            print(f"[warn] missing folder: {folder}")
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(".avi"):
                video_files.append(os.path.join(folder, fname))

    print(f"[yawdd] found {len(video_files)} Mirror video files")

    parsed = []
    skipped = 0
    for path in video_files:
        fname = os.path.basename(path)
        name_no_ext = re.sub(r"\.avi$", "", fname, flags=re.IGNORECASE)
        parts = name_no_ext.split("-")
        if len(parts) < 3:
            skipped += 1
            continue
        subject = parts[0]
        behavior_raw = parts[-1].lower()
        if behavior_raw not in behavior_map:
            skipped += 1
            continue
        parsed.append({
            "path": path, "subject": subject,
            "unified_label": behavior_map[behavior_raw],
            "is_yawn_video": behavior_raw in ("yawning", "talkingyawning"),
        })

    print(f"[yawdd] parsed {len(parsed)} videos, skipped {skipped}")

    subjects = sorted(set(p["subject"] for p in parsed), key=lambda s: int(s))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    val_subjects = set(subjects[:n_val])
    test_subjects = set(subjects[n_val:n_val + n_test])
    print(f"[yawdd] {n - n_val - n_test} train / {n_val} val / {n_test} test subjects")

    def split_for(subject):
        if subject in val_subjects:
            return "val"
        if subject in test_subjects:
            return "test"
        return "train"

    os.makedirs(frames_out_dir, exist_ok=True)
    rows = []
    total_no_face = 0
    for i, item in enumerate(parsed):
        video_path = item["path"]
        cap = cv2.VideoCapture(video_path)
        base = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(os.path.basename(video_path))[0])
        split = split_for(item["subject"])
        idx, saved, no_face, yawn_frames_kept = 0, 0, 0, 0

        scan_interval = yawn_scan_every_n if item["is_yawn_video"] else every_n_frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % scan_interval == 0:
                keep_this_frame = True
                if item["is_yawn_video"]:
                    if yawn_frames_kept >= max_yawn_frames_per_video:
                        idx += 1
                        continue
                    mar = compute_mar(frame)
                    keep_this_frame = (mar is not None and mar > mar_threshold)

                if keep_this_frame:
                    crop = crop_face(frame)
                    if crop is None:
                        no_face += 1
                    else:
                        frame_fname = f"{base}_frame{idx:06d}.jpg"
                        frame_path = os.path.join(frames_out_dir, frame_fname)
                        cv2.imwrite(frame_path, crop)
                        rows.append({
                            "filepath": frame_path, "task": "drowsiness",
                            "label": item["unified_label"],
                            "source_dataset": "yawdd", "split": split,
                        })
                        saved += 1
                        if item["is_yawn_video"]:
                            yawn_frames_kept += 1
            idx += 1
        cap.release()
        total_no_face += no_face
        if (i + 1) % 20 == 0 or (i + 1) == len(parsed):
            print(f"[yawdd] processed {i+1}/{len(parsed)} videos, "
                  f"{len(rows)} frames extracted so far")

    _write_manifest(rows, out_path)
    print(f"[yawdd] wrote {len(rows)} rows -> {out_path}")
    print(f"[yawdd] total frames skipped due to no detected face: {total_no_face}")

def build_uah_driveset_manifest(root_dir: str, out_path: str,
                                  skip_seconds=60, val_frac=0.2, test_frac=0.2, seed=42):
    """
    Processes UAH-DriveSet's SEMANTIC_ONLINE.txt files.

    Unlike the image datasets, this manifest stores numeric sensor
    features directly (no image file to reference) -- one row per
    1-second sample, skipping the first `skip_seconds` of each session
    since the 60-second rolling-window scores aren't meaningful yet.

    root_dir expected structure:
        root_dir/D1/{timestamp}-{km}km-D1-{BEHAVIOR}-{ROADTYPE}/SEMANTIC_ONLINE.txt
        root_dir/D2/...
        ...

    Behavior in folder name maps to unified aggression labels. Only
    NORMAL/NORMAL1/NORMAL2 -> normal and AGGRESSIVE -> aggressive are
    used for the aggression task; DROWSY sessions exist but are session-
    level driving-style labels (not visual drowsiness), so they are
    recorded with source_dataset='uah_driveset' but task='aggression_drowsy'
    so they don't get silently mixed into the vision-based drowsiness task.
    """
    import re

    behavior_map = {
        "normal": "normal",
        "normal1": "normal",
        "normal2": "normal",
        "aggressive": "aggressive",
        "drowsy": "drowsy",  # kept separate, see docstring
    }

    driver_folders = [d for d in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, d)) and re.match(r"^D\d+$", d)]
    driver_folders.sort(key=lambda d: int(d[1:]))
    print(f"[uah_driveset] found {len(driver_folders)} driver folders: {driver_folders}")

    rng = random.Random(seed)
    drivers_shuffled = driver_folders[:]
    rng.shuffle(drivers_shuffled)
    n = len(drivers_shuffled)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    val_drivers = set(drivers_shuffled[:n_val])
    test_drivers = set(drivers_shuffled[n_val:n_val + n_test])
    print(f"[uah_driveset] {n - n_val - n_test} train / {n_val} val / {n_test} test drivers")
    print(f"[uah_driveset] val drivers: {sorted(val_drivers)}, test drivers: {sorted(test_drivers)}")

    def split_for(driver):
        if driver in val_drivers:
            return "val"
        if driver in test_drivers:
            return "test"
        return "train"

    rows = []
    skipped_sessions = 0
    for driver in driver_folders:
        driver_path = os.path.join(root_dir, driver)
        split = split_for(driver)
        session_folders = [s for s in os.listdir(driver_path)
                            if os.path.isdir(os.path.join(driver_path, s))]

        for session in session_folders:
            parts = session.split("-")
            if len(parts) < 5:
                skipped_sessions += 1
                continue
            behavior_raw = parts[3].lower()
            road_type = parts[4].lower()
            if behavior_raw not in behavior_map:
                print(f"[warn] unknown behavior '{behavior_raw}' in session {session}, skipping")
                skipped_sessions += 1
                continue

            semantic_path = os.path.join(driver_path, session, "SEMANTIC_ONLINE.txt")
            if not os.path.isfile(semantic_path):
                print(f"[warn] missing SEMANTIC_ONLINE.txt in {session}, skipping")
                skipped_sessions += 1
                continue

            unified_label = behavior_map[behavior_raw]
            task = "aggression" if unified_label in ("normal", "aggressive") else "aggression_drowsy"

            with open(semantic_path, "r") as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) != 27:
                        continue
                    timestamp_sec = float(vals[0])
                    if timestamp_sec < skip_seconds:
                        continue
                    rows.append({
                        "score_accel": float(vals[16]),
                        "score_brake": float(vals[17]),
                        "score_turn": float(vals[18]),
                        "score_weave": float(vals[19]),
                        "score_drift": float(vals[20]),
                        "score_overspeed": float(vals[21]),
                        "score_carfollow": float(vals[22]),
                        "task": task,
                        "label": unified_label,
                        "source_dataset": "uah_driveset",
                        "driver": driver,
                        "road_type": road_type,
                        "split": split,
                    })

    print(f"[uah_driveset] {skipped_sessions} sessions skipped, {len(rows)} total rows")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["score_accel", "score_brake", "score_turn", "score_weave", "score_drift",
                  "score_overspeed", "score_carfollow", "task", "label", "source_dataset",
                  "driver", "road_type", "split"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[uah_driveset] wrote {len(rows)} rows -> {out_path}")

def build_dmd_manifest(root_dir: str, out_path: str, frames_out_dir: str,
                         every_n_frames=10, val_frac=0.2, test_frac=0.2, seed=42):
    """
    Processes DMD (Driver Monitoring Dataset) body-camera videos for
    distraction detection.

    Unlike State Farm/YawDD, DMD's JSON annotations give PRECISE
    frame-level ground truth (exact start/end frame of each labeled
    action) -- no video-to-frame label noise problem here, since every
    frame within a labeled interval genuinely shows that behavior.

    root_dir expected structure:
        root_dir/{subject_id}/s1/{...}_rgb_ann_distraction.json
        root_dir/{subject_id}/s1/{...}_rgb_body.mp4
        root_dir/{subject_id}/s2/...
        root_dir/{subject_id}/s3/...
    (root_dir points at a single group folder, e.g. D:\\DMD\\gA)
    """
    import cv2
    import json
    import glob
    import re

    task, mapping = load_mapping("dmd_distraction")

    subject_folders = [d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    subject_folders.sort(key=int)
    print(f"[dmd] found {len(subject_folders)} subjects: {subject_folders}")

    rng = random.Random(seed)
    subjects_shuffled = subject_folders[:]
    rng.shuffle(subjects_shuffled)
    n = len(subjects_shuffled)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    val_subjects = set(subjects_shuffled[:n_val])
    test_subjects = set(subjects_shuffled[n_val:n_val + n_test])
    print(f"[dmd] {n - n_val - n_test} train / {n_val} val / {n_test} test subjects")
    print(f"[dmd] val: {sorted(val_subjects)}, test: {sorted(test_subjects)}")

    def split_for(subject):
        if subject in val_subjects:
            return "val"
        if subject in test_subjects:
            return "test"
        return "train"

    os.makedirs(frames_out_dir, exist_ok=True)
    rows = []
    skipped_unmapped = 0

    for subject in subject_folders:
        split = split_for(subject)
        subject_path = os.path.join(root_dir, subject)
        session_folders = [s for s in os.listdir(subject_path)
                            if os.path.isdir(os.path.join(subject_path, s))]

        for session in session_folders:
            session_path = os.path.join(subject_path, session)
            json_matches = glob.glob(os.path.join(session_path, "*_ann_distraction.json"))
            body_matches = glob.glob(os.path.join(session_path, "*_rgb_body.mp4"))
            if not json_matches or not body_matches:
                print(f"[warn] missing json or body video in {session_path}, skipping")
                continue

            with open(json_matches[0], "r") as f:
                data = json.load(f)
            actions = data["openlabel"]["actions"]

            # Build: frame_idx -> unified_label, only for driver_actions/* entries
            frame_to_label = {}
            for action_id, a in actions.items():
                action_type = a["type"]
                if not action_type.startswith("driver_actions/"):
                    continue
                raw_label = action_type.split("/", 1)[1]
                if raw_label not in mapping:
                    skipped_unmapped += 1
                    continue
                unified_label = mapping[raw_label]
                effective_stride = every_n_frames

                for interval in a["frame_intervals"]:
                    start, end = interval["frame_start"], interval["frame_end"]
                    length = end - start + 1
                    trim_start_amount = int(length * 0.15)
                    trim_end_amount = int(length * 0.35)
                    trimmed_start = start + trim_start_amount
                    trimmed_end = end - trim_end_amount
                    if trimmed_start >= trimmed_end:
                        mid = (start + end) // 2
                        frame_to_label[mid] = unified_label
                        continue
                    for f_idx in range(trimmed_start, trimmed_end + 1, effective_stride):
                        frame_to_label[f_idx] = unified_label

            if not frame_to_label:
                continue

            # Extract only the frames we need, reading sequentially
            cap = cv2.VideoCapture(body_matches[0])
            base = re.sub(r"[^A-Za-z0-9_-]", "_",
                           os.path.splitext(os.path.basename(body_matches[0]))[0])
            wanted_frames = frame_to_label  # dict, frame_idx -> label
            idx = 0
            saved = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in wanted_frames:
                    frame_fname = f"dmd_{subject}_{session}_{base}_frame{idx:06d}.jpg"
                    frame_path = os.path.join(frames_out_dir, frame_fname)
                    cv2.imwrite(frame_path, frame)
                    rows.append({
                        "filepath": frame_path,
                        "task": task,
                        "label": wanted_frames[idx],
                        "source_dataset": "dmd",
                        "split": split,
                    })
                    saved += 1
                idx += 1
            cap.release()
            print(f"[dmd] subject {subject} {session}: saved {saved} frames")

    if skipped_unmapped:
        print(f"[dmd] skipped {skipped_unmapped} action entries with unmapped labels")

    _write_manifest(rows, out_path)
    print(f"[dmd] wrote {len(rows)} rows -> {out_path}")

def build_dmd_video_manifest(root_dir: str, out_path: str, clips_out_dir: str,
                               clip_len=16, clip_stride=8, val_frac=0.2, test_frac=0.2, seed=42):
    """
    Extracts short OVERLAPPING video clips (not single frames) from DMD's
    labeled intervals, for temporal/motion-based classification. Each
    clip is clip_len consecutive frames, saved as its own folder of
    numbered JPGs (frame_00.jpg ... frame_15.jpg by default).

    Motivation: single-frame classification can't distinguish behaviors
    that look visually similar in a static pose but differ in MOTION
    (e.g. hand moving toward the ear vs. down toward the lap --
    phone_call vs texting vs reaching_behind were confirmed, via
    repeated evaluation, to be confused by the frame-based model).

    Same interval trimming (15% start / 35% end) as build_dmd_manifest,
    for the same reason (avoid ambiguous transition moments).

    root_dir expected structure: same as build_dmd_manifest.
    """
    import cv2
    import json
    import glob
    import re

    task, mapping = load_mapping("dmd_distraction")

    subject_folders = [d for d in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()]
    subject_folders.sort(key=int)
    print(f"[dmd_video] found {len(subject_folders)} subjects: {subject_folders}")

    rng = random.Random(seed)
    subjects_shuffled = subject_folders[:]
    rng.shuffle(subjects_shuffled)
    n = len(subjects_shuffled)
    n_val = max(1, round(n * val_frac))
    n_test = max(1, round(n * test_frac))
    val_subjects = set(subjects_shuffled[:n_val])
    test_subjects = set(subjects_shuffled[n_val:n_val + n_test])
    print(f"[dmd_video] {n - n_val - n_test} train / {n_val} val / {n_test} test subjects")
    print(f"[dmd_video] val: {sorted(val_subjects)}, test: {sorted(test_subjects)}")

    def split_for(subject):
        if subject in val_subjects:
            return "val"
        if subject in test_subjects:
            return "test"
        return "train"

    os.makedirs(clips_out_dir, exist_ok=True)
    rows = []
    clip_counter = 0
    skipped_unmapped = 0

    for subject in subject_folders:
        split = split_for(subject)
        subject_path = os.path.join(root_dir, subject)
        session_folders = [s for s in os.listdir(subject_path)
                            if os.path.isdir(os.path.join(subject_path, s))]

        for session in session_folders:
            session_path = os.path.join(subject_path, session)
            json_matches = glob.glob(os.path.join(session_path, "*_ann_distraction.json"))
            body_matches = glob.glob(os.path.join(session_path, "*_rgb_body.mp4"))
            if not json_matches or not body_matches:
                continue

            with open(json_matches[0], "r") as f:
                data = json.load(f)
            actions = data["openlabel"]["actions"]

            # Build list of (start_frame, end_frame, label) trimmed intervals
            trimmed_intervals = []
            for action_id, a in actions.items():
                action_type = a["type"]
                if not action_type.startswith("driver_actions/"):
                    continue
                raw_label = action_type.split("/", 1)[1]
                if raw_label not in mapping:
                    skipped_unmapped += 1
                    continue
                unified_label = mapping[raw_label]
                for interval in a["frame_intervals"]:
                    start, end = interval["frame_start"], interval["frame_end"]
                    length = end - start + 1
                    trim_start_amount = int(length * 0.15)
                    trim_end_amount = int(length * 0.35)
                    trimmed_start = start + trim_start_amount
                    trimmed_end = end - trim_end_amount
                    if trimmed_end - trimmed_start + 1 < clip_len:
                        continue  # interval too short for even one clip
                    trimmed_intervals.append((trimmed_start, trimmed_end, unified_label))

            if not trimmed_intervals:
                continue

            # Determine clip start frames (overlapping windows) for each interval
            clips_to_extract = []  # (clip_start, clip_end, label)
            for t_start, t_end, label in trimmed_intervals:
                pos = t_start
                while pos + clip_len - 1 <= t_end:
                    clips_to_extract.append((pos, pos + clip_len - 1, label))
                    pos += clip_stride

            if not clips_to_extract:
                continue

            # Map frame_idx -> list of (clip_id, position_in_clip) so we can
            # fill clips while reading the video sequentially just once
            frame_to_clip_slots = {}
            clip_frame_buffers = {}
            base = re.sub(r"[^A-Za-z0-9_-]", "_",
                           os.path.splitext(os.path.basename(body_matches[0]))[0])

            for i, (c_start, c_end, label) in enumerate(clips_to_extract):
                clip_id = f"dmd_{subject}_{session}_{base}_clip{clip_counter}"
                clip_counter += 1
                clip_frame_buffers[clip_id] = {"label": label, "frames": {}}
                for pos_in_clip, f_idx in enumerate(range(c_start, c_end + 1)):
                    frame_to_clip_slots.setdefault(f_idx, []).append((clip_id, pos_in_clip))

            cap = cv2.VideoCapture(body_matches[0])
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx in frame_to_clip_slots:
                    # Resize to 256x256 IMMEDIATELY on read, before buffering.
                    # Do NOT buffer full-resolution frames (e.g. 1280x720) across
                    # many overlapping clips -- this was empirically found to consume
                    # multiple GB of RAM per session, causing Windows page-file
                    # exhaustion and OOM crashes. Resize-on-read, not resize-on-write.
                    frame_small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                    for clip_id, pos_in_clip in frame_to_clip_slots[idx]:
                        clip_frame_buffers[clip_id]["frames"][pos_in_clip] = frame_small
                idx += 1
            cap.release()

            for clip_id, info in clip_frame_buffers.items():
                if len(info["frames"]) != clip_len:
                    continue  # incomplete clip (shouldn't normally happen), skip
                clip_dir = os.path.join(clips_out_dir, clip_id)
                os.makedirs(clip_dir, exist_ok=True)
                for pos_in_clip in range(clip_len):
                    frame_path = os.path.join(clip_dir, f"frame_{pos_in_clip:02d}.jpg")
                    cv2.imwrite(frame_path, info["frames"][pos_in_clip])
                rows.append({
                    "filepath": clip_dir,  # NOTE: for clip manifests, this is a FOLDER
                    "task": task,
                    "label": info["label"],
                    "source_dataset": "dmd",
                    "split": split,
                })

        print(f"[dmd_video] subject {subject}: {clip_counter} total clips so far")

    if skipped_unmapped:
        print(f"[dmd_video] skipped {skipped_unmapped} action entries with unmapped labels")

    _write_manifest(rows, out_path)
    print(f"[dmd_video] wrote {len(rows)} clip rows -> {out_path}")

def _write_manifest(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["filepath", "task", "label", "source_dataset", "split"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                         choices=["state_farm", "yawdd", "uah_driveset", "dmd"])
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--driver_list", help="Path to driver_imgs_list.csv (state_farm only)")
    parser.add_argument("--frames_out_dir", help="Where extracted frames are saved (yawdd/dmd only)")
    parser.add_argument("--yunet_model", default="face_detection_yunet_2026may.onnx",
                         help="Path to YuNet ONNX model (yawdd only)")
    parser.add_argument("--every_n_frames", type=int, default=15)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.dataset == "state_farm":
        build_state_farm_manifest(args.raw_dir, args.driver_list, args.out)
    elif args.dataset == "yawdd":
        if not args.frames_out_dir:
            parser.error("--frames_out_dir is required for yawdd")
        build_yawdd_manifest(args.raw_dir, args.out, args.frames_out_dir,
                              yunet_model_path=args.yunet_model,
                              every_n_frames=args.every_n_frames)
    elif args.dataset == "uah_driveset":
        build_uah_driveset_manifest(args.raw_dir, args.out)
    elif args.dataset == "dmd":
        if not args.frames_out_dir:
            parser.error("--frames_out_dir is required for dmd")
        build_dmd_manifest(args.raw_dir, args.out, args.frames_out_dir,
                            every_n_frames=args.every_n_frames)