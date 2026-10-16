"""
build_all_behavior_clips.py
===========================
Extracts authentic, continuous, UNCUT driving sessions (20s - 45s) from:
- DMD dataset (body-camera recording sessions, 100% uncut single-shot takes)
- YawDD dataset (full natural mirror camera fatigue/alert recordings)

Outputs clean, raw, unedited H.264 MP4 videos to demo_clips/ for real-time testing.
"""

import os
import sys
import cv2
import subprocess
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
OUT_DIR = "demo_clips"
os.makedirs(OUT_DIR, exist_ok=True)

CONTINUOUS_DRIVES = [
    {
        "id": "01_uncut_drive_gearshift_and_radio",
        "title": "Continuous Drive (Safe + Gear Shift + Radio)",
        "source": r"E:\nare\DMD\gA\7\s1\gB_7_s1_2019-03-11T14;22;01+01;00_rgb_body.mp4",
        "start_frame": 400,
        "num_frames": 850,  # ~28.5s
    },
    {
        "id": "02_uncut_drive_texting_righthand",
        "title": "Continuous Drive (Safe Driving + Right-Hand Texting)",
        "source": r"E:\nare\DMD\gA\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_body.mp4",
        "start_frame": 2650,
        "num_frames": 850,  # ~28.5s
    },
    {
        "id": "03_uncut_drive_phonecall_righthand",
        "title": "Continuous Drive (Safe Driving + Right-Hand Phone Call)",
        "source": r"E:\nare\DMD\gA\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_body.mp4",
        "start_frame": 1550,
        "num_frames": 900,  # ~30.2s
    },
    {
        "id": "04_uncut_drive_drinking_bottle",
        "title": "Continuous Drive (Safe Driving + Drinking from Bottle)",
        "source": r"E:\nare\DMD\gA\7\s1\gB_7_s1_2019-03-11T14;22;01+01;00_rgb_body.mp4",
        "start_frame": 1450,
        "num_frames": 650,  # ~21.8s
    },
    {
        "id": "05_uncut_drive_texting_lefthand",
        "title": "Continuous Drive (Safe Driving + Left-Hand Texting)",
        "source": r"E:\nare\DMD\gA\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_body.mp4",
        "start_frame": 9700,
        "num_frames": 900,  # ~30.2s
    },
    {
        "id": "06_uncut_drive_phonecall_lefthand",
        "title": "Continuous Drive (Safe Driving + Left-Hand Phone Call)",
        "source": r"E:\nare\DMD\gA\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_body.mp4",
        "start_frame": 7600,
        "num_frames": 900,  # ~30.2s
    },
    {
        "id": "07_uncut_drive_talking_passenger",
        "title": "Continuous Drive (Safe Driving + Talking to Passenger)",
        "source": r"E:\nare\DMD\gA\10\s1\gB_10_s1_2019-03-11T15;24;54+01;00_rgb_body.mp4",
        "start_frame": 1700,
        "num_frames": 850,  # ~28.5s
    },
    {
        "id": "08_uncut_drive_reaching_backseat",
        "title": "Continuous Drive (Safe Driving + Reaching into Backseat)",
        "source": r"E:\nare\DMD\gA\7\s3\gB_7_s3_2019-03-13T13;59;54+01;00_rgb_body.mp4",
        "start_frame": 450,
        "num_frames": 700,  # ~23.5s
    },
    {
        "id": "09_uncut_drive_hair_and_makeup",
        "title": "Continuous Drive (Safe Driving + Mirror Check & Hair Grooming)",
        "source": r"E:\nare\DMD\gA\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_body.mp4",
        "start_frame": 650,
        "num_frames": 780,  # ~26.2s
    },
    {
        "id": "10_uncut_drowsiness_female_yawn",
        "title": "Continuous Fatigue Recording (Female Driver Yawning)",
        "source": r"C:\Users\venka\Downloads\yawdd\Mirror\Mirror\Female_mirror\1-FemaleNoGlasses-Yawning.avi",
        "start_frame": 0,
        "num_frames": 387,  # full video ~12.9s
    },
    {
        "id": "11_uncut_drowsiness_male_yawn",
        "title": "Continuous Fatigue Recording (Male Driver Yawning)",
        "source": r"C:\Users\venka\Downloads\yawdd\Mirror\Mirror\Male_mirror Avi Videos\1-MaleNoGlasses-Yawning.avi",
        "start_frame": 0,
        "num_frames": 441,  # full video ~14.7s
    },
    {
        "id": "12_uncut_alert_driver_talking",
        "title": "Continuous Alert Recording (Alert Driver Conversing)",
        "source": r"C:\Users\venka\Downloads\yawdd\Mirror\Mirror\Female_mirror\1-FemaleNoGlasses-Talking.avi",
        "start_frame": 0,
        "num_frames": 560,  # full video ~18.7s
    },
    {
        "id": "13_master_long_drive_session",
        "title": "Extended Continuous Drive (Full 45s Multi-Behavior Trip)",
        "source": r"E:\nare\DMD\gA\7\s1\gB_7_s1_2019-03-11T14;22;01+01;00_rgb_body.mp4",
        "start_frame": 450,
        "num_frames": 1350,  # ~45.3s
    },
]


def extract_continuous_clip(spec):
    out_final = os.path.join(OUT_DIR, f"{spec['id']}.mp4")
    raw_tmp = os.path.join(OUT_DIR, f"_raw_{spec['id']}.mp4")

    if not os.path.exists(spec["source"]):
        print(f"  [!] Source missing for {spec['id']}: {spec['source']}", flush=True)
        return None

    cap = cv2.VideoCapture(spec["source"])
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 29.8
    if src_fps > 45 or src_fps < 10:
        src_fps = 29.8

    cap.set(cv2.CAP_PROP_POS_FRAMES, spec["start_frame"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Clean 640x360 raw video, memory safe and fast to stream
    writer = cv2.VideoWriter(raw_tmp, fourcc, src_fps, (640, 360))

    frames_written = 0
    while frames_written < spec["num_frames"]:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        writer.write(frame_resized)
        frames_written += 1

    cap.release()
    writer.release()

    # Re-encode to H.264 / yuv420p for fast web playback
    cmd = [
        FFMPEG, "-y", "-i", raw_tmp,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "22",
        out_final
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_tmp):
        os.remove(raw_tmp)

    dur_sec = frames_written / src_fps
    size_mb = os.path.getsize(out_final) / (1024 * 1024)
    print(f"  [+] Extracted: {out_final:45s} ({frames_written:4d} frames, {dur_sec:4.1f}s, {size_mb:4.1f} MB)", flush=True)
    return out_final


if __name__ == "__main__":
    print("=" * 75, flush=True)
    print(" Extracting Continuous, UNCUT Real Driving Drives (20s - 45s)", flush=True)
    print("=" * 75, flush=True)

    extracted = []
    for spec in CONTINUOUS_DRIVES:
        p = extract_continuous_clip(spec)
        if p:
            extracted.append(p)

    print(f"\nSuccessfully extracted {len(extracted)} continuous uncut test drives in {OUT_DIR}/!", flush=True)
