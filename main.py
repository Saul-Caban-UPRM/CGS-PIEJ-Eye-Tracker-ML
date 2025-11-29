from ultralytics import YOLO
from utils.motion_analysis import detect_motion
from pathlib import Path
import sys, os
import cv2

# check model path and provide a fallback
MODEL_PATH = Path("models/best.pt")
if not MODEL_PATH.exists():
    print(f"Model not found at: {MODEL_PATH.resolve()}")
    print("Place your weights at that path, update MODEL_PATH, or allow fallback to the official pretrained model (will be downloaded).")
    MODEL_PATH = "yolov8n.pt"  # fallback: official pretrained YOLOv8 small model

model = YOLO(str(MODEL_PATH))

# target class names you care about
TARGET_NAMES = [
    "calculator",
    "greenfolder",
    "pen",
    "ruler",
    "sunglasses",
    "toy",
    "whitebook",
]

def analyze(video_path, frame_step: int = 30):
    """
    Run detection every `frame_step` frames instead of every frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    behavior = detect_motion(video_path)
    detected_target_names = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            # run inference on this frame (one numpy BGR image)
            results = model(frame, conf=0.5)  # ultralytics model call on a frame
            r = results[0] if len(results) > 0 else None

            if r is not None and hasattr(r, "boxes") and len(r.boxes) > 0:
                cls_indices = [int(x) for x in r.boxes.cls]
                name_map = r.names if hasattr(r, "names") else getattr(model, "names", {})
                detected_names = [name_map.get(i, str(i)) for i in cls_indices]
                for n in detected_names:
                    if n in TARGET_NAMES:
                        detected_target_names.add(n)

        frame_idx += 1

    cap.release()

    if detected_target_names:
        print(f"Found target items: {', '.join(sorted(detected_target_names))} | Behavior: {behavior}")
    else:
        print(f"No target items found | Behavior: {behavior}")

analyze("videos/video1.mp4") # changes the path to the video you want to analyze
