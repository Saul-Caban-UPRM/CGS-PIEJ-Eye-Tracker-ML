from ultralytics import YOLO
from utils.motion_analysis import detect_motion
from pathlib import Path
import sys, os
import cv2

"""
main.py

Purpose
-------
Simple video analyzer that:
- Runs YOLO inference on every Nth frame of a video.
- Collects detections that match a list of target class names.
- Runs a separate motion analysis (detect_motion) and reports the detected behaviour.

Usage
-----
- By default the script calls analyze("videos/video1.mp4") at the bottom.
- Configure MODEL_PATH, TARGET_NAMES, FRAME_STEP, or uncomment and use the argparse section
  below to provide command line arguments (see "Optional changes" below).

Notes / Extensions
------------------
- See the "Optional changes" section at the end of this file for recommended improvements:
  argparse CLI, saving JSON results, annotating/saving output video, logging, GPU device choice, etc.
"""

# Path to model weights (local override). If not found, falls back to the official small model.
MODEL_PATH = Path("models/best.pt") # This should point to your custom trained model weights (currently not trained))
if not MODEL_PATH.exists():
    print(f"Model not found at: {MODEL_PATH.resolve()}")
    print("Place your weights at that path, update MODEL_PATH, or allow fallback to the official pretrained model (will be downloaded).")
    MODEL_PATH = "yolov8n.pt"  # fallback: official pretrained YOLOv8 small model

# instantiate model
model = YOLO(str(MODEL_PATH))

# target class names you care about (update to match model class names)
TARGET_NAMES = [
    "calculator",
    "greenfolder",
    "pen",
    "ruler",
    "sunglasses",
    "toy",
    "whitebook",
]

def analyze(video_path, frame_step: int = 30, conf_threshold: float = 0.5):
    """
    Analyze a video and report which target items were detected and the motion behavior.

    Parameters
    ----------
    video_path : str or Path
        Path to the video file to analyze.
    frame_step : int
        Process every `frame_step` frames (default 30).
    conf_threshold : float
        Confidence threshold for detection (default 0.5).

    Behaviour
    ---------
    - Opens the video with OpenCV.
    - Calls detect_motion(video_path) for motion behavior analysis.
    - For every `frame_step` frame, runs YOLO inference on the BGR numpy frame.
    - Collects any classes that match TARGET_NAMES and prints a summary at the end.

    Returns
    -------
    None (prints summary). You can change it to return a dict for programmatic use.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    # run auxiliary motion analysis (your implementation)
    behavior = detect_motion(video_path)

    detected_target_names = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            # run inference on this frame (one numpy BGR image)
            results = model(frame, conf=conf_threshold)  # ultralytics model call on a frame
            r = results[0] if len(results) > 0 else None

            if r is not None and hasattr(r, "boxes") and len(r.boxes) > 0:
                # convert class indices to ints and then map to names
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


# Default invocation (change path as needed)
if __name__ == "__main__":
    analyze("videos/video1.mp4")  # change the path to the video you want to analyze


# Optional changes / extensions (commented examples)
#
# 1) Command-line interface (argparse)
# -----------------------------------
# Uncomment and adapt this block to allow running from CLI with options.
#
# import argparse
# def main_cli():
#     parser = argparse.ArgumentParser(description="Analyze video with YOLO and motion analysis")
#     parser.add_argument("video", help="Path to video file")
#     parser.add_argument("--model", default=str(MODEL_PATH), help="Path to weights or model name")
#     parser.add_argument("--step", type=int, default=30, help="Frame step for inference")
#     parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
#     args = parser.parse_args()
#     global model
#     model = YOLO(args.model)
#     analyze(args.video, frame_step=args.step, conf_threshold=args.conf)
#
# if __name__ == "__main__":
#     main_cli()
#
# 2) Save results to JSON
# -----------------------
# - Collect results into a dict and write to a .json file for later analysis.
#
# import json
# def analyze_and_save(video_path, out_json="results.json", **kwargs):
#     # collect detections and behavior in a dict then json.dump(...)
#     pass
#
# 3) Annotate frames and write an output video
# --------------------------------------------
# - Use r.boxes.xyxy, r.boxes.cls, r.boxes.conf to draw boxes and labels.
# - Use cv2.VideoWriter to write the annotated frames.
