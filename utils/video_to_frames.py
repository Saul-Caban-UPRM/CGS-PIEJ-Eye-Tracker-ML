# utils/video_to_frames.py
import cv2
import os

def extract_frames(video_path, output_folder, frame_skip):

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip == 0:
            frame_name = f"{output_folder}/frame_{saved_id:04d}.jpg"
            cv2.imwrite(frame_name, frame)
            saved_id += 1
        frame_id += 1

    cap.release()
    print(f"✅ Extracted {saved_id} frames from {video_path}")

# Example usage: extract 1 frame per second from a 30 fps video
extract_frames("videos/recording.mp4", "yolov8_training/images/train", frame_skip=30)
