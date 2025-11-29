# utils/video_to_frames.py
# - Purpose: extract frames from a video file and save them as image files at a specified interval.
# - Function: extract_frames(video_path, output_folder, frame_skip)
#   - video_path: path to the input video file.
#   - output_folder: directory where extracted frames will be saved (created if missing).
#   - frame_skip: save every Nth frame (integer). e.g., frame_skip=30 saves one frame every 30 frames.
# - Notes:
#   - Uses OpenCV (pip install opencv-python).
#   - The example call at the bottom runs on import; comment it out if you import this as a module.
#   - Consider adjusting filename pattern or format if needed.
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
extract_frames("videos/video1.mp4", "ScreenShots/video1", frame_skip=5)
