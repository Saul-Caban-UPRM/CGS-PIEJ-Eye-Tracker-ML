# utils/motion_analysis.py
# - Purpose: simple frame-difference motion analyzer that computes a scaled motion score and
#   classifies overall behavior as 'delicate', 'normal', or 'withoutcare'.
# - Function: detect_motion(video_path)
#   - video_path: path to input video file.
#   - returns: one of 'delicate', 'normal', 'withoutcare'.
# - Notes:
#   - Uses OpenCV and numpy (pip install opencv-python numpy).
#   - Heuristic: sums pixel-wise frame differences above a threshold and scales the result.
#   - For robustness, consider adding handling for unreadable/empty videos and normalizing by frame count.
# - Suggested improvements:
#   - Return numeric motion score in addition to label, tune threshold/scale, or use optical flow for finer analysis.
import cv2, numpy as np

def detect_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_score = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        motion_score += np.sum(diff > 30)
        prev_gray = gray

    cap.release()
    avg_motion = motion_score / 10000000  # scale factor
    if avg_motion < 0.5:
        return "delicate"
    elif avg_motion < 1.0:
        return "normal"
    else:
        return "withoutcare"
