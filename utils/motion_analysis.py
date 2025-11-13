# utils/motion_analysis.py
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
