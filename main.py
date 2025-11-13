from ultralytics import YOLO
from utils.motion_analysis import detect_motion

model = YOLO("models/best.pt")

def analyze(video_path):
    results = model.predict(video_path, conf=0.5, save=True)
    item_detected = results[0].names[int(results[0].boxes.cls[0])]
    behavior = detect_motion(video_path)
    print(f"Item: {item_detected} | Behavior: {behavior}")

analyze("videos/test_clip.mp4") # changes the path to the video you want to analyze
