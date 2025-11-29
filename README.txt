# Project: VideoEdition-YOLO — quick setup & run instructions

Purpose
-------
Small project to:
- Run YOLO inference on video frames (main.py).
- Extract frames from videos (utils/video_to_frames.py).
- Do simple motion analysis (utils/motion_analysis.py).
- Provide a dataset config for training (dataset.yaml).

Prerequisites (Windows)
-----------------------
- Python 3.12 installed.
- Git
- If you want GPU support, install a CUDA-compatible PyTorch first:
  - See https://pytorch.org for the correct pip command for your CUDA version.

Recommended quick setup (PowerShell / CMD)
------------------------------------------
# upgrade pip
pip install --upgrade pip

Install Python dependencies
---------------------------
# minimal required packages
pip install ultralytics opencv-python-headless numpy

# If you prefer the GUI OpenCV (imshow etc):
pip install opencv-python

Notes:
- If using GPU, install torch first with the right CUDA build (see pytorch.org), then install ultralytics.
- ultralytics may install the "yolo" CLI command; ensure your PATH/venv is active.

Files of interest
-----------------
- main.py
  - Runs YOLO on every Nth frame and calls utils/motion_analysis.detect_motion.
  - Edit MODEL_PATH at top to point to models/best.pt (or allow fallback to yolov8n.pt).
  - Default invocation: analyze("videos/video1.mp4")

- utils/video_to_frames.py
  - extract_frames(video_path, output_folder, frame_skip)
  - Note: current file contains an example call at the bottom that executes on import.
    Comment out that call if you want to import the function instead of running it.

- utils/motion_analysis.py
  - detect_motion(video_path) -> returns one of: delicate, normal, withoutcare
  - Simple frame-difference heuristic; tune thresholds/scale as needed.

- dataset.yaml
  - Dataset config for YOLO training. Ensure paths and classes match your data.

How to run
----------
1) Run the analyzer on a video:
   python main.py

   - Edit the call at bottom of main.py or enable the argparse block for CLI options.
   - Ensure videos/video1.mp4 exists or change path.

2) Extract frames:
   # Edit or call function from a small wrapper script:
   python -c "from utils.video_to_frames import extract_frames; extract_frames('videos/video1.mp4','ScreenShots/video1',30)"

3) Train (example using ultralytics CLI)
   # from project root (with venv active and correct data paths in dataset.yaml)
   yolo task=detect mode=train data=dataset.yaml model=yolov8n.pt epochs=100

Tips and troubleshooting
------------------------
- Model weights: place custom weights at models/best.pt or edit MODEL_PATH in main.py.
- If running main.py raises ModuleNotFoundError for ultralytics, ensure venv is active and packages installed in that environment.
- If video cannot be opened: confirm the path, and try absolute path or convert video codec with ffmpeg.
- For large videos: increase frame_step in analyze(...) to reduce inference load.
- To prevent scripts from auto-running on import, remove or comment sample calls at bottom of those scripts.

Extending the project
---------------------
- Save detection results to JSON.
- Annotate frames and save output video using cv2.VideoWriter.
- Replace simple motion heuristic with optical flow for better accuracy.