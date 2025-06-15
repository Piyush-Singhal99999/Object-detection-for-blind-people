# Object-detection-for-blind-people
Real-Time Object Detection and Voice Feedback System for Enhanced Mobility of Blind and Visually Impaired Individuals Using YOLOv8

This Python script uses YOLO for object tracking in real-time video streams. It detects objects in each frame and tracks them across frames, while also using text-to-speech to announce the presence of detected objects.

## Dependencies
- [cv2](https://pypi.org/project/opencv-python/): OpenCV library for computer vision
- [numpy](https://pypi.org/project/numpy/): Numerical operations library
- [ultralytics](https://github.com/ultralytics/yolov5): YOLO object detection library
- [pyttsx3](https://pypi.org/project/pyttsx3/): Text-to-speech library

## Installation
pip install opencv-python numpy ultralytics pyttsx3

## Usage
Ensure all dependencies are installed.
Run the script:
python r.py

The script will open the default camera or a specified video stream (adjust the cap = cv2.VideoCapture(...) line accordingly).

Detected objects will be annotated in the video feed, and their names will be announced using text-to-speech.


## Configuration
Modify the cap = cv2.VideoCapture(...) line to use a different video source.
Adjust parameters like frame width (w), frame height (h), and frames per second (fps) as needed.
The YOLO model file is set to "yolov8n.pt." Ensure you have the correct model file or update the path accordingly.
