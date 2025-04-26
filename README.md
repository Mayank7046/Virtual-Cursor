# Virtual-Cursor
This project implements a virtual mouse/cursor controlled by hand gestures using computer vision techniques. It uses OpenCV, MediaPipe, and PyAutoGUI to detect and track hand movements, enabling the user to control the mouse pointer, perform clicks, and interact with their computer — touchless!
# Real-time Object Detection

This project implements real-time object detection using YOLOv3 and OpenCV. It can detect 80 different types of objects in real-time using your webcam.

## Setup

1. Install the required packages:
```
pip install -r requirements.txt
```

2. Run the program:
```
python object_detection.py
```

## Usage

- The program will automatically download the required YOLO model files on first run
- Once started, it will open your webcam and begin detecting objects in real-time
- Press 'q' to quit the program

## Features

- Real-time object detection using YOLOv3
- Support for 80 different object classes
- Displays object labels and confidence scores
- Handles multiple object detection in the same frame
