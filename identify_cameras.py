import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import os
from collections import deque
from twilio.rest import Client
from datetime import datetime
from itertools import islice
from dotenv import load_dotenv
import subprocess
import queue

# Load environment variables
load_dotenv()

# Twilio Credentials
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
client = Client(account_sid, auth_token)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load YOLO model
model = YOLO("yolo_model.pt")

# Open webcam
cap = cv2.VideoCapture(0)  

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Fall detection parameters
fall_detection_window = 7  # seconds
frame_rate = 30  # Manually set frame rate (ensure consistency)
frame_count_for_window = fall_detection_window * frame_rate  # Frames in 7 seconds

fall_flag = None  # Timestamp when fall starts
fall_queue = deque(maxlen=frame_count_for_window)  # Track falls
video_buffer = deque(maxlen=frame_rate * 120)  # Stores last 2 min of frames
frame_queue = deque(maxlen=frame_rate * 5)  # Queue for YOLO processing

# Thread lock to synchronize video buffer access
buffer_lock = threading.Lock()
cam_name = "HP Wide Vision HD Camera"
rtsp_url = "rtsp://34.133.182.109:8554/live"

def start_camera_stream(camera_name, rtsp_url):
    """
    Captures video from a camera and streams it to an RTSP server using the provided ffmpeg command.
    
    Args:
    - camera_name (str): The name of the camera (e.g., "HP Wide Vision HD Camera").
    - rtsp_url (str): The RTSP URL to stream the video to.
    """
    # Define the ffmpeg command with your parameters
    command = [
        'ffmpeg',
        '-f', 'dshow',  # Windows-specific option for DirectShow devices (cameras)
        '-rtbufsize', '100M',  # Buffer size for DirectShow
        '-i', f'video={camera_name}',  # Camera input (use the exact camera name here)
        '-r', '30',  # Frame rate (30fps)
        '-s', '640x480',  # Resolution (adjust as needed)
        '-b:v', '1000k',  # Video bitrate (1 Mbps)
        '-c:v', 'libx264',  # Video codec (H.264)
        '-preset', 'ultrafast',  # Encoding speed (fastest)
        '-tune', 'zerolatency',  # Tune for low latency
        '-rtsp_transport', 'tcp',  # Use TCP for RTSP transport
        '-f', 'rtsp',  # RTSP output format
        rtsp_url  # RTSP server URL (replace with your own)
    ]

    # Create a subprocess to run the ffmpeg command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Monitor the subprocess output (optional, for debugging purposes)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error streaming from camera {camera_name}: {stderr.decode()}")
    else:
        print(f"Streaming from camera {camera_name} to {rtsp_url}")

    # Log output for debugging purposes
    print(f"FFmpeg stdout: {stdout.decode()}")
    print(f"FFmpeg stderr: {stderr.decode()}")

def process_yolo():
    """ Continuously process frames with YOLO for fall detection """
    global fall_flag

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except Exception as e:
            continue

        results = model(frame, conf=0.5)
        fall_detected = any(
            int(box.cls[0]) == 0 for result in results for box in result.boxes
        )

        # Fall tracking
        fall_queue.append(1 if fall_detected else 0)
        print(f"[YOLO] Fall count: {sum(fall_queue)}/{frame_count_for_window}")

        # Fall flag management
        if fall_detected:
            if fall_flag is None:
                fall_flag = time.time()
        else:
            if len(fall_queue) >= 2 and sum(list(fall_queue)[-2:]) == 0:
                fall_flag = None

        # Handle fall detection event
        if fall_flag is not None and (time.time() - fall_flag >= fall_detection_window):
            handle_fall_detection()

def capture_frames():
    """ Capture frames and send to both RTSP stream and YOLO processing """
    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from camera")
            break

        # Store frame in video buffer
        with buffer_lock:
            video_buffer.append(frame.copy())

        # Send frame to YOLO queue
        if len(frame_queue) < frame_queue.maxlen:
            frame_queue.append(frame.copy())

        time.sleep(1 / frame_rate)  # Maintain frame rate


def handle_fall_detection():
    """ Function to be called when a continuous fall is detected """
    print("[ALERT] Fall detected. Saving video...")

    # Save video before fall
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_detected_{timestamp}.avi"

    with buffer_lock:
        if not video_buffer:
            print("[ERROR] No frames available in buffer")
            return

        height, width, _ = video_buffer[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

        for frame in video_buffer:
            out.write(frame)

        out.release()

    print(f"[SAVED] Pre-fall video: {filename}")

# Start RTSP Streaming
start_camera_stream(cam_name, rtsp_url)

# Start YOLO Processing Thread
yolo_thread = threading.Thread(target=process_yolo, daemon=True)
yolo_thread.start()

# Start Frame Capture
capture_frames()