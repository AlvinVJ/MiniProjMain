import cv2
import subprocess
import threading
import time
import queue
import os
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RTSP Server Details
server_ip = os.getenv("SERVER_IP", "your_server_ip")
mediamtx_port = os.getenv("SERVER_PORT", "8554")
rtsp_url = "rtsp://34.133.182.109:8554/live"

# Load YOLO Model
model = YOLO("yolo_model.pt")

# Video Parameters
frame_rate = 30  # FPS
fall_detection_window = 5  # Seconds
frame_count_for_window = fall_detection_window * frame_rate
video_buffer_size = frame_rate * 120  # Store last 2 min of frames

# Camera Index
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"[ERROR] Cannot access camera {camera_index}")
    exit()

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Fall Detection Variables
fall_flag = None
fall_queue = deque(maxlen=frame_count_for_window)
video_buffer = deque(maxlen=video_buffer_size)

# Frame Queue for YOLO
frame_queue = queue.Queue(maxsize=10)

# Lock for synchronizing video buffer access
buffer_lock = threading.Lock()


def start_rtsp_stream():
    """ Start streaming to RTSP using FFmpeg """
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite existing files
        "-f", "rawvideo",  # Raw video format
        "-pix_fmt", "bgr24",  # OpenCV uses BGR format
        "-s", "640x480",  # Frame size
        "-r", "30",  # Frame rate
        "-i", "-",  # Read input from stdin (pipe)
        "-c:v", "libx264",  # Use H.264 codec
        "-preset", "ultrafast",  # Lower latency
        "-tune", "zerolatency",  # Optimize for streaming
        "-rtsp_transport", "tcp",
        "-f", "rtsp",  # Output format
        rtsp_url  # RTSP server URL
    ]

    print(f"[INFO] Starting RTSP stream to {rtsp_url}")
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from camera")
            break

        try:
            # Write frame to FFmpeg
            process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("[ERROR] FFmpeg process closed")
            break

    process.stdin.close()
    process.wait()


def process_yolo():
    """ Continuously process frames with YOLO for fall detection """
    global fall_flag

    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
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
        try:
            frame_queue.put_nowait(frame.copy())  # Use `put_nowait()` instead of `.full()`
        except queue.Full:
            pass  # If the queue is full, discard the frame

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

        video_buffer.clear()
        fall_queue.clear()

        out.release()

    print(f"[SAVED] Pre-fall video: {filename}")


# Start RTSP Streaming Thread
rtsp_thread = threading.Thread(target=start_rtsp_stream, daemon=True)
rtsp_thread.start()

# Start YOLO Processing Thread
yolo_thread = threading.Thread(target=process_yolo, daemon=True)
yolo_thread.start()

# Start Frame Capture
capture_frames()
