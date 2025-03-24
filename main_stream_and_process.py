from dotenv import load_dotenv
import os
import subprocess
import SafeSoloLifeFunctions as sslf
from google.cloud import firestore
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
import cv2
import threading
from collections import deque
from datetime import datetime
import time
import queue


# Load environment variables
load_dotenv()
db = firestore.Client()

uid = os.environ.get('UID')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
phone = None

# Fetch camera & RTSP server details from Firestore
data = db.collection("adminData").document(uid).get().to_dict()
server_info = db.collection("config").document("mediamtx_data").get().to_dict()
server_ip, mediamtx_port = server_info["ip"], server_info["port"]

if data:
    phone = data["phoneNumber"]

model = YOLO("yolo_model.pt")

# Global variables for camera selection
selected_camera_idx = None
cam_name = None

# Function to handle camera selection
def on_camera_select(i):
    global selected_camera_idx, cam_name
    selected_camera_idx = int(data['camera_index'][i])
    location = data["locations"][i]
    cam_name = data['cameras'][i]
    root.destroy()
# Create camera selection GUI
root = tk.Tk()
root.title("Camera Selection")
root.geometry("300x200")

for i, cam in enumerate(data['cameras']):
    btn = tk.Button(root, text=f"{cam} ({data['locations'][i]})", 
                    command=lambda idx=i: on_camera_select(idx))
    btn.pack(pady=10)

root.mainloop()

if selected_camera_idx is None:
    messagebox.showerror("Error", "No camera selected")
    exit()

# Construct RTSP URL
rtsp_url = f"rtsp://{server_ip}:{mediamtx_port}/stream_{uid}_cam{selected_camera_idx}"

# Initialize fall detection parameters
fall_detection_window = 3  # seconds
frame_rate = 30  # Assuming 30 FPS
frame_count_for_window = fall_detection_window * frame_rate  # Frames in 5 sec
fall_flag = None  # Timestamp when fall starts
fall_queue = deque(maxlen=frame_count_for_window)
video_buffer_size = frame_rate * 120  # Store last 2 min of frames
video_buffer = deque(maxlen=video_buffer_size)
last_call_time = None

cap = cv2.VideoCapture(selected_camera_idx)
if not cap.isOpened():
    print(f"[ERROR] Cannot access camera {selected_camera_idx}")
    exit()

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Lock for buffer access
buffer_lock = threading.Lock()

def start_rtsp_stream(rtsp_url):
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
            frame = video_buffer.popleft()
        except IndexError:
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
            video_buffer.append(frame.copy())  # Use `put_nowait()` instead of `.full()`
        except queue.Full:
            pass  # If the queue is full, discard the frame

        time.sleep(1 / frame_rate)  # Maintain frame rate


def handle_fall_detection():
    global last_call_time, location
    """ Function to be called when a continuous fall is detected """
    global fall_flag  # Ensure we modify the global variable

    # Save video before fall
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"fall_detected_{timestamp}.avi"
    if last_call_time is not None and time.time() - last_call_time < 120:
        print("[INFO] Cooldown period active. Skipping fall detection...")
        return
    with buffer_lock:
        if len(video_buffer) < (video_buffer.maxlen*0.6):
            print("[ERROR] No frames available in buffer")
            return

        # 🔹 Clear buffers to remove old frames
        video_buffer.clear()
        fall_queue.clear()

        # 🔹 Reset fall flag with a cooldown
        fall_flag = None  
        print("[ALERT] Fall detected. Saving video...")
        sslf.call(phone, server_ip, location)
        print("[COOLDOWN] Pausing fall detection for 10 seconds...")
        last_call_time = time.time()  # Cooldown period to prevent re-triggering




# Start RTSP Streaming Thread
rtsp_thread = threading.Thread(target=start_rtsp_stream, daemon=True, args=(rtsp_url,))
rtsp_thread.start()

# Start YOLO Processing Thread
yolo_thread = threading.Thread(target=process_yolo, daemon=True)
yolo_thread.start()

# Start Frame Capture
capture_frames()