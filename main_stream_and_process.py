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
location = "home"

data = db.collection("adminData").document(uid).get().to_dict()
server_info = db.collection("config").document("mediamtx_data").get().to_dict()
server_ip, mediamtx_port = server_info["ip"], server_info["port"]

if data:
    phone = data["phoneNumber"]

model = YOLO("yolo_model.pt")

selected_camera_idx = None
cam_name = None

def on_camera_select(i):
    global selected_camera_idx, cam_name, location
    selected_camera_idx = int(data['camera_index'][i])
    location = data["locations"][i]
    cam_name = data['cameras'][i]
    root.destroy()

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

rtsp_url = f"rtsp://{server_ip}:{mediamtx_port}/stream_{uid}_cam{selected_camera_idx}"

fall_detection_window = 3  
frame_rate = 30  
frame_count_for_window = fall_detection_window * frame_rate  
fall_flag = None  
fall_queue = deque(maxlen=frame_count_for_window)
video_buffer_size = frame_rate * 120  
video_buffer = deque(maxlen=video_buffer_size)
last_call_time = None

cap = cv2.VideoCapture(selected_camera_idx)
if not cap.isOpened():
    print(f"[ERROR] Cannot access camera {selected_camera_idx}")
    exit()

cap.set(3, 640)  
cap.set(4, 480)  

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

        fall_queue.append(1 if fall_detected else 0)
        print(f"[YOLO] Fall count: {sum(fall_queue)}/{frame_count_for_window}")

        if fall_detected:
            if fall_flag is None:
                fall_flag = time.time()
        else:
            if len(fall_queue) >= 2 and sum(list(fall_queue)[-2:]) == 0:
                fall_flag = None

        if fall_flag is not None and (time.time() - fall_flag >= fall_detection_window):
            handle_fall_detection()


def capture_frames():
    """ Capture frames and send to both RTSP stream and YOLO processing """
    while True:
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame from camera")
            break

        with buffer_lock:
            video_buffer.append(frame.copy())

        try:
            video_buffer.append(frame.copy())  
        except queue.Full:
            pass  
        time.sleep(1 / frame_rate)  

def handle_fall_detection():
    global last_call_time, location
    """ Function to be called when a continuous fall is detected """
    global fall_flag

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if last_call_time is not None and time.time() - last_call_time < 120:
        print("[INFO] Cooldown period active. Skipping fall detection...")
        return
    with buffer_lock:
        if len(video_buffer) < (video_buffer.maxlen*0.6):
            print("[ERROR] No frames available in buffer")
            return

        video_buffer.clear()
        fall_queue.clear()

        fall_flag = None  
        print("[ALERT] Fall detected. Saving video...")
        sslf.call(phone, server_ip, location)
        print("[COOLDOWN] Pausing fall detection for 10 seconds...")
        last_call_time = time.time()  




rtsp_thread = threading.Thread(target=start_rtsp_stream, daemon=True, args=(rtsp_url,))
rtsp_thread.start()

yolo_thread = threading.Thread(target=process_yolo, daemon=True)
yolo_thread.start()

capture_frames()