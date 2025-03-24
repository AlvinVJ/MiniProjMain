import cv2
import queue
import time
import threading
import subprocess
import os
from collections import deque
from datetime import datetime
from google.cloud import firestore
from ultralytics import YOLO
from dotenv import load_dotenv
import SafeSoloLifeFunctions as sslf

# Load environment variables
load_dotenv()

# Twilio Credentials
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

# Load Firestore database
db = firestore.Client()
username = "user1"
password_typed = "pass1"
doc = db.collection("logins").stream()
user = None
phone = None
for d in doc:
    if d.id == username:
        if d.to_dict()["password"] == password_typed:
            print("Login successful")
            user = d.to_dict()["id"].strip()
            phone = db.collection("adminData").document(user).get().to_dict()["phoneNumber"]
        else:
            print("Login failed")

server_ip = db.collection("config").document("mediamtx_data").get().to_dict()["ip"]
port = db.collection("config").document("mediamtx_data").get().to_dict()["port"]
print(server_ip, port)

# Load YOLO model
model = YOLO("yolo_model.pt")

# Parameters
fall_detection_window = 5  # seconds
frame_rate = 30  # Assuming 30 FPS
frame_count_for_window = fall_detection_window * frame_rate  # Frames in 5 sec
video_buffer_size = frame_rate * 120  # Store last 2 min of frames

# Structure to track all cameras
cameras = {}

# Load cameras from Firestore
def load_cameras():
    global cameras
    cameras = {}  # Reset cameras
    cam_ref = db.collection("adminData").document(user).collection("cameras").document("camera_details").get().to_dict()
    no_cameras = int(len(cam_ref)/3)
    l = []
    for i in range(no_cameras):
        cam_name = cam_ref[f"cam{i+1}Name"]
        cam_idx = cam_ref[f"cam{i+1}Idx"]
        l.append([cam_name, cam_idx])
    for cam in l:
        cameras[cam[0]] = {
            "rtsp_url" : f"rtsp://{server_ip}:{port}/stream_{user}_cam{cam[1]}", 
            "fall_flag": None,
            "fall_queue": deque(maxlen=frame_count_for_window),
            "video_buffer": deque(maxlen=video_buffer_size),
            "frame_queue": queue.Queue(maxsize=10)
        }
    print(f"Loaded {len(cameras)} cameras from Firestore")
    print(cameras)

# Start FFmpeg stream from Python
def start_ffmpeg_stream(camera_name):
    """ Start streaming to RTSP using FFmpeg. """
    rtsp_url = cameras[camera_name]["rtsp_url"]
    camera_index = 0  # Change if using multiple cameras

    ffmpeg_cmd = [
        "ffmpeg", "-f", "dshow", "-rtbufsize", "100M",
        "-i", f'video="{camera_name.strip()}"',  # Change for different cameras
        "-r", "30", "-s", "640x480", "-b:v", "1000k",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-rtsp_transport", "tcp", "-f", "rtsp", rtsp_url
    ]
    print(" ".join(ffmpeg_cmd))
    
    return subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Process each camera stream
def process_camera_stream(camera_name):
    cap = cv2.VideoCapture(0)  # Read from webcam
    if not cap.isOpened():
        print(f"Error: Cannot access {camera_name}")
        return

    video_buffer = cameras[camera_name]["video_buffer"]
    rtsp_url = cameras[camera_name]["rtsp_url"]
    
    # FFmpeg command to take raw frames from OpenCV and send to RTSP
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "640x480", "-framerate", "30", "-i", "-",  # OpenCV as input
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-f", "rtsp", "-rtsp_transport", "tcp", rtsp_url
    ]

    print(f"Starting FFmpeg stream to {rtsp_url}")

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Add frame to buffer
        video_buffer.append(frame.copy())

        # **Send frame to FFmpeg**
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print(f"[ERROR] FFmpeg pipeline broken for {camera_name}")
            break

        # YOLO processing
        results = model(frame, conf=0.5)
        fall_detected = any(
            int(box.cls[0]) == 0 for result in results for box in result.boxes
        )

        # Fall detection tracking
        cameras[camera_name]["fall_queue"].append(1 if fall_detected else 0)
        print(f"[{camera_name}] Fall count: {sum(cameras[camera_name]['fall_queue'])}/{frame_count_for_window}")

        # Manage fall flag
        if fall_detected:
            if cameras[camera_name]["fall_flag"] is None:
                cameras[camera_name]["fall_flag"] = time.time()
        else:
            if len(cameras[camera_name]["fall_queue"]) >= 2 and sum(list(cameras[camera_name]["fall_queue"])[-2:]) == 0:
                cameras[camera_name]["fall_flag"] = None

        # Trigger fall detection action
        if cameras[camera_name]["fall_flag"] is not None and (time.time() - cameras[camera_name]["fall_flag"] >= fall_detection_window):
            handle_fall_detection(camera_name)

    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
# Stop FFmpeg stream

# Fall detection handler
def handle_fall_detection(camera_name):
    print(f"Fall detected in {camera_name}. Initiating response...")

    # Save the buffered video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{camera_name}_{timestamp}.avi"

    video_buffer = cameras[camera_name]["video_buffer"]
    if len(video_buffer) == 0:
        print(f"No frames available to save for {camera_name}")
        return

    height, width, _ = video_buffer[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

    for frame in video_buffer:
        out.write(frame)
    
    out.release()
    print(f"Saved pre-fall video for {camera_name}: {filename}")
    sslf.call(phone, server_ip)


    # Reset fall flag
    cameras[camera_name]["fall_flag"] = None
    cameras[camera_name]["fall_queue"].clear()

# Start processing all cameras
def main():
    load_cameras()
    threads = []
    
    for camera_name in cameras:
        t = threading.Thread(target=process_camera_stream, args=(camera_name,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
