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

# Thread lock to synchronize video buffer access
buffer_lock = threading.Lock()

def capture_frames():
    """ Continuously capture frames and store them in video_buffer """
    while True:
        success, frame = cap.read()
        if not success:
            break
        with buffer_lock:
            video_buffer.append(frame.copy())  # Store frame for 2 min buffer

# Start background thread for video capture
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

def handle_fall_detection():
    """ Function to be called when a continuous fall is detected """
    print("Fall detected. Initiating call...")
    # call = client.calls.create(
    #     url='http://35.225.45.126:5000/voice',
    #     to='+918921357368',
    #     from_='+12568073757'
    # )
    # print(f"Call initiated: {call.sid}")

    # Save the buffered video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_detected_{timestamp}.avi"

    # Lock buffer while writing video
    with buffer_lock:
        if not video_buffer:
            print("Error: No frames in buffer!")
            return
        height, width, _ = video_buffer[0].shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))
        
        for frame in video_buffer:
            out.write(frame)

        out.release()

    print(f"Saved pre-fall video: {filename}")

while True:
    with buffer_lock:
        if len(video_buffer) == 0:
            continue  # Wait until frames are in buffer

        frame = video_buffer[-1]  # Process the latest frame
    
    # Run YOLO model
    results = model(frame, conf=0.5)
    annotated_frame = frame.copy()
    fall_detected = False  

    if results:
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  
                confidence = float(box.conf[0])  
                x1, y1, x2, y2 = map(int, box.xyxy[0])  

                if class_id == 1:  # Standing
                    label = "Standing"
                    color = (0, 255, 0)
                elif class_id == 0:  # Fall detected
                    label = "Fall Detected!"
                    color = (0, 0, 255)
                    fall_detected = True

                # Draw annotations
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Track falls
    fall_queue.append(1 if fall_detected else 0)
    print(f"Fall count: {sum(fall_queue)}/{frame_count_for_window}")

    # Fall flag management
    if fall_detected:
        if fall_flag is None:
            fall_flag = time.time()  # Start flag when fall starts
    else:
        if len(fall_queue) >= 2 and sum(islice(fall_queue, len(fall_queue) - 2, None)) == 0:
            fall_flag = None

    # Decision to call function
    if fall_flag is not None and (time.time() - fall_flag >= fall_detection_window):
        handle_fall_detection()
        fall_flag = None  # Reset flag
        fall_queue.clear()  # Clear queue after action

    # Display output
    cv2.imshow("YOLO Fall Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
