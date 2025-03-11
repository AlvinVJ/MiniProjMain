import cv2
import numpy as np
import time
from ultralytics import YOLO
import os
from collections import deque
from twilio.rest import Client


account_sid = OS.environ.get('TWILIO_ACCOUNT_SID')
auth_token = OS.environ.get('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model = YOLO("yolo_model.pt")  # Load the YOLO model

cap = cv2.VideoCapture(0)  # Capture from webcam

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Function to be called when a fall is detected
def handle_fall_detection():
    call = client.calls.create(
                        url='http://35.225.45.126:5000/voice',
                        to='+918921357368',
                        from_='+12568073757'
                    )

    print(call.sid)

# Variables to manage fall detection over 7 seconds
fall_detection_window = 3  # 7 seconds window for detecting fall
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))  # Get frame rate of the webcam
frame_count_for_7_seconds = fall_detection_window * frame_rate  # Number of frames in 7 seconds
print(frame_count_for_7_seconds)

# Initialize a deque (queue) to store fall detection results (1 for fall, 0 for no fall)
fall_queue = deque(maxlen=frame_count_for_7_seconds)

while True:
    # Read frame from webcam
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO on the frame
    results = model(frame, conf=0.5)

    # Annotate the frame with bounding boxes and text
    annotated_frame = frame.copy()  # Make a copy to draw on

    # Flag to check if a fall is detected
    fall_detected_in_frame = False

    if results:
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = float(box.conf[0])  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

                # **Fixed Class Label Mapping**
                if class_id == 1:  # Assuming '0' is the standing class
                    label = "Standing"
                    color = (0, 255, 0)  # Green for standing
                elif class_id == 0:  # Assuming '1' is the fall class
                    label = "Fall Detected!"
                    color = (0, 0, 255)  # Red for fall
                    fall_detected_in_frame = True  # Mark fall detection

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Append the fall detection result (1 for fall, 0 for no fall) to the queue
    fall_queue.append(1 if fall_detected_in_frame else 0)
    print(str(sum(fall_queue)) + '/' + str(frame_count_for_7_seconds))

    # Once the queue reaches the required number of frames, check for the number of falls
    if len(fall_queue) == frame_count_for_7_seconds:
        # Count the number of falls (1s) in the queue
        fall_count = sum(fall_queue)
        
        # If the number of falls is more than half the window size, call the function
        if fall_count > frame_count_for_7_seconds // 2:
            handle_fall_detection()

        # Clear the queue after calling the function
        fall_queue.clear()

    # Display the frame with annotations
    cv2.imshow("YOLO Fall Detection", annotated_frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
