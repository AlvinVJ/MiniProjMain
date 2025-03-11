from ultralytics import YOLO
import cv2
import os

# Avoid OpenMP conflict errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the trained YOLO model
model = YOLO("pleaseWork.pt")  # Ensure this is your trained model

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Set webcam resolution (Optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    # Read frame from webcam
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO on the frame
    results = model(frame, conf=0.5)

    # Annotate the frame with bounding boxes and text
    annotated_frame = frame.copy()  # Make a copy to draw on

    if results:
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = float(box.conf[0])  # Confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

                # **Fixed Class Label Mapping**
                if class_id == 0:  # Originally Standing
                    label = "Falling"  # Ensure it shows "Standing"
                    color = (0, 0, 255)  # Green for standing
                elif class_id == 1:  # Originally Fall
                    label = "Standing!"  # Ensure it shows "Fall Detected!"
                    color = (0, 255, 0)  # Red for fall

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"{label} {confidence:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display the frame
    cv2.imshow("YOLO Fall Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
