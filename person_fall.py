import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def main():
    # Load the YOLOv12 model (use the path to your downloaded file)
    model = YOLO("yolo12n.pt")  # Replace with the correct path to your model file

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Initialize confusion matrix variables
    y_true = []  # Ground truth labels (1 for person, 0 for no person)
    y_pred = []  # Model predictions

    print("Person Detection Instructions:")
    print("- Move in and out of camera view")
    print("- Press 'y' when YOU think a person is present")
    print("- Press 'n' when YOU think no person is present")
    print("- Press 'q' to quit and show results")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv12 inference
        results = model(frame)

        # Check if a person is detected
        person_detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get class ID
                conf = box.conf[0].item()  # Confidence score
                if class_id == 0 and conf > 0.5:  # If it's a person and confidence > 50%
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display detection result
        text = "Person Detected" if person_detected else "No Person Detected"
        color = (0, 255, 0) if person_detected else (0, 0, 255)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Person Detection", frame)

        # Wait for key press with a timeout
        key = cv2.waitKey(100)  # Reduced wait time to 100ms

        if key == ord('y'):
            print("You said: Person Present")
            y_true.append(1)  # Ground truth: Person is there
            y_pred.append(1 if person_detected else 0)  # Model's prediction

        elif key == ord('n'):
            print("You said: No Person")
            y_true.append(0)  # Ground truth: No person
            y_pred.append(1 if person_detected else 0)  # Model's prediction

        elif key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Compute Accuracy and Confusion Matrix
    if len(y_true) > 0 and len(y_pred) > 0:
        # Compute metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        print("\n--- Detection Performance ---")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nConfusion Matrix Breakdown:")
        print("True Negatives (TN):", conf_matrix[0][0])
        print("False Positives (FP):", conf_matrix[0][1])
        print("False Negatives (FN):", conf_matrix[1][0])
        print("True Positives (TP):", conf_matrix[1][1])
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Person', 'Person']))
        
        print(f"\nOverall Accuracy: {accuracy:.2%}")
    else:
        print("\nNo data collected. Please provide inputs during the detection process.")

if __name__ == "__main__":
    main()