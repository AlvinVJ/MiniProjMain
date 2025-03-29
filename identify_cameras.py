import cv2

def list_available_cameras():
    available_cameras = []
    for i in range(10):  # Check for cameras from index 0 to 9
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better Windows compatibility
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def open_camera_streams(camera_indices):
    caps = {i: cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in camera_indices}
    
    while True:
        for i, cap in caps.items():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera {i}', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = list_available_cameras()
    if cameras:
        print("Available cameras:", cameras)
        open_camera_streams(cameras)
    else:
        print("No cameras detected.")