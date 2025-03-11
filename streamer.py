# main.py
import threading
import SafeSoloLifeFunctions as sslf

def main():

    # Define the RTSP server URL
    rtsp_server_url = "rtsp://35.225.45.126:8554/"  # Replace with your RTSP server URL

    # List of camera names you want to stream from (replace with actual camera names on your system)
    camera_names = [
        "HP Wide Vision HD Camera", "DroidCam Video"  # Replace with your actual camera name    ]
    ]

    channels = [
        "mystream1", "mystream2"
    ]
    # Create and start threads for each camera stream
    threads = []
    
    for i in range(len(camera_names)):
        camera_name = camera_names[i]
        thread = threading.Thread(target=sslf.start_camera_stream, args=(camera_name, rtsp_server_url+channels[i]))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish (streaming will continue running in the background)
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
