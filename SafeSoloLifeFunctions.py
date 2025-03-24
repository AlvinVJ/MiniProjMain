# functions.py
import subprocess
from twilio.rest import Client
import os

account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

def call(phone, server_ip, location):
    client = Client(account_sid, auth_token)

    call = client.calls.create(
                            url=f'http://{server_ip}:5000/voice/{location}',
                            to=f'+91{phone}',
                            from_='+12568073757'
                        )

    print(call.sid)

    # To get the call status
    call = client.calls(call.sid).fetch()
    print(call.status)

def start_camera_stream(camera_name, rtsp_url):
    """
    Captures video from a camera and streams it to an RTSP server using the provided ffmpeg command.
    
    Args:
    - camera_name (str): The name of the camera (e.g., "HP Wide Vision HD Camera").
    - rtsp_url (str): The RTSP URL to stream the video to.
    """
    # Define the ffmpeg command with your parameters
    command = [
        'ffmpeg',
        '-f', 'dshow',  # Windows-specific option for DirectShow devices (cameras)
        '-rtbufsize', '100M',  # Buffer size for DirectShow
        '-i', f'video={camera_name}',  # Camera input (use the exact camera name here)
        '-r', '30',  # Frame rate (30fps)
        '-s', '640x480',  # Resolution (adjust as needed)
        '-b:v', '1000k',  # Video bitrate (1 Mbps)
        '-c:v', 'libx264',  # Video codec (H.264)
        '-preset', 'ultrafast',  # Encoding speed (fastest)
        '-tune', 'zerolatency',  # Tune for low latency
        '-rtsp_transport', 'tcp',  # Use TCP for RTSP transport
        '-f', 'rtsp',  # RTSP output format
        rtsp_url  # RTSP server URL (replace with your own)
    ]

    # Create a subprocess to run the ffmpeg command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Monitor the subprocess output (optional, for debugging purposes)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error streaming from camera {camera_name}: {stderr.decode()}")
    else:
        print(f"Streaming from camera {camera_name} to {rtsp_url}")

    # Log output for debugging purposes
    print(f"FFmpeg stdout: {stdout.decode()}")
    print(f"FFmpeg stderr: {stderr.decode()}")




