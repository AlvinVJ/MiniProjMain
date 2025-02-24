import subprocess
import time

rtsp_url = "rtsp://35.232.192.197:8554/mystream"
ffmpeg_cmd = [
    r"C:\Users\hp\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-7.1-full_build\bin\ffmpeg.exe", "-f", "dshow", "-rtbufsize", "100M", "-i", "video=HP Wide Vision HD Camera",
    "-r", "10", "-s", "640x360", "-b:v", "500k",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-rtsp_transport", "tcp", "-f", "rtsp", rtsp_url
]

def start_stream():
    while True:
        print("🚀 Starting FFmpeg...")
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        
        proc.wait()  # Wait for FFmpeg to exit
        
        print("⚠️ FFmpeg disconnected! Restarting in 2 seconds...")
        time.sleep(2)  # Wait before restarting

if __name__ == "__main__":
    start_stream()
