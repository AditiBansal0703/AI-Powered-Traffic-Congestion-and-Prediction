import os
import cv2

# Specify your video directory path
video_dir_path = r'C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\video'  # <-- update to your actual path

# Get all video files (add more extensions if needed)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(video_dir_path) if f.lower().endswith(video_extensions)]

# Function to check if a video file is corrupt
def is_video_corrupt(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open: {video_path}")
            cap.release()
            return True
        ret, _ = cap.read()
        cap.release()
        if not ret:
            print(f"Cannot read first frame: {video_path}")
        return not ret
    except Exception as e:
        print(f"Exception for {video_path}: {str(e)}")
        return True

# Check for corrupt videos
corrupt_videos = []
for video_file in video_files:
    full_path = os.path.join(video_dir_path, video_file)
    if is_video_corrupt(full_path):
        corrupt_videos.append(video_file)

# Summary
print("\n🔍 Scan Complete")
if corrupt_videos:
    print("❌ Corrupt or unreadable video files found:")
    for file in corrupt_videos:
        print(f"- {file}")
else:
    print("✅ All video files are readable and intact.")