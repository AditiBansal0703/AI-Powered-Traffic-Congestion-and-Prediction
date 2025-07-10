import cv2
import os

# path where your videos are
video_dir = r"C:\Users\Lenovo\Desktop\nitttr_project\video"

# path where you want to save extracted frames
output_dir = r"C:\Users\Lenovo\Desktop\nitttr_project\frames"

# how many frames to extract per video
num_frames = 5

# make sure the output folder exists
os.makedirs(output_dir, exist_ok=True)

# list all video files
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing {video_file}...")

    # create subfolder for this video
    video_name = os.path.splitext(video_file)[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // num_frames, 1)  # avoid division by zero

    frame_idx = 0
    extracted_count = 0

    while extracted_count < num_frames and cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        # save frame
        frame_filename = os.path.join(video_output_dir, f"frame_{extracted_count+1}.jpg")
        cv2.imwrite(frame_filename, frame)
        extracted_count += 1
        frame_idx += frame_step

    cap.release()

print("✅ Frame extraction completed for all videos!")
