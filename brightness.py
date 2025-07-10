import os
import cv2
import matplotlib.pyplot as plt

# your extracted frames directory
frames_dir = r"C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\frames"

# function to calculate brightness of an image
def calculate_brightness(frame):
    return frame.mean()

all_brightness = []

# loop through each subfolder
for video_folder in os.listdir(frames_dir):
    video_folder_path = os.path.join(frames_dir, video_folder)
    if os.path.isdir(video_folder_path):
        for frame_file in os.listdir(video_folder_path):
            if frame_file.endswith(('.jpg', '.png')):
                frame_path = os.path.join(video_folder_path, frame_file)
                img = cv2.imread(frame_path)
                if img is not None:
                    brightness = calculate_brightness(img)
                    all_brightness.append(brightness)

# plot the histogram
plt.figure(figsize=(10, 5))
plt.hist(all_brightness, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title("Brightness Distribution Across All Extracted Frames")
plt.xlabel("Brightness Level")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
