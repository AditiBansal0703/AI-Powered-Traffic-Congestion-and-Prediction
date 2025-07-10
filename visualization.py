import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Path to folder containing folders of extracted frames
frames_dir = r"C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\frames"  # Update this path

# Initialize lists
resolutions = []
brightness_values = []
vehicle_densities = []

# Loop through video folders and sample a few frames
for video_folder in os.listdir(frames_dir):
    folder_path = os.path.join(frames_dir, video_folder)
    if not os.path.isdir(folder_path):
        continue

    frame_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    sample_frames = frame_files[:5]  # Sample first 5 frames

    for frame_file in sample_frames:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        # Frame resolution
        h, w = frame.shape[:2]
        resolutions.append((w, h))

        # Brightness: average pixel intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_values.append(brightness)

        # Estimate vehicle density via thresholding
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        density = np.sum(binary == 255) / binary.size
        vehicle_densities.append(density)

# ---- Visualization 1: Frame Resolution Distribution ----
res_counter = Counter(resolutions)
labels = list(res_counter.keys())
values = list(res_counter.values())
labels_str = [f"{res[0]}x{res[1]}" for res in labels]

plt.figure(figsize=(10, 6))
plt.bar(labels_str, values, color='green', alpha=0.7)
plt.title("Frame Resolution Distribution across Videos")
plt.xlabel("Resolution (Width x Height)")
plt.ylabel("Number of Frames")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# ---- Visualization 2: Brightness Distribution ----
plt.figure(figsize=(10, 6))
plt.hist(brightness_values, bins=20, color='orange', alpha=0.7)
plt.title("Brightness Distribution of Sampled Frames")
plt.xlabel("Average Brightness")
plt.ylabel("Number of Frames")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# ---- Visualization 3: Vehicle Density Distribution ----
vehicle_density_percent = [round(d * 100, 2) for d in vehicle_densities]

plt.figure(figsize=(10, 6))
plt.hist(vehicle_density_percent, bins=20, color='blue', alpha=0.7)
plt.title("Estimated Vehicle Density Distribution")
plt.xlabel("Vehicle Density (%)")
plt.ylabel("Number of Frames")
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()
