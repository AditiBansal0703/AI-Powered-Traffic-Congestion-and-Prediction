import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# ------------------------ PART 1: CHECK FOR CORRUPT VIDEOS ------------------------

video_dir_path = r'C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\video'  # UPDATE if needed
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
video_files = [f for f in os.listdir(video_dir_path) if f.lower().endswith(video_extensions)]

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

corrupt_videos = []
for video_file in video_files:
    full_path = os.path.join(video_dir_path, video_file)
    if is_video_corrupt(full_path):
        corrupt_videos.append(video_file)

print("\n🔍 CORRUPT VIDEO CHECK COMPLETE")
if corrupt_videos:
    print("❌ Corrupt or unreadable video files found:")
    for file in corrupt_videos:
        print(f"- {file}")
else:
    print("✅ All video files are readable and intact.")

# ------------------------ PART 2: CHECK FOR MISSING FRAMES ------------------------

def has_missing_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    actual_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_frames += 1
    
    cap.release()
    if expected_frames != actual_frames:
        print(f"[⚠] {os.path.basename(video_path)} | Expected: {expected_frames}, Actual: {actual_frames}")
        return True
    return False

videos_with_missing_frames = []
for video_file in video_files:
    full_path = os.path.join(video_dir_path, video_file)
    if has_missing_frames(full_path):
        videos_with_missing_frames.append(video_file)

print("\n🎯 MISSING FRAMES CHECK COMPLETE")
if videos_with_missing_frames:
    print("❌ Videos with potential missing/dropped frames:")
    for v in videos_with_missing_frames:
        print(f"- {v}")
else:
    print("✅ All videos have complete frames as per metadata.")

# ------------------------ PART 3: VEHICLE COUNT ESTIMATION ------------------------

frames_dir = r"C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\frames"  # UPDATE if needed
frame_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

def estimate_vehicle_count(frame, threshold_value=180):
    normalized_frame = frame / 255.0
    gray_frame = cv2.cvtColor((normalized_frame * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, threshold_value, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_count = len(contours)
    return vehicle_count, normalized_frame, gray_frame, binary_frame

results = []

# 🔄 Loop through each subfolder (each video)
for folder_name in os.listdir(frames_dir):
    subfolder_path = os.path.join(frames_dir, folder_name)
    if not os.path.isdir(subfolder_path):
        continue

    # Get sorted list of frames in the subfolder
    frame_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(frame_extensions)])

    # Skip the first frame
    frame_files = frame_files[1:]

    for frame_file in frame_files:
        frame_path = os.path.join(subfolder_path, frame_file)
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        count, norm_frame, gray, binary = estimate_vehicle_count(frame)
        results.append({
            'video_folder': folder_name,
            'filename': frame_file,
            'vehicle_count': count
        })

        # Optional: visualize a sample
        if frame_file.endswith("001.jpg"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(frame)
            ax1.set_title("Original Frame")
            ax1.axis('off')

            ax2.imshow(binary, cmap='gray')
            ax2.set_title(f"Binary Thresholded (Vehicle Count: {count})")
            ax2.axis('off')
            plt.tight_layout()
            plt.show()

# Summary print
print("\n✅ VEHICLE COUNT ESTIMATION COMPLETE")
for res in results[:10]:  # show first 10 results
    print(f"{res['video_folder']}/{res['filename']}: {res['vehicle_count']} vehicles")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("vehicle_counts.csv", index=False)
print("📄 Saved vehicle counts to 'vehicle_counts.csv'")


# ----------------- CONFIGURATION -----------------
frames_dir = r"C:\Users\Lenovo\Desktop\SEM6\nitttr_project\nitttr_project\frames"
frame_extensions = ('.jpg', '.jpeg', '.png')
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
yolo_model_path = "yolov8n-seg.pt"

# ----------------- LOAD YOLOv8 MODEL -----------------
model = YOLO(yolo_model_path)
print("✅ YOLOv8 segmentation model loaded.")

results = []

# ----------------- FRAME-LEVEL ANALYSIS -----------------
for folder_name in os.listdir(frames_dir):
    subfolder_path = os.path.join(frames_dir, folder_name)
    if not os.path.isdir(subfolder_path):
        continue

    frame_files = sorted([f for f in os.listdir(subfolder_path) if f.lower().endswith(frame_extensions)])

    for frame_file in frame_files:
        frame_path = os.path.join(subfolder_path, frame_file)
        frame = cv2.imread(frame_path)
        h, w = frame.shape[:2]

        # YOLOv8 segmentation prediction
        preds = model.predict(frame, verbose=False)[0]

        # Build mask for selected vehicle classes
        object_mask = np.zeros((h, w), dtype=np.uint8)
        for i, cls in enumerate(preds.boxes.cls):
            if int(cls.item()) in vehicle_classes:
                mask = preds.masks.data[i].cpu().numpy()
                resized_mask = cv2.resize(mask, (w, h))
                object_mask[resized_mask > 0.5] = 255

        road_mask = cv2.bitwise_not(object_mask)

        road_pixels = np.sum(road_mask > 0)
        object_pixels = np.sum(object_mask > 0)

        occupancy_ratio = 1.0 if road_pixels == 0 else object_pixels / road_pixels

        # Traffic label rule
        if occupancy_ratio < 0.04:
            traffic_level = "Low"
            unusual = False
        elif occupancy_ratio < 0.08:
            traffic_level = "Moderate"
            unusual = False
        elif occupancy_ratio < 0.10:
            traffic_level = "High"
            unusual = False
        else:
            traffic_level = "unusual congestion"
            unusual = True  # High with unusual congestion

        results.append({
            "video_folder": folder_name,
            "frame": frame_file,
            "road_pixels": int(road_pixels),
            "object_pixels": int(object_pixels),
            "occupancy_ratio": round(occupancy_ratio, 4),
            "traffic_level": traffic_level,
            "unusual_congestion": unusual
        })

        
        if frame_file.endswith("001.jpg"):
            overlay = frame.copy()
            overlay[object_mask > 0] = (0, 0, 255)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{traffic_level} | Unusual: {unusual} | Ratio: {occupancy_ratio:.2f}")
            plt.axis('off')
            plt.show()

# ----------------- SAVE FRAME-LEVEL RESULTS -----------------
df = pd.DataFrame(results)
df.to_csv("traffic_analysis.csv", index=False)
print("\n✅ Frame-level traffic data saved to 'traffic_analysis.csv'")

# ----------------- VIDEO-LEVEL AGGREGATION + MODEL TRAINING -----------------
# Load the saved CSV
df = pd.read_csv("traffic_analysis.csv")
df['unusual_congestion'] = df['unusual_congestion'].astype(bool)
df['traffic_level'] = np.where(df['unusual_congestion'], 'Unusual Congestion', df['traffic_level'])

# Aggregate per video
agg_df = df.groupby("video_folder").agg({
    "occupancy_ratio": ["mean", "std", "min", "max"],
    "traffic_level": lambda x: x.value_counts().idxmax(),
    "unusual_congestion": "mean"
}).reset_index()

# Flatten column names
agg_df.columns = ["video_folder", "occ_mean", "occ_std", "occ_min", "occ_max", "traffic_level", "unusual_pct"]

# Encode labels
label_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Unusual Congestion': 3}
agg_df["label"] = agg_df["traffic_level"].map(label_map)
agg_df.dropna(subset=["label"], inplace=True)

# Features and labels
X = agg_df[["occ_mean", "occ_std", "occ_min", "occ_max", "unusual_pct"]]
y = agg_df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
with open("traffic_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✅ Trained model saved to 'traffic_classifier.pkl'")

# Save video-level results
agg_df.to_csv("video_level_summary.csv", index=False)
print("📄 Saved video-level summary to 'video_level_summary.csv'")