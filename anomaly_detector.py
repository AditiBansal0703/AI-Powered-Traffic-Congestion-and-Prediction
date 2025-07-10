import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import joblib  # or use pickle
from sklearn.preprocessing import StandardScaler

# === Load models ===
yolo_model = YOLO('yolov8n-seg.pt')  # Segmentation model
accident_model = load_model('model.h5')  # CNN model (expects image input)
with open('traffic_classifier.pkl', 'rb') as f:
    traffic_classifier = joblib.load(f)

# === Define label categories ===
label_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Unusual Congestion'}

# === Vehicle classes in YOLO COCO segmentation ===
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

def get_vehicle_mask(result):
    mask = np.zeros((result.orig_shape[0], result.orig_shape[1]), dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return mask

    names = result.names
    for i, box in enumerate(result.boxes.cls):
        cls_id = int(box.item())
        cls_name = names.get(cls_id, None)
        if cls_name in vehicle_classes and result.masks is not None:
            seg = result.masks.xy[i]
            points = np.array(seg, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [points], 255)
    return mask

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    accident_detected = False
    occupancy_ratios = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # === Run YOLO segmentation ===
        results = yolo_model(frame)[0]

        # === Create vehicle mask ===
        vehicle_mask = get_vehicle_mask(results)
        road_area = frame.shape[0] * frame.shape[1]
        vehicle_area = np.sum(vehicle_mask > 0)

        occupancy_ratio = vehicle_area / road_area
        occupancy_ratios.append(occupancy_ratio)

        # === Accident detection using CNN on frame ===
        try:
            frame_resized = cv2.resize(frame, (250, 250))  # Match model input shape
            frame_normalized = frame_resized.astype('float32') / 255.0
            frame_input = np.expand_dims(frame_normalized, axis=0)  # Shape: (1, 250, 250, 3)
            prediction = accident_model.predict(frame_input, verbose=0)
            if prediction[0][0] > 0.5:
                accident_detected = True
        except Exception as e:
            print(f"❌ Error processing frame {frame_count}: {e}")

    cap.release()

    if len(occupancy_ratios) == 0:
        print("❌ No frames read from video.")
        return {
            "accident": False,
            "congestion": "Unknown"
        }

    # === Aggregate stats for traffic level classification ===
    occ_mean = np.mean(occupancy_ratios)
    occ_std = np.std(occupancy_ratios)
    occ_min = np.min(occupancy_ratios)
    occ_max = np.max(occupancy_ratios)

    # Dummy placeholder — no "unusual_pct" at frame-level in this script
    features = pd.DataFrame([[occ_mean, occ_std, occ_min, occ_max, 0]], columns=[
        "occ_mean", "occ_std", "occ_min", "occ_max", "unusual_pct"
    ])

    # === Predict video-level traffic category ===
    traffic_pred = traffic_classifier.predict(features)[0]
    traffic_label = label_map[traffic_pred]

    print("\n📊 Final Prediction Results:")
    print("🚧 Accident Detected in Video:", "✅ Yes" if accident_detected else "❌ No")
    print("🛣️ Traffic Congestion Category:", traffic_label)
    
    return {
        "accident": accident_detected,
        "congestion": traffic_label
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AI-based anomaly detection on a traffic video.")
    parser.add_argument("video_path", type=str, help="Path to the traffic video file")
    args = parser.parse_args()

    analyze_video(args.video_path)
