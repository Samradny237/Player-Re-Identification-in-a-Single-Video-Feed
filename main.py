import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

# ---- Load YOLOv11 model ----
MODEL_PATH = '/Users/samradny/Desktop/ML_assignment/best.pt'
model = YOLO(MODEL_PATH)

# ---- Load video ----
cap = cv2.VideoCapture('15sec_input_720p.mp4')
output_path = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

# ---- Helper for ID tracking ----
next_id = 0
player_memory = {}  # {player_id: [x1, y1, x2, y2]}
frame_count = 0

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter_area / float(bb1_area + bb2_area - inter_area + 1e-6)

# ---- Main loop ----
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Run detection
    results = model(frame, verbose=False)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = r[:6]
        if int(cls) == 0 and conf > 0.3:  # Class 0 = player
            detections.append([int(x1), int(y1), int(x2), int(y2)])

    # ID assignment
    assigned = {}
    for det in detections:
        matched = False
        for pid, prev_det in player_memory.items():
            if iou(det, prev_det) > 0.3:
                assigned[pid] = det
                matched = True
                break
        if not matched:
            assigned[next_id] = det
            next_id += 1
    player_memory = assigned

    # Draw bounding boxes with IDs
    for pid, (x1, y1, x2, y2) in player_memory.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Player Re-Identification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Cleanup ----
cap.release()
out.release()
cv2.destroyAllWindows()
