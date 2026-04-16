import cv2
import numpy as np
from ultralytics import YOLO
import torch
VIDEO_PATH = "test_6_small.mp4"
MODEL_PATH = "yolov8s_1_finetuned.pt"
USE_GPU = True
DETECT_EVERY_N = 1
RESIZE_TO = None
FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=11,
                   iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
SAMPLE_STEP = 6
MIN_MAG = 0.08
MIN_SAMPLE_COUNT = 4
ACCEL_THRESHOLD = 0.30
NORMAL_MAX = 0.40      
WARNING_MAX = 0.75     
WINDOW_NAME = "Crowd Panic Detector"
SHOW_DEBUG_TEXT = False   
device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
print("Using:", device)
model = YOLO(MODEL_PATH).to(device)
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Error: Cannot read video")
if RESIZE_TO:
    frame = cv2.resize(frame, RESIZE_TO)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_centers = {}
persons_prev_v = {}
next_id = 0
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)
def match_id(prev_centers, cx, cy, max_dist2=10000):
    best_id = None
    best_d = max_dist2
    for pid, (px, py) in prev_centers.items():
        d = (px - cx)**2 + (py - cy)**2
        if d < best_d:
            best_d = d
            best_id = pid
    return best_id
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if RESIZE_TO:
        frame = cv2.resize(frame, RESIZE_TO)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FLOW_PARAMS)
    fx = flow[..., 0]
    fy = flow[..., 1]
    boxes = []
    if frame_idx % DETECT_EVERY_N == 0:
        results = model(frame, conf=0.25, classes=[0], verbose=False, device=device)
        for r in results:
            if not r.boxes:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                if (x2-x1) < 18 or (y2-y1) < 30:
                    continue
                boxes.append((x1, y1, x2, y2))
    new_centers = {}
    detections = []
    prev_map = prev_centers.copy()
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pid = match_id(prev_map, cx, cy)
        if pid is None:
            pid = next_id
            next_id += 1
            persons_prev_v[pid] = 0.0
        new_centers[pid] = (cx, cy)
        detections.append((pid, x1, y1, x2, y2))
    prev_centers = new_centers
    spike_count = 0
    total_people = len(detections)
    for (pid, x1, y1, x2, y2) in detections:
        total_dx = total_dy = 0
        count = 0
        for yy in range(y1, y2, SAMPLE_STEP):
            for xx in range(x1, x2, SAMPLE_STEP):
                vx = fx[yy, xx]
                vy = fy[yy, xx]
                if np.hypot(vx, vy) < MIN_MAG:
                    continue
                total_dx += vx
                total_dy += vy
                count += 1
        if count < MIN_SAMPLE_COUNT:
            velocity = 0
        else:
            velocity = np.hypot(total_dx/count, total_dy/count)
        accel = abs(velocity - persons_prev_v.get(pid, 0))
        persons_prev_v[pid] = velocity
        if accel > ACCEL_THRESHOLD:
            spike_count += 1
    if total_people > 0:
        spike_ratio = spike_count / total_people
    else:
        spike_ratio = 0
    if spike_ratio <= NORMAL_MAX:
        box_color = (255, 255, 255)   
        crowd_status = "NORMAL"
    elif spike_ratio <= WARNING_MAX:
        box_color = (0, 255, 255)     
        crowd_status = "WARNING"
    else:
        box_color = (0, 0, 255)       
        crowd_status = "PANIC"
    for (pid, x1, y1, x2, y2) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(frame,
                f"{crowd_status} - {spike_ratio*100:.1f}% spiking",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                box_color,
                3)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
