import cv2
import numpy as np
from ultralytics import YOLO
import torch
MODEL_PATH = "yolov8s_1_finetuned.pt"
USE_GPU = True
DETECT_EVERY_N = 1
RESIZE_TO = None
FLOW_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=11,
    iterations=3, poly_n=5, poly_sigma=1.1, flags=0
)
SAMPLE_STEP = 6
MIN_MAG = 0.08
MIN_SAMPLE_COUNT = 4
ACCEL_THRESHOLD = 0.60          
MIN_WALK_VELOCITY = 0.20        
EMA_ALPHA = 0.25               
PANIC_RATIO = 0.80              
NORMAL_MAX = 0.40
WARNING_MAX = 0.75
WINDOW_NAME = "Live Crowd Panic Detector"
device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
print("Using:", device)
model = YOLO(MODEL_PATH).to(device)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ret, frame = cap.read()
if not ret:
    raise SystemExit("Error: Could not access webcam!")
if RESIZE_TO:
    frame = cv2.resize(frame, RESIZE_TO)
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_centers = {}
prev_vel_ema = {}       
next_id = 0
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)
def match_id(prev_centers, cx, cy, max_dist2=15000):
    best = None
    best_d = max_dist2
    for pid, (px, py) in prev_centers.items():
        d = (px - cx)**2 + (py - cy)**2
        if d < best_d:
            best = pid
            best_d = d
    return best
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected!")
        break
    if RESIZE_TO:
        frame = cv2.resize(frame, RESIZE_TO)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **FLOW_PARAMS)
    fx = flow[..., 0]
    fy = flow[..., 1]
    boxes = []
    if frame_idx % DETECT_EVERY_N == 0:
        results = model(frame, conf=0.30, classes=[0], verbose=False, device=device)
        for r in results:
            for b in r.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                if (x2-x1) < 20 or (y2-y1) < 40:
                    continue
                boxes.append((x1,y1,x2,y2))
    new_centers = {}
    detections = []
    prev_map = prev_centers.copy()
    for (x1,y1,x2,y2) in boxes:
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        pid = match_id(prev_map, cx, cy)
        if pid is None:
            pid = next_id
            next_id += 1
            prev_vel_ema[pid] = 0.0
        new_centers[pid] = (cx,cy)
        detections.append((pid, x1, y1, x2, y2))
    prev_centers = new_centers
    spike_count = 0
    total_people = len(detections)
    for (pid,x1,y1,x2,y2) in detections:
        total_dx = total_dy = 0
        count = 0
        for yy in range(y1,y2,SAMPLE_STEP):
            for xx in range(x1,x2,SAMPLE_STEP):
                vx = fx[yy,xx]
                vy = fy[yy,xx]
                if np.hypot(vx,vy) < MIN_MAG:
                    continue
                total_dx += vx
                total_dy += vy
                count += 1
        if count < MIN_SAMPLE_COUNT:
            raw_vel = 0
        else:
            raw_vel = np.hypot(total_dx/count, total_dy/count)
        if raw_vel < MIN_WALK_VELOCITY:
            raw_vel = 0.0
        old_vel = prev_vel_ema.get(pid, 0.0)
        vel_ema = EMA_ALPHA*raw_vel + (1-EMA_ALPHA)*old_vel
        prev_vel_ema[pid] = vel_ema
        accel = abs(vel_ema - old_vel)
        if vel_ema > MIN_WALK_VELOCITY and accel > ACCEL_THRESHOLD:
            spike_count += 1
    spike_ratio = spike_count / total_people if total_people > 0 else 0
    if spike_ratio >= PANIC_RATIO:
        color = (0,0,255)
        status = "PANIC"
    elif spike_ratio >= WARNING_MAX:
        color = (0,255,255)
        status = "WARNING"
    else:
        color = (255,255,255)
        status = "NORMAL"
    for (pid,x1,y1,x2,y2) in detections:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame,
                f"{status} | crowd spike: {spike_ratio*100:.1f}%",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3)
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prev_gray = gray.copy()
    frame_idx += 1
cap.release()
cv2.destroyAllWindows()