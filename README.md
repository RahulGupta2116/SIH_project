# Temple & Pilgrimage Crowd Management (SIH PS 25165)

## Problem Statement
**SIH Problem Statement ID:** 25165  
**Title:** Temple & Pilgrimage Crowd Management (Somnath, Dwarka, Ambaji, Pavagadh)

This project builds an ML-first crowd management solution focused on pilgrimage safety and operational efficiency. The implementation in this repository addresses major PS needs through:
- **Crowd forecasting** (daily visitor prediction)
- **Real-time panic detection** (computer vision + motion analytics)
- **Operational API layer** (FastAPI inference service)

---

## What This Repository Contains

### 1) Crowd Prediction Model (Time-Series Regression)
- Notebook: `crowd_prediction_model/cat+Lgbm.ipynb`
- Training data: `crowd_prediction_model/test.csv`
- Trained artifacts:
  - `crowd_prediction_model/blended_model.pkl`
  - `crowd_prediction_model/scaler.pkl`

### 2) Prediction API for Deployment
- API: `model_api/app.py`
- Runtime artifacts:
  - `model_api/models/blended_model.pkl`
  - `model_api/models/scaler.pkl`
  - `model_api/test_clean_encoded_2.csv`
- API dependencies: `model_api/requirements.txt`

### 3) Panic Detection for Safety Monitoring
- Barricade panic detection: `panic_detection/panic_detection_in_barricade/panic.py`
- Temple panic detection (live feed): `panic_detection/panic_detection_in_temple/prediction(panic)_on_live_feed.py`
- Temple panic detection (video): `panic_detection/panic_detection_in_temple/prediction(panic)_on_video.py`
- Alarm handling: `panic_detection/panic_detection_in_barricade/alarm.py`
- Detection weights: `yolov8*.pt`, `yolov8n.onnx`

---

## End-to-End Solution Strategy (How It Solves PS 25165)

### A) AI/ML-based Crowd Prediction
The model predicts likely visitor volume for a given day using weather, weekday/holiday/festival context, and historical lag behavior.

**Impact for temples (Somnath/Dwarka/Ambaji/Pavagadh style deployment):**
- Early warning of high-footfall days
- Better police/volunteer deployment
- Better medical and queue resource planning

### B) Smart Monitoring for Panic & Safety
The CV modules detect people + motion spikes and classify crowd state as **NORMAL / WARNING / PANIC**.

**Impact:**
- Faster detection of abnormal crowd movement
- Early intervention before stampede-like risk escalation
- On-ground alerting via audio siren/beep signals

### C) API Layer for Integration
FastAPI service exposes prediction endpoint (`/predict`) for app/dashboard integration.

**Impact:**
- Can power queue dashboards, admin panels, and command center workflows
- Enables integration with mobile/web systems for informed crowd advisories

---

## ML Workflow Implemented in Notebook

## 1. Data Loading & Normalization
- Reads `test.csv`
- Parses `date`
- Sorts chronologically
- Handles missing festival names

## 2. Festival Encoding
Festival names are mapped to numeric categories:
- None → 0
- Diwali → 1
- Holi → 2
- Janmashtami → 3
- Makar Sakranti → 4
- MahaShivratri → 5
- Navratri → 6

## 3. Feature Engineering
Features created for seasonal and behavioral trends:

### Calendar/Temporal features
- `day_of_week`, `month`, `day_of_year`, `year`, `weekofyear`

### Cyclical encoding
- `sin_dow`, `cos_dow`
- `sin_month`, `cos_month`
- `sin_year`, `cos_year`

### Demand memory (historical signal)
- Lags: `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_30`, `lag_60`

### Trend and volatility
- Rolling means/std: `roll_mean_7`, `roll_std_7`, `roll_mean_14`, `roll_std_14`, `roll_mean_30`, `roll_std_30`
- Exponential moving averages: `ema_7`, `ema_30`

### Context features already in data
- `is_weekend`, `festival_flag`, `public_holiday`
- Weather: `temp_avg_c`, `precipitation_mm`

**Target:** `visitor_count`

## 4. Time-Aware Split
- Holdout strategy: last **365 days** as test set
- Train/test are split chronologically (no random leakage)

## 5. Scaling
- `StandardScaler` fit on train features
- Saved as model artifact for consistent inference pipeline

## 6. Model Training
Two gradient boosting regressors are trained and tuned:
- **LightGBM Regressor** (Optuna-tuned)
- **CatBoost Regressor** (Optuna-tuned)

## 7. Ensemble Methods
- **Weighted blending** of CatBoost + LightGBM (optimized blend weight)
- **Stacking** with Ridge meta-model on out-of-fold predictions

## 8. Artifact Export
Saved model package includes:
- `blended_model.pkl` (contains cat model, lgb model, and blend weight)
- `scaler.pkl`

---

## Model Performance (from notebook outputs)

Dataset summary:
- Total rows loaded: **3653**
- Train rows: **3228**
- Test rows: **365**

Base model test metrics:
- **CatBoost** → MAE: `5868.76`, RMSE: `7418.60`, R²: `0.8898`
- **LightGBM** → MAE: `6905.16`, RMSE: `8796.71`, R²: `0.8450`

Ensemble test metrics:
- **Blended** → MAE: `6583.75`, RMSE: `8357.03`, R²: `0.8601`
- **Stacked** → MAE: `5789.39`, RMSE: `7897.73`, R²: `0.8751`

Note: In this run, CatBoost is strongest overall; blending/stacking provide additional comparison paths.

---

## API Inference Architecture (`model_api/app.py`)

### Startup behavior
- Loads model and scaler from `model_api/models/`
- Loads historical reference data from `model_api/test_clean_encoded_2.csv`
- Enables CORS for local frontend ports

### Input schema
`POST /predict` expects:
- `date` (str)
- `temperature` (float)
- `precipitation` (float)
- `festival` (str)
- `temple_name` (str)
- `day_of_week` (int)
- `is_weekend` (int)
- `festival_flag` (int)
- `public_holiday` (int)

### Runtime feature generation
- Reuses historical sequence
- Creates all engineered temporal/lag/statistical features
- Applies trained scaler
- Runs blended model prediction

### Output
- `predicted_visitors` (int)
- `status`
- `date`

### Additional API routes
- `GET /` → service health message
- `GET /health` → service health + historical row count

### Database hook
- `save_to_database` attempts insertion into PostgreSQL (`real_world_data`) for incoming rows.
- This supports real deployment feedback loops (actual data capture + model refresh potential).

---

## Panic Detection Modules (Computer Vision)

## 1) Barricade-side Detection (`panic.py`)
Pipeline:
1. Detects persons using YOLO
2. Computes dense optical flow between frames
3. Estimates per-person motion direction and magnitude
4. Compares movement direction against expected crowd-flow side
5. Computes corrected-behavior percentage
6. Classifies state:
   - High compliance → NORMAL
   - Medium compliance → WARNING
   - Low compliance → PANIC
7. Triggers audio via `alarm.py` (`stop_alarm`, `fast_beep`, `siren`)

This is useful in barricaded queue lanes to detect reverse/chaotic movement.

## 2) Temple-area Detection (Live/Video)
Files:
- `prediction(panic)_on_live_feed.py`
- `prediction(panic)_on_video.py`

Pipeline:
1. Person detection (YOLO)
2. Optical-flow velocity extraction per person
3. Frame-to-frame acceleration/spike estimation
4. Spike ratio = people with abrupt motion / total detected people
5. Crowd state classification:
   - NORMAL
   - WARNING
   - PANIC

This supports open-area monitoring where panic often appears as sudden collective speed spikes.

---

## Mapping to SIH Expected Solution Components

- **AI/ML Crowd Prediction Models** ✅  
  Implemented via CatBoost/LightGBM + feature-rich time-series pipeline.

- **Smart Queue / Real-Time Updates** ⚠️ Partial  
  Prediction API is available and can feed dashboards/apps; full queue-slot product UI is not in this repo.

- **IoT & Surveillance Analytics** ✅  
  CV-based crowd safety monitoring with YOLO + optical flow.

- **Emergency & Safety Response** ✅  
  Panic state detection and alarm triggers are implemented.

- **Traffic & Mobility Management** ⚠️ Future scope  
  Can be added by integrating traffic feeds + route recommendation engine.

- **Pilgrim Engagement Platform** ⚠️ Partial  
  Backend API foundation exists; full multilingual app not included here.

- **Accessibility Features** ⚠️ Future scope  
  Can be layered at application UX/workflow level.

---

## How to Run

## A) Crowd Prediction API
```bash
cd model_api
pip install -r requirements.txt
python app.py
```
Server runs at `http://0.0.0.0:8000`.

Try:
- `GET /`
- `GET /health`
- `POST /predict`

Example payload:
```json
{
  "date": "2026-04-20",
  "temperature": 33.5,
  "precipitation": 0.0,
  "festival": "None",
  "temple_name": "Ambaji",
  "day_of_week": 0,
  "is_weekend": 0,
  "festival_flag": 0,
  "public_holiday": 0
}
```

## B) Panic Detection (Barricade)
```bash
cd panic_detection/panic_detection_in_barricade
python panic.py
```
Press `q` to exit.

## C) Panic Detection (Temple Live Feed)
```bash
cd panic_detection/panic_detection_in_temple
python "prediction(panic)_on_live_feed.py"
```

## D) Panic Detection (Temple Video)
```bash
cd panic_detection/panic_detection_in_temple
python "prediction(panic)_on_video.py"
```

---

## Dependencies
Primary dependency list exists at:
- `model_api/requirements.txt`

For panic detection scripts, ensure environment has:
- `opencv-python`
- `numpy`
- `ultralytics`
- `torch`

Audio alert scripts expect Linux `aplay` utility.

---

## Current Limitations
- Dataset appears temple-specific (Ambaji-focused in shared sample), so transfer learning/retraining is needed for other temple towns.
- Full production orchestration (camera stream manager, alert dashboard, operator app) is outside current repo.
- Queue tokenization, multilingual pilgrim app, and traffic optimization are not yet implemented end-to-end here.
- `model_api/requirements.txt` has repeated entries and extra libraries that are not required by `app.py`.

---

## Recommended Next Improvements
1. Add multi-temple data (Somnath, Dwarka, Pavagadh) and retrain with temple embeddings.
2. Add external signals: event calendars, weather forecast APIs, holiday metadata, transport arrivals.
3. Build real-time dashboard combining:
   - predicted next-day/next-week load,
   - current panic index,
   - live staffing recommendation.
4. Add automated retraining/MLOps pipeline from DB-collected real-world data.
5. Integrate alert escalation matrix (police, medical, volunteer control rooms).
6. Add accessibility-first user modules (priority lanes, assisted navigation, emergency help points).

---

## Conclusion
This repository demonstrates a practical ML solution for pilgrimage crowd management:
- **Predictive intelligence** for expected crowd surges
- **Vision-based safety intelligence** for panic detection
- **Deployable API layer** for operational integration

Together, these components form a strong prototype foundation for SIH PS 25165 and can be scaled toward a full smart-temple command-and-control platform.
