# CPU Temperature Prediction System

A comprehensive system to predict CPU temperature using Machine Learning, with a real-time FastAPI backend and React dashboard.

## Project Structure

```
├── ML/                    # Machine Learning Core
│   ├── data/              # Datasets (CSV files)
│   ├── scripts/           # Data collection utilities
│   │   ├── script_windows.py
│   │   ├── script_mac.py
│   │   └── combine.py
│   ├── training/          # Model training scripts
│   │   └── train.py
│   ├── testing/           # Real-time inference tests
│   │   └── realtime_test.py
│   └── models/            # Trained models & scaler
│       ├── LightGBM/      # Best model (91.9% confidence)
│       ├── XGBoost/
│       ├── RandomForest/
│       ├── ExtraTrees/
│       ├── LinearRegression/
│       ├── RidgeRegression/
│       └── data_scaler.pkl
├── backend/               # FastAPI Backend
│   ├── main.py            # API server
│   └── requirements.txt
└── frontend/              # React Dashboard
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── ContextLayer.jsx
    │   │   ├── SystemLayer.jsx
    │   │   ├── IntelligenceLayer.jsx
    │   │   └── DecisionLayer.jsx
    │   └── index.css
    └── package.json
```

## ⚡ How to Run Everything

```bash
# 1. Start OpenHardwareMonitor (run as Administrator, keep in background)

# 2. Start the backend (from project root)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 3. Start the frontend (in a separate terminal)
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` — the dashboard will display live predictions.

## 🚀 Quick Start (Data Collection & Training)

### Prerequisites
- **Windows 10/11** or **macOS**
- **Python 3.10+**
- **Temperature monitoring tool** (platform-specific, see below)

### Platform-Specific Setup

####  **Windows Users**

1. **Install OpenHardwareMonitor:**
   - Download from https://openhardwaremonitor.org/
   - Extract and run `OpenHardwareMonitor.exe` as Administrator
   - Keep it running in background

2. **Use:** `ML/scripts/script_windows.py`

####  **macOS Users**

1. **Install temperature monitoring tool** (choose one):
   
   **Option 1: osx-cpu-temp (Recommended)**
   ```bash
   brew install osx-cpu-temp
   ```
   
   **Option 2: iStats**
   ```bash
   sudo gem install iStats
   ```

2. **Use:** `ML/scripts/script_mac.py`

### Setup on Each System

1. **Clone this repository:**
   ```bash
   git clone https://github.com/RudrakshChugh/Predict-CPU-Temperature
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r ML/requirements.txt
   ```

3. **Choose the correct script for your platform:**
   - **Windows:** Edit `ML/scripts/script_windows.py`
   - **macOS:** Edit `ML/scripts/script_mac.py`
   
   Change these two lines:
   ```python
   SYSTEM_ID = "S1"              # Change to S2, S3, S4, S5 for other systems
   OUTPUT_FILE = "system_S1.csv" # Change to system_S2.csv, etc.
   ```

4. **Run data collection (30 minutes):**
   ```bash
   # Windows:
   python ML/scripts/script_windows.py
   
   # macOS:
   python ML/scripts/script_mac.py
   ```

5. **After all systems complete, combine data:**
   ```bash
   python ML/scripts/combine.py
   ```

## 📊 Model Performance

| Model | R² Score | MAE (°C) | RMSE (°C) | Confidence |
|---|---|---|---|---|
| **LightGBM** ⭐ | 0.9761 | 0.59 | 0.96 | **91.9%** |
| RandomForest | 0.9749 | 0.62 | 0.97 | 91.4% |
| XGBoost | 0.9731 | 0.64 | 1.01 | 91.2% |
| ExtraTrees | 0.9661 | 0.74 | 1.12 | 89.4% |
| LinearRegression | 0.9570 | 0.85 | 1.26 | 87.4% |
| RidgeRegression | 0.9565 | 0.86 | 1.25 | 87.4% |


## 🔌 Backend API

### Setup
```bash
# From project root
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Real-time system metrics + ML prediction |
| `/health` | GET | Health check (returns model name) |
| `/docs` | GET | Auto-generated Swagger UI |

### Response Shape (`/api/status`)
```json
{
  "context": {
    "cpuUtil": 45.2, "memoryUsage": 13.1, "clockSpeed": 3.2,
    "ambientTemp": 22.0, "voltage": 12.0, "current": 6.2
  },
  "system": { "currentTemp": 48.0, "predictedTemp": 47.5 },
  "intelligence": {
    "predictionHistory": [{ "time": 1, "temp": 47.5 }, { "time": 2, "temp": 48.1 }]
  },
  "decision": {
    "status": "STABLE",
    "reason": "System operating efficiently. Predicted temp: 47.5°C.",
    "action": "No action required."
  }
}
```

>  **Requires OpenHardwareMonitor running** for accurate temperature readings.

## 🖥️ Frontend Dashboard

Built with **Vite + React 19 + TailwindCSS 4 + Recharts**.

### Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` (backend must be running on port 8000).

### Dashboard Layout
- **Context Layer** (left) — Real-time input parameters (CPU%, Memory, Clock, Voltage, Current)
- **System Layer** (center) — Server visualization with temperature-coded glow
- **Intelligence Layer** (top right) — Rolling prediction chart with 80°C reference line
- **Decision Layer** (bottom right) — STABLE / WATCH / CRITICAL status with recommended action

##  Data Collected

The data collection scripts capture **20 features** per sample:

| Category | Features |
|---|---|
| CPU | `cpu_util`, `cpu_util_avg_10s/30s/60s`, `cpu_util_peak_10s`, `cpu_util_var_30s` |
| Memory | `mem_util` |
| Hardware | `clock_speed`, `cpu_temp`, `cpu_temp_delta`, `thermal_momentum` |
| Power | `voltage`, `current`, `power_estimated`, `power_source` |
| Environment | `ambient_temp` |
| Behavior | `time_since_idle`, `load_pattern`, `num_processes` |
| Identity | `system_id`, `timestamp` |

Additionally, `train.py` engineers 6 extra features: `hour`, `minute`, `temp_lag_1/2/5`, `temp_rolling_5`.

##  Safety

 **100% Read-only** - No system modifications  
 Only monitors system metrics  
 Only writes to CSV files  

##  Files

### ML
- `ML/scripts/script_windows.py` — Data collection for Windows
- `ML/scripts/script_mac.py` — Data collection for macOS
- `ML/scripts/combine.py` — Combines CSV files from all systems
- `ML/training/train.py` — Trains 6 models, saves best with metrics
- `ML/testing/realtime_test.py` — Standalone real-time inference test
- `ML/requirements.txt` — Python dependencies for ML

### Backend
- `backend/main.py` — FastAPI server (loads LightGBM, serves predictions)
- `backend/requirements.txt` — Python dependencies for backend

### Frontend
- `frontend/src/App.jsx` — Main component (fetches from backend every 1s)
- `frontend/src/components/` — ContextLayer, SystemLayer, IntelligenceLayer, DecisionLayer

## Output

Each system generates: `system_S1.csv`, `system_S2.csv`, etc.  
Combined output: `combined_system_data.csv`

---

**Data Collection Duration:** 30 minutes per system  
**Platform:** Windows & macOS supported  
**Tech Stack:** Python · FastAPI · LightGBM · React 19 · Vite · TailwindCSS · Recharts
