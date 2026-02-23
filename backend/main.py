"""
CPU Thermal Predictor — FastAPI Backend
Default model: LightGBM (best performer: 91.9% confidence, MAE 0.59°C)

Run from project root:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import psutil
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# ──────────────────────────────────────────────
# PATHS  (resolved from this file → project root)
# ──────────────────────────────────────────────
_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(_ROOT, "ML", "models")
DEFAULT_MODEL = "LightGBM"

# ──────────────────────────────────────────────
# LOAD MODEL & SCALER
# ──────────────────────────────────────────────
def load_best_model():
    """Load LightGBM by default. Falls back to scanning metrics if not found."""
    model_path = os.path.join(MODELS_DIR, DEFAULT_MODEL, f"{DEFAULT_MODEL}_model.pkl")

    if os.path.exists(model_path):
        print(f"[✔] Loaded model: {DEFAULT_MODEL} (default)")
        return joblib.load(model_path), DEFAULT_MODEL

    # Fallback: scan metrics.txt files and pick highest Confidence Score
    print(f"[!] {DEFAULT_MODEL} not found — scanning for best available model...")
    best_name, best_score, best_path = None, -1, None

    for folder in os.listdir(MODELS_DIR):
        metrics_file = os.path.join(MODELS_DIR, folder, "metrics.txt")
        if not os.path.isfile(metrics_file):
            continue
        try:
            with open(metrics_file) as f:
                for line in f:
                    if line.startswith("Confidence Score:"):
                        score = float(line.split(":")[1].strip().replace("%", ""))
                        if score > best_score:
                            pkl = os.path.join(MODELS_DIR, folder, f"{folder}_model.pkl")
                            if os.path.exists(pkl):
                                best_score, best_name, best_path = score, folder, pkl
        except Exception:
            continue

    if best_path:
        print(f"[✔] Fallback: loaded {best_name} (Confidence: {best_score}%)")
        return joblib.load(best_path), best_name

    raise FileNotFoundError(f"No trained model found in {MODELS_DIR}")


model, model_name = load_best_model()
scaler = joblib.load(os.path.join(MODELS_DIR, "data_scaler.pkl"))
print(f"[✔] Scaler loaded.")

# Try WMI for real CPU temperature (Windows + OpenHardwareMonitor)
try:
    import wmi as _wmi
    _w = _wmi.WMI(namespace="root\\OpenHardwareMonitor")
    _ohm_available = True
    print("[✔] OpenHardwareMonitor connected — real CPU temp available.")
except Exception:
    _w = None
    _ohm_available = False
    print("[!] OpenHardwareMonitor not detected — cpu_temp will be None.")

# ──────────────────────────────────────────────
# ROLLING STATE  (in-memory, resets on restart)
# ──────────────────────────────────────────────
_cpu_10s    = deque(maxlen=10)
_cpu_30s    = deque(maxlen=30)
_cpu_60s    = deque(maxlen=60)
_therm_hist = deque(maxlen=10)
_temp_lag_hist = deque(maxlen=6)   # last 6 temps for lag-1/2/5 + rolling-5
_temp_history: List[dict] = []     # for prediction history chart (max 20 points)

_prev_temp     = None
_last_idle_t   = time.time()
IDLE_THRESHOLD = 5.0
OVERHEAT_C     = 80.0

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def get_cpu_temp() -> float | None:
    if not _ohm_available:
        return None
    try:
        for sensor in _w.Sensor():
            if sensor.SensorType == "Temperature" and "CPU Package" in sensor.Name:
                return round(float(sensor.Value), 2)
    except Exception:
        pass
    return None


def get_voltage_current(cpu_util: float):
    battery = psutil.sensors_battery()
    if battery:
        voltage = 11.1 if battery.power_plugged else 10.8
        current = round(1.5 + (cpu_util / 100.0) * 3.5, 2)
    else:
        voltage = 12.0
        current = round(2.0 + (cpu_util / 100.0) * 8.0, 2)
    return round(voltage, 2), current


def collect_and_predict():
    """Collect live metrics, predict temperature, return full payload."""
    global _prev_temp, _last_idle_t

    # ── Raw metrics ──
    cpu_util = psutil.cpu_percent(interval=None)
    mem      = psutil.virtual_memory()
    freq     = psutil.cpu_freq()
    clock    = round(freq.current / 1000, 3) if freq else 0.0   # MHz → GHz

    cpu_temp = get_cpu_temp()

    # Temperature delta
    cpu_temp_delta = 0.0
    if _prev_temp is not None and cpu_temp is not None:
        cpu_temp_delta = round(cpu_temp - _prev_temp, 2)
    _prev_temp = cpu_temp

    # Thermal momentum (recency-weighted avg)
    thermal_momentum = 0.0
    if cpu_temp is not None:
        _therm_hist.append(cpu_temp)
        weights = list(range(1, len(_therm_hist) + 1))
        thermal_momentum = round(
            sum(t * w for t, w in zip(_therm_hist, weights)) / sum(weights), 2
        )

    # Power
    battery     = psutil.sensors_battery()
    power_src   = 1 if (not battery or battery.power_plugged) else 0
    voltage, current = get_voltage_current(cpu_util)
    power_est   = round(voltage * current, 2)

    # Idle tracking
    if cpu_util < IDLE_THRESHOLD:
        _last_idle_t = time.time()
    time_since_idle = round(time.time() - _last_idle_t, 2)

    # Load pattern
    load_pattern = 0 if cpu_util < 10 else (1 if cpu_util < 30 else (2 if cpu_util < 60 else 3))
    num_processes = len(psutil.pids())

    # Rolling buffers
    _cpu_10s.append(cpu_util)
    _cpu_30s.append(cpu_util)
    _cpu_60s.append(cpu_util)

    avg_10s  = round(sum(_cpu_10s) / len(_cpu_10s), 2)
    avg_30s  = round(sum(_cpu_30s) / len(_cpu_30s), 2)
    avg_60s  = round(sum(_cpu_60s) / len(_cpu_60s), 2)
    peak_10s = round(max(_cpu_10s), 2)
    var_30s  = round(sum((x - avg_30s) ** 2 for x in _cpu_30s) / len(_cpu_30s), 2)

    # ── Lag features (from recent temp history) ──
    # Uses actual OHM temp if available, else thermal_momentum as proxy
    _temp_lag_hist.append(cpu_temp if cpu_temp is not None else thermal_momentum)
    hist = list(_temp_lag_hist)

    temp_lag_1   = hist[-2] if len(hist) >= 2 else hist[-1]
    temp_lag_2   = hist[-3] if len(hist) >= 3 else hist[-1]
    temp_lag_5   = hist[-6] if len(hist) >= 6 else hist[0]
    temp_rolling_5 = round(sum(hist[-5:]) / len(hist[-5:]), 4) if hist else 0.0

    now = datetime.now()

    # ── Feature DataFrame — must EXACTLY match train.py column order ──
    # train.py: X = df.drop([cpu_temp, timestamp, system_id])
    # create_features adds: hour, minute, temp_lag_1/2/5, temp_rolling_5
    features = {
        "cpu_util":          [cpu_util],
        "cpu_util_avg_10s":  [avg_10s],
        "cpu_util_avg_30s":  [avg_30s],
        "cpu_util_avg_60s":  [avg_60s],
        "cpu_util_peak_10s": [peak_10s],
        "cpu_util_var_30s":  [var_30s],
        "mem_util":          [mem.percent],
        "clock_speed":       [freq.current if freq else 0],  # raw MHz — matches training
        "cpu_temp_delta":    [cpu_temp_delta],
        "thermal_momentum":  [thermal_momentum],
        "ambient_temp":      [22.0],                         # fixed assumption
        "voltage":           [voltage],
        "current":           [current],
        "power_estimated":   [power_est],
        "power_source":      [power_src],
        "time_since_idle":   [time_since_idle],
        "load_pattern":      [load_pattern],
        "num_processes":     [num_processes],
        "hour":              [now.hour],
        "minute":            [now.minute],
        "temp_lag_1":        [temp_lag_1],
        "temp_lag_2":        [temp_lag_2],
        "temp_lag_5":        [temp_lag_5],
        "temp_rolling_5":    [temp_rolling_5],
    }
    df = pd.DataFrame(features)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    predicted_temp = round(float(model.predict(df_scaled)[0]), 1)
    current_temp   = int(cpu_temp) if cpu_temp is not None else predicted_temp

    # ── History (for chart) ──
    _temp_history.append({"time": datetime.now().second, "temp": predicted_temp})
    if len(_temp_history) > 20:
        _temp_history.pop(0)

    # ── Decision logic ──
    if predicted_temp > 75:
        status = "CRITICAL"
        reason = f"Thermal runaway risk. Predicted temp {predicted_temp}°C exceeds danger threshold."
        action = "IMMEDIATE: Throttle CPU voltage and increase fan speed to 100%."
    elif predicted_temp > 60:
        status = "WATCH"
        reason = f"Temperature rising ({predicted_temp}°C predicted). Above optimal baseline."
        action = "Increase fan speed by 20% preemptively."
    else:
        status = "STABLE"
        reason = f"System operating efficiently. Predicted temp: {predicted_temp}°C."
        action = "No action required."

    return {
        "context": {
            "cpuUtil":     round(cpu_util, 1),
            "memoryUsage": round(mem.used / (1024 ** 3), 2),     # bytes → GB
            "clockSpeed":  clock,
            "ambientTemp": 22.0,                                  # fixed assumption
            "voltage":     voltage,
            "current":     current,
        },
        "system": {
            "currentTemp":   current_temp,
            "predictedTemp": predicted_temp,
        },
        "intelligence": {
            "predictionHistory": list(_temp_history),
        },
        "decision": {
            "status": status,
            "reason": reason,
            "action": action,
        },
        "_meta": {
            "model":     model_name,
            "timestamp": datetime.now().isoformat(),
        }
    }


# ──────────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────────
app = FastAPI(
    title="CPU Thermal Predictor API",
    description=f"Real-time CPU temperature prediction using {model_name}.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prime the psutil CPU counter (first call always returns 0.0)
psutil.cpu_percent(interval=None)


@app.get("/api/status", summary="Get real-time system status + ML prediction")
def get_status():
    """
    Returns live system metrics and the ML-predicted CPU temperature.
    Frontend should poll this every 1 second.
    """
    return collect_and_predict()


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model": model_name}
