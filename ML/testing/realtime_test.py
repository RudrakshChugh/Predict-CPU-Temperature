import psutil
import time
import pandas as pd
import joblib
import wmi
from collections import deque
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "models", "cpu_xgboost_model.pkl")
SAMPLING_INTERVAL = 1.0  # seconds
IDLE_THRESHOLD = 5.0     # CPU % below which system is considered idle

class RealtimeTester:
    def __init__(self):
        print("Initializing Realtime Tester...")
        
        # Load the model
        try:
            self.model = joblib.load(MODEL_FILE)
            print(f"Loaded model from {MODEL_FILE}")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

        # Initialize WMI for temperature reading
        try:
            self.w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            self.ohm_available = True
        except Exception:
            self.ohm_available = False
            print("WARNING: OpenHardwareMonitor not detected. Actual temp might be unavailable.")

        # Rolling buffers
        self.cpu_util_10s = deque(maxlen=10)
        self.cpu_util_30s = deque(maxlen=30)
        self.cpu_util_60s = deque(maxlen=60)
        self.thermal_momentum_hist = deque(maxlen=10)
        
        self.prev_cpu_temp = None
        self.last_idle_time = time.time()
        
        # Initial sensor reading to prime buffers
        psutil.cpu_percent(interval=None)
        
    def get_real_cpu_temperature(self):
        if not self.ohm_available:
            return None
        try:
            for sensor in self.w.Sensor():
                if sensor.SensorType == u'Temperature' and "CPU Package" in sensor.Name:
                    return round(float(sensor.Value), 2)
        except Exception:
            pass
        return None

    def get_voltage_current(self, cpu_util):
        battery = psutil.sensors_battery()
        if battery:
            voltage = 11.1 if battery.power_plugged else 10.8
            current = 1.5 + (cpu_util / 100.0) * 3.5
        else:
            voltage = 12.0
            current = 2.0 + (cpu_util / 100.0) * 8.0
        return round(voltage, 2), round(current, 2)

    def run(self):
        try:
            duration_seconds = int(input("Enter test duration in seconds: "))
        except ValueError:
            print("Invalid input. Defaulting to 60 seconds.")
            duration_seconds = 60

        print(f"\nStarting Realtime Inference Loop for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early\n")
        print(f"{'Actual':<10} | {'Predicted':<10} | {'Diff':<10} | {'CPU%':<6} | {'Status':<10}")
        print("-" * 60)

        start_time = time.time()
        actuals = []
        predictions = []

        try:
            while True:
                current_time = time.time()
                if current_time - start_time >= duration_seconds:
                    break

                loop_start = current_time

                # --- 1. Collect Metrics ---
                cpu_util = psutil.cpu_percent(interval=None)
                mem_util = psutil.virtual_memory().percent
                
                freq = psutil.cpu_freq()
                clock_speed = freq.current if freq else 0
                
                # Temperature
                cpu_temp = self.get_real_cpu_temperature()
                
                # Temperature delta
                if self.prev_cpu_temp is not None and cpu_temp is not None:
                    cpu_temp_delta = round(cpu_temp - self.prev_cpu_temp, 2)
                else:
                    cpu_temp_delta = 0.0
                self.prev_cpu_temp = cpu_temp

                # Thermal momentum
                if cpu_temp is not None:
                    self.thermal_momentum_hist.append(cpu_temp)
                    weights = [i+1 for i in range(len(self.thermal_momentum_hist))]
                    thermal_momentum = round(
                        sum(t*w for t, w in zip(self.thermal_momentum_hist, weights)) / sum(weights), 2
                    )
                else:
                    thermal_momentum = 0.0

                # Power
                battery = psutil.sensors_battery()
                power_source = 1 if (not battery or battery.power_plugged) else 0
                voltage, current = self.get_voltage_current(cpu_util)
                power_estimated = round(voltage * current, 2)

                # Idle time
                if cpu_util < IDLE_THRESHOLD:
                    self.last_idle_time = time.time()
                time_since_idle = round(time.time() - self.last_idle_time, 2)

                # Load pattern
                if cpu_util < 10:
                    load_pattern = 0
                elif cpu_util < 30:
                    load_pattern = 1
                elif cpu_util < 60:
                    load_pattern = 2
                else:
                    load_pattern = 3

                num_processes = len(psutil.pids())

                # Rolling stats
                self.cpu_util_10s.append(cpu_util)
                self.cpu_util_30s.append(cpu_util)
                self.cpu_util_60s.append(cpu_util)

                # We need full buffers for accurate averages, but let's calculate with what we have
                avg_10s = round(sum(self.cpu_util_10s) / len(self.cpu_util_10s), 2)
                avg_30s = round(sum(self.cpu_util_30s) / len(self.cpu_util_30s), 2)
                avg_60s = round(sum(self.cpu_util_60s) / len(self.cpu_util_60s), 2)
                peak_10s = round(max(self.cpu_util_10s), 2)
                var_30s = round(
                    sum((x - avg_30s) ** 2 for x in self.cpu_util_30s) / len(self.cpu_util_30s), 2
                )

                # --- 2. Prepare Features for Model ---
                # Check train_xgboost.py for feature list/order.
                # Features used: all cols in CSV except EXCLUDE_FEATURES.
                # CSV Cols: 
                # timestamp, cpu_util, cpu_util_avg_10s, cpu_util_avg_30s, cpu_util_avg_60s, 
                # cpu_util_peak_10s, cpu_util_var_30s, mem_util, clock_speed, 
                # cpu_temp, cpu_temp_delta, thermal_momentum, ambient_temp, 
                # voltage, current, power_estimated, power_source, 
                # time_since_idle, load_pattern, num_processes, system_id
                
                # Excluded: timestamp, cpu_temp, ambient_temp, system_id, 
                # (voltage, current - EXCLUDED in training)
                
                # Expected Features dict
                features = {
                    'cpu_util': [cpu_util],
                    'cpu_util_avg_10s': [avg_10s],
                    'cpu_util_avg_30s': [avg_30s],
                    'cpu_util_avg_60s': [avg_60s],
                    'cpu_util_peak_10s': [peak_10s],
                    'cpu_util_var_30s': [var_30s],
                    'mem_util': [mem_util],
                    'clock_speed': [clock_speed],
                    'cpu_temp_delta': [cpu_temp_delta],
                    'thermal_momentum': [thermal_momentum],
                    'power_estimated': [power_estimated],
                    'power_source': [power_source],
                    'time_since_idle': [time_since_idle],
                    'load_pattern': [load_pattern],
                    'num_processes': [num_processes]
                }
                
                df_input = pd.DataFrame(features)
                
                # --- 3. Predict ---
                predicted_temp = self.model.predict(df_input)[0]
                
                # --- 4. Display ---
                actual_str = f"{cpu_temp:.1f}°C" if cpu_temp is not None else "N/A"
                pred_str = f"{predicted_temp:.1f}°C"
                
                diff_str = "N/A"
                if cpu_temp is not None:
                     diff = abs(cpu_temp - predicted_temp)
                     diff_str = f"{diff:.1f}°C"

                status = ""
                if len(self.cpu_util_60s) < 60:
                     status = "Warming up"
                
                print(f"{actual_str:<10} | {pred_str:<10} | {diff_str:<10} | {cpu_util:<6.1f} | {status}")

                # Collect data for final accuracy IF we have a valid actual temp
                if cpu_temp is not None:
                    actuals.append(cpu_temp)
                    predictions.append(predicted_temp)

                # Sleep
                time.sleep(max(0, SAMPLING_INTERVAL - (time.time() - loop_start)))

        except KeyboardInterrupt:
            print("\nStopped early.")
        
        # --- Final Accuracy Report ---
        print("\n" + "="*40)
        print("FINAL ACCURACY REPORT")
        print("="*40)
        
        if len(actuals) > 0:
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            max_error = np.max(np.abs(np.array(actuals) - np.array(predictions)))
            
            # Calculate accuracy percentage (1 - MAPE)
            # Avoid division by zero
            mape_values = []
            for a, p in zip(actuals, predictions):
                if a != 0:
                    mape_values.append(abs(a - p) / abs(a))
            
            mape = np.mean(mape_values) if mape_values else 0
            accuracy = 100 * (1 - mape)

            print(f"Duration:     {time.time() - start_time:.1f} seconds")
            print(f"Samples:      {len(actuals)}")
            print(f"MAE:          {mae:.2f}°C")
            print(f"RMSE:         {rmse:.2f}°C")
            print(f"Max Error:    {max_error:.2f}°C")
            print(f"Accuracy:     {accuracy:.2f}%")
        else:
            print("No valid temperature samples collected.")
        print("="*40 + "\n")

if __name__ == "__main__":
    RealtimeTester().run()
