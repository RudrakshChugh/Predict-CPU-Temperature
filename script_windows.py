import psutil
import time
import csv
import sys
from datetime import datetime
import wmi
from collections import deque

# --- CONFIGURATION ---
SYSTEM_ID = "S1"
OUTPUT_FILE = f"system_{SYSTEM_ID}.csv"
SAMPLING_INTERVAL = 1.0  # seconds
DURATION = 1800           # 30 minutes
IDLE_THRESHOLD = 5.0     # CPU % below which system is considered idle

class SystemMonitor:
    def __init__(self):
        self.system_id = SYSTEM_ID
        self.output_file = OUTPUT_FILE
        
        # Initialize WMI safely
        try:
            self.w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            self.ohm_available = True
        except Exception:
            self.ohm_available = False
            print("WARNING: OpenHardwareMonitor not detected via WMI. Temperature will be None.")

        # Buffers for rolling metrics
        self.cpu_util_10s = deque(maxlen=10)
        self.cpu_util_30s = deque(maxlen=30)
        self.cpu_util_60s = deque(maxlen=60)
        self.cpu_temp_hist = deque(maxlen=5)

        self.last_idle_time = time.time()
        
        # Seed psutil cpu calculation
        psutil.cpu_percent(interval=None)
        
        freq = psutil.cpu_freq()
        self.clock_speed_max = freq.max if freq else 0

    def get_real_cpu_temperature(self):
        """Fetches CPU Package temperature from OpenHardwareMonitor."""
        if not self.ohm_available:
            return None
        try:
            sensors = self.w.Sensor()
            for sensor in sensors:
                if sensor.SensorType == u'Temperature' and "CPU Package" in sensor.Name:
                    return round(float(sensor.Value), 2)
        except Exception:
            return None
        return None

    def get_voltage_current(self, cpu_util):
        """
        Estimates voltage and current. 
        Note: These are heuristic models based on load.
        """
        battery = psutil.sensors_battery()
        if battery:
            # Laptop Logic
            voltage = 11.1 if battery.power_plugged else 10.8
            current = 1.5 + (cpu_util / 100.0) * 3.5
        else:
            # Desktop Logic
            voltage = 12.0
            current = 2.0 + (cpu_util / 100.0) * 8.0
        return round(voltage, 2), round(current, 2)

    def run(self):
        print(f"Initializing data collection for {self.system_id}...")
        print(f"Logging to: {self.output_file}")
        print("Press Ctrl+C to stop manually.")
        print("=" * 60)

        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "cpu_util", "cpu_util_avg_10s", "cpu_util_avg_30s",
                "cpu_util_avg_60s", "cpu_util_peak_10s", "cpu_util_var_30s",
                "mem_util", "clock_speed", "clock_speed_max", "cpu_temp",
                "cpu_temp_prev_1s", "cpu_temp_prev_5s", "ambient_temp",
                "voltage", "current", "power_estimated", "power_source",
                "time_since_idle", "system_id"
            ])

            start_time = time.time()
            try:
                while (time.time() - start_time) < DURATION:
                    loop_start = time.time()
                    
                    # 1. Basic Metrics
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cpu_util = psutil.cpu_percent(interval=None)
                    mem_util = psutil.virtual_memory().percent
                    
                    freq = psutil.cpu_freq()
                    clock_speed = freq.current if freq else 0
                    
                    # 2. Temperature Logic
                    cpu_temp = self.get_real_cpu_temperature()
                    ambient_temp = round(cpu_temp - 12.0, 2) if cpu_temp else 25.0
                    
                    # 3. Power Logic
                    battery = psutil.sensors_battery()
                    power_source = 1 if (not battery or battery.power_plugged) else 0
                    voltage, current = self.get_voltage_current(cpu_util)
                    power_estimated = round(voltage * current, 2)

                    # 4. Idle Tracking
                    if cpu_util < IDLE_THRESHOLD:
                        self.last_idle_time = time.time()
                    time_since_idle = round(time.time() - self.last_idle_time, 2)

                    # 5. Rolling Statistics
                    self.cpu_util_10s.append(cpu_util)
                    self.cpu_util_30s.append(cpu_util)
                    self.cpu_util_60s.append(cpu_util)

                    avg_10s = round(sum(self.cpu_util_10s) / len(self.cpu_util_10s), 2)
                    avg_30s = round(sum(self.cpu_util_30s) / len(self.cpu_util_30s), 2)
                    avg_60s = round(sum(self.cpu_util_60s) / len(self.cpu_util_60s), 2)
                    peak_10s = round(max(self.cpu_util_10s), 2)
                    var_30s = round(sum((x - avg_30s)**2 for x in self.cpu_util_30s) / len(self.cpu_util_30s), 2)

                    # Temperature history
                    temp_prev_1s = self.cpu_temp_hist[-1] if len(self.cpu_temp_hist) >= 1 else None
                    temp_prev_5s = self.cpu_temp_hist[0] if len(self.cpu_temp_hist) == 5 else None
                    if cpu_temp is not None:
                        self.cpu_temp_hist.append(cpu_temp)

                    # 6. Write to CSV
                    writer.writerow([
                        timestamp, cpu_util, avg_10s, avg_30s, avg_60s, peak_10s, var_30s,
                        mem_util, clock_speed, self.clock_speed_max, cpu_temp,
                        temp_prev_1s, temp_prev_5s, ambient_temp,
                        voltage, current, power_estimated, power_source,
                        time_since_idle, self.system_id
                    ])

                    # Console Output
                    temp_str = f"{cpu_temp}°C" if cpu_temp else "N/A"
                    print(f"\r[{self.system_id}] CPU: {cpu_util:5.1f}% | Temp: {temp_str:>7} | IdleΔ: {time_since_idle:6.1f}s", end="", flush=True)

                    # 7. Precise Timing
                    # Subtract the time taken to process logic from the sleep interval
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, SAMPLING_INTERVAL - elapsed)
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n\nCollection interrupted by user.")
            finally:
                print(f"\nData saved to {self.output_file}")
                print("Exiting safely.")

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.run()