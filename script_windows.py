import psutil
import time
import csv
from datetime import datetime
import wmi
from collections import deque

# --- CONFIGURATION ---
SYSTEM_ID = "S2"
OUTPUT_FILE = f"system_{SYSTEM_ID}_new.csv"
SAMPLING_INTERVAL = 1.0  # seconds
IDLE_THRESHOLD = 5.0     # CPU % below which system is considered idle
DURATION_MINUTES = 30    # Duration to run the script (in minutes), set to None for infinite

class SystemMonitor:
    def __init__(self):
        self.system_id = SYSTEM_ID
        self.output_file = OUTPUT_FILE
        
        # Initialize WMI
        try:
            self.w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            self.ohm_available = True
        except Exception:
            self.ohm_available = False
            print("WARNING: OpenHardwareMonitor not detected.")

        # Rolling CPU buffers
        self.cpu_util_10s = deque(maxlen=10)
        self.cpu_util_30s = deque(maxlen=30)
        self.cpu_util_60s = deque(maxlen=60)
        self.cpu_temp_hist = deque(maxlen=5)
        
        # Temperature tracking for new features
        self.prev_cpu_temp = None
        self.thermal_momentum_hist = deque(maxlen=10)  # For weighted average

        self.last_idle_time = time.time()

        psutil.cpu_percent(interval=None)

        freq = psutil.cpu_freq()
        self.clock_speed_max = freq.max if freq else 1

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
        print(f"Logging system data to {self.output_file}")
        if DURATION_MINUTES:
            print(f"Script will run for {DURATION_MINUTES} minutes")
        print("Press Ctrl+C to stop\n")

        start_time = time.time()
        end_time = start_time + (DURATION_MINUTES * 60) if DURATION_MINUTES else None

        with open(self.output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp", "cpu_util", "cpu_util_avg_10s", "cpu_util_avg_30s",
                "cpu_util_avg_60s", "cpu_util_peak_10s", "cpu_util_var_30s",
                "mem_util", "clock_speed",
                "cpu_temp", "cpu_temp_delta", "thermal_momentum", "ambient_temp",
                "voltage", "current", "power_estimated", "power_source",
                "time_since_idle", "load_pattern", "num_processes",
                "system_id"
            ])

            try:
                while True:
                    # Check if duration has elapsed
                    if end_time and time.time() >= end_time:
                        print("\n\nTimer expired. Data collection stopped.")
                        print(f"Saved to {self.output_file}")
                        break

                    loop_start = time.time()

                    # --- Basic metrics ---
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cpu_util = psutil.cpu_percent(interval=None)
                    mem_util = psutil.virtual_memory().percent

                    freq = psutil.cpu_freq()
                    clock_speed = freq.current if freq else 0

                    # --- Temperature ---
                    cpu_temp = self.get_real_cpu_temperature()

                    # NEW FEATURE: Temperature delta (rate of change)
                    if self.prev_cpu_temp is not None and cpu_temp is not None:
                        cpu_temp_delta = round(cpu_temp - self.prev_cpu_temp, 2)
                    else:
                        cpu_temp_delta = 0.0
                    self.prev_cpu_temp = cpu_temp

                    # NEW FEATURE: Thermal momentum (weighted average of recent temps)
                    if cpu_temp is not None:
                        self.thermal_momentum_hist.append(cpu_temp)
                        # Weighted average: recent temps have more weight
                        weights = [i+1 for i in range(len(self.thermal_momentum_hist))]
                        thermal_momentum = round(
                            sum(t*w for t, w in zip(self.thermal_momentum_hist, weights)) / sum(weights), 2
                        )
                    else:
                        thermal_momentum = 0.0

                    # Ambient temperature (data-center assumption)
                    ambient_temp = 22.0

                    # --- Power ---
                    battery = psutil.sensors_battery()
                    power_source = 1 if (not battery or battery.power_plugged) else 0
                    voltage, current = self.get_voltage_current(cpu_util)
                    power_estimated = round(voltage * current, 2)

                    # --- Idle tracking ---
                    if cpu_util < IDLE_THRESHOLD:
                        self.last_idle_time = time.time()
                    time_since_idle = round(time.time() - self.last_idle_time, 2)

                    # NEW FEATURE: Load pattern classification
                    if cpu_util < 10:
                        load_pattern = 0  # Idle
                    elif cpu_util < 30:
                        load_pattern = 1  # Light
                    elif cpu_util < 60:
                        load_pattern = 2  # Medium
                    else:
                        load_pattern = 3  # Heavy

                    num_processes = len(psutil.pids())

                    # --- Rolling CPU stats ---
                    self.cpu_util_10s.append(cpu_util)
                    self.cpu_util_30s.append(cpu_util)
                    self.cpu_util_60s.append(cpu_util)

                    avg_10s = round(sum(self.cpu_util_10s) / len(self.cpu_util_10s), 2)
                    avg_30s = round(sum(self.cpu_util_30s) / len(self.cpu_util_30s), 2)
                    avg_60s = round(sum(self.cpu_util_60s) / len(self.cpu_util_60s), 2)
                    peak_10s = round(max(self.cpu_util_10s), 2)
                    var_30s = round(
                        sum((x - avg_30s) ** 2 for x in self.cpu_util_30s) / len(self.cpu_util_30s), 2
                    )

                    # --- Write CSV ---
                    writer.writerow([
                        timestamp, cpu_util, avg_10s, avg_30s, avg_60s,
                        peak_10s, var_30s, mem_util,
                        clock_speed,
                        cpu_temp, cpu_temp_delta, thermal_momentum, ambient_temp,
                        voltage, current, power_estimated, power_source,
                        time_since_idle, load_pattern, num_processes,
                        self.system_id
                    ])

                    # Display with remaining time
                    elapsed = time.time() - start_time
                    if end_time:
                        remaining = end_time - time.time()
                        mins_remaining = int(remaining // 60)
                        secs_remaining = int(remaining % 60)
                        time_str = f" | Time Left: {mins_remaining:02d}:{secs_remaining:02d}"
                    else:
                        time_str = ""

                    print(
                        f"\rCPU: {cpu_util:5.1f}% | Temp: {cpu_temp if cpu_temp else 'N/A'}°C | Power: {power_estimated}W{time_str}",
                        end="", flush=True
                    )

                    time.sleep(max(0, SAMPLING_INTERVAL - (time.time() - loop_start)))

            except KeyboardInterrupt:
                print("\n\nData collection stopped.")
                print(f"Saved to {self.output_file}")

if __name__ == "__main__":
    SystemMonitor().run()
