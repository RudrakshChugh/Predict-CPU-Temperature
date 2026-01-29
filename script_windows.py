import psutil
import time
import csv
from datetime import datetime
import wmi

SYSTEM_ID = "S1"
OUTPUT_FILE = "system_S1.csv"
SAMPLING_INTERVAL = 1          # seconds
DURATION = 1800                # 30 minutes

# Initialize WMI for OpenHardwareMonitor
w = wmi.WMI(namespace="root\\OpenHardwareMonitor")


def get_real_cpu_temperature():
    """
    Reads REAL CPU Package temperature from OpenHardwareMonitor via WMI
    """
    for sensor in w.Sensor():
        if sensor.SensorType == u'Temperature' and "CPU Package" in sensor.Name:
            return round(sensor.Value, 2)
    return None


def get_voltage_current(cpu_util):
    """
    Approximate voltage and current (acceptable as input features)
    """
    battery = psutil.sensors_battery()
    if battery:
        voltage = 11.1 if battery.power_plugged else 10.8
        current = 1.5 + (cpu_util / 100.0) * 3.5
    else:
        voltage = 12.0
        current = 2.0 + (cpu_util / 100.0) * 8.0

    return round(voltage, 2), round(current, 2)


def get_ambient_temperature(cpu_temp):
    """
    Approximate ambient temperature (input feature)
    """
    if cpu_temp is not None:
        ambient = cpu_temp - 12
        return round(max(20, min(30, ambient)), 2)
    return 25.0


print(f"Initializing data collection for {SYSTEM_ID}...")
print("Using REAL CPU temperature via OpenHardwareMonitor (WMI)")
print("=" * 60)

# Test temperature once before starting
test_temp = get_real_cpu_temperature()
print(f"CPU Package temperature detected: {test_temp} °C")
print("=" * 60)


with open(OUTPUT_FILE, mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "timestamp",
        "cpu_util",
        "mem_util",
        "clock_speed",
        "cpu_temp",
        "ambient_temp",
        "voltage",
        "current",
        "system_id"
    ])

    start_time = time.time()
    print(f"Starting data collection for {SYSTEM_ID}...")
    print(f"Duration: {DURATION // 60} minutes")
    print("-" * 60)

    while time.time() - start_time < DURATION:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - start_time

        cpu_util = psutil.cpu_percent(interval=None)
        mem_util = psutil.virtual_memory().percent

        freq = psutil.cpu_freq()
        clock_speed = freq.current if freq else None

        cpu_temp = get_real_cpu_temperature()
        ambient_temp = get_ambient_temperature(cpu_temp)

        voltage, current = get_voltage_current(cpu_util)

        writer.writerow([
            timestamp,
            cpu_util,
            mem_util,
            clock_speed,
            cpu_temp,
            ambient_temp,
            voltage,
            current,
            SYSTEM_ID
        ])

        progress = (elapsed / DURATION) * 100

        print(
            f"\rProgress: {progress:5.1f}% | "
            f"CPU: {cpu_util:5.1f}% | "
            f"Temp: {cpu_temp:5.1f}°C",
            end="",
            flush=True
        )

        time.sleep(SAMPLING_INTERVAL)

print("\n" + "-" * 60)
print(f"Data collection completed. Saved to {OUTPUT_FILE}")
