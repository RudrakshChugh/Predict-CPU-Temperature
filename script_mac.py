import psutil
import time
import csv
from datetime import datetime
import subprocess
import platform

SYSTEM_ID = "S1"
OUTPUT_FILE = "system_S1.csv"
SAMPLING_INTERVAL = 1          # seconds
DURATION = 1800                # 30 minutes


def get_cpu_temperature_mac():
    """
    Get CPU temperature on macOS using multiple methods
    """
    # Method 1: Try psutil sensors (works on some Macs)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current is not None:
                        return round(entry.current, 2)
    except:
        pass
    
    # Method 2: Try osx-cpu-temp command (if installed)
    try:
        result = subprocess.run(
            ['osx-cpu-temp'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Output format: "61.2°C"
            temp_str = result.stdout.strip()
            temp_value = float(temp_str.replace('°C', '').strip())
            return round(temp_value, 2)
    except:
        pass
    
    # Method 3: Try powermetrics (requires sudo, so likely won't work)
    try:
        result = subprocess.run(
            ['powermetrics', '--samplers', 'smc', '-i1', '-n1'],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            # Parse output for CPU temperature
            for line in result.stdout.split('\n'):
                if 'CPU die temperature' in line:
                    temp_value = float(line.split(':')[1].strip().split()[0])
                    return round(temp_value, 2)
    except:
        pass
    
    # Method 4: Try istats (if installed via gem install iStats)
    try:
        result = subprocess.run(
            ['istats', 'cpu', 'temp', '--value-only'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            temp_value = float(result.stdout.strip())
            return round(temp_value, 2)
    except:
        pass
    
    return None


def get_voltage_current(cpu_util):
    """
    Estimate voltage and current on macOS
    """
    battery = psutil.sensors_battery()
    if battery:
        # MacBook battery voltage
        voltage = 11.4 if battery.power_plugged else 11.1
        current = 1.5 + (cpu_util / 100.0) * 3.5
    else:
        # Mac desktop (iMac, Mac Mini, Mac Studio)
        voltage = 12.0
        current = 2.0 + (cpu_util / 100.0) * 8.0
    
    return round(voltage, 2), round(current, 2)


def get_ambient_temperature(cpu_temp):
    """
    Estimate ambient temperature
    """
    if cpu_temp is not None:
        ambient = cpu_temp - 12
        return round(max(20, min(30, ambient)), 2)
    return 25.0


# Verify we're on macOS
if platform.system() != 'Darwin':
    print("ERROR: This script is for macOS only!")
    print("For Windows, use script_windows.py")
    exit(1)

print(f"Initializing data collection for {SYSTEM_ID}...")
print("Detecting CPU temperature method for macOS...")
print("=" * 60)

# Test temperature reading
test_temp = get_cpu_temperature_mac()

if test_temp is None:
    print("⚠️  WARNING: Could not detect CPU temperature!")
    print("")
    print("To enable temperature reading on macOS, install one of these:")
    print("")
    print("Option 1: osx-cpu-temp (Recommended)")
    print("  brew install osx-cpu-temp")
    print("")
    print("Option 2: iStats")
    print("  sudo gem install iStats")
    print("")
    print("After installation, run this script again.")
    print("=" * 60)
    
    response = input("\nContinue WITHOUT temperature data? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Please install temperature monitoring tool first.")
        exit(1)
    print("\n⚠️  Proceeding without CPU temperature (will be None in CSV)")
else:
    print(f"✓ CPU Temperature detected: {test_temp}°C")

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

        cpu_temp = get_cpu_temperature_mac()
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

        temp_display = f"{cpu_temp:5.1f}°C" if cpu_temp else "  N/A  "
        
        print(
            f"\rProgress: {progress:5.1f}% | "
            f"CPU: {cpu_util:5.1f}% | "
            f"Temp: {temp_display}",
            end="",
            flush=True
        )

        time.sleep(SAMPLING_INTERVAL)

print("\n" + "-" * 60)
print(f"Data collection completed. Saved to {OUTPUT_FILE}")
