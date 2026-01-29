# CPU Temperature Prediction - Data Collection

Collects system metrics to train ML model for CPU temperature prediction.

## 🚀 Quick Start

### Prerequisites
- **Windows 10/11** (OpenHardwareMonitor required)
- **Python 3.7+**
- **OpenHardwareMonitor** - Download from https://openhardwaremonitor.org/

### Setup on Each System

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd AI-project
   ```

2. **Install OpenHardwareMonitor:**
   - Download and extract
   - Run `OpenHardwareMonitor.exe` as Administrator
   - Keep it running in background

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Edit `script_windows.py` for each system:**
   ```python
   # Change these two lines:
   SYSTEM_ID = "S1"              # Change to S2, S3, S4, S5 for other systems
   OUTPUT_FILE = "system_S1.csv" # Change to system_S2.csv, etc.
   ```

5. **Run data collection (30 minutes):**
   ```bash
   python script_windows.py
   ```

6. **After all systems complete, combine data:**
   ```bash
   python combine.py
   ```

## 📊 Data Collected

- CPU utilization
- Memory usage
- Clock speed
- **CPU temperature** (real sensor data)
- Ambient temperature
- Voltage & current
- System ID

## 🔒 Safety

✅ **100% Read-only** - No system modifications  
✅ Only monitors system metrics  
✅ Only writes to CSV files  

## 📁 Files

- `script_windows.py` - Main data collection script
- `combine.py` - Combines CSV files from all systems
- `requirements.txt` - Python dependencies

## 📝 Output

Each system generates: `system_S1.csv`, `system_S2.csv`, etc.  
Combined output: `combined_system_data.csv`

---

**Duration:** 30 minutes per system  
**Platform:** Windows only (requires OpenHardwareMonitor)
