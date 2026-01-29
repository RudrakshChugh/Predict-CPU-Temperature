# CPU Temperature Prediction - Data Collection

Collects system metrics to train ML model for CPU temperature prediction.

##  Quick Start

### Prerequisites
- **Windows 10/11** OR **macOS**
- **Python 3.7+**
- **Temperature monitoring tool** (platform-specific, see below)

### Platform-Specific Setup

####  **Windows Users**

1. **Install OpenHardwareMonitor:**
   - Download from https://openhardwaremonitor.org/
   - Extract and run `OpenHardwareMonitor.exe` as Administrator
   - Keep it running in background

2. **Use:** `script_windows.py`

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

2. **Use:** `script_mac.py`

### Setup on Each System

1. **Clone this repository:**
   ```bash
   git clone https://github.com/RudrakshChugh/Predict-CPU-Temperature
   ```

2. **Install OpenHardwareMonitor:**
   - Download and extract
   - Run `OpenHardwareMonitor.exe` as Administrator
   - Keep it running in background

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Choose the correct script for your platform:**
   - **Windows:** Edit `script_windows.py`
   - **macOS:** Edit `script_mac.py`
   
   Change these two lines:
   ```python
   SYSTEM_ID = "S1"              # Change to S2, S3, S4, S5 for other systems
   OUTPUT_FILE = "system_S1.csv" # Change to system_S2.csv, etc.
   ```

5. **Run data collection (30 minutes):**
   ```bash
   # Windows:
   python script_windows.py
   
   # macOS:
   python script_mac.py
   ```

6. **After all systems complete, combine data:**
   ```bash
   python combine.py
   ```

##  Data Collected

- CPU utilization
- Memory usage
- Clock speed
- **CPU temperature** (real sensor data)
- Ambient temperature
- Voltage & current
- System ID

##  Safety

 **100% Read-only** - No system modifications  
 Only monitors system metrics  
 Only writes to CSV files  

##  Files

- `script_windows.py` - Data collection script for Windows
- `script_mac.py` - Data collection script for macOS
- `combine.py` - Combines CSV files from all systems
- `requirements.txt` - Python dependencies

## Output

Each system generates: `system_S1.csv`, `system_S2.csv`, etc.  
Combined output: `combined_system_data.csv`

---

**Duration:** 30 minutes per system  
**Platform:** Windows & macOS supported
