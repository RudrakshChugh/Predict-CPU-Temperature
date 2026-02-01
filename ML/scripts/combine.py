import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

files = []
# Check for S1 to S5
for i in range(1, 6):
    fname = f"system_S{i}.csv"
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        files.append(fpath)
    else:
        print(f"Warning: {fname} not found, skipping.")

if not files:
    print("No data files found to combine!")
    exit(1)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(os.path.join(DATA_DIR, "combined_system_data.csv"), index=False)
print(f"Combined {len(files)} files into combined_system_data.csv")
