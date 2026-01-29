import pandas as pd

files = [
    "system_S1.csv",
    "system_S2.csv",
    "system_S3.csv",
    "system_S4.csv",
    "system_S5.csv"
]

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv("combined_system_data.csv", index=False)
