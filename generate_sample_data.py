# ════════════════════════════════════════════════════════════════════════════
#  RLIAS — Sample Data Generator
#  Run this to generate a realistic rocker lever inspection dataset
#  Usage: python generate_sample_data.py
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

N = 500  # number of inspection records

# Simulate realistic rocker lever dimensions and quality attributes
dates = [datetime(2024, 1, 2) + timedelta(hours=i*0.5) for i in range(N)]
shifts = ["Morning", "Afternoon", "Night"]
operators = ["OP001", "OP002", "OP003", "OP004"]
machines = ["MC-01", "MC-02", "MC-03"]
materials = ["Forging_LotA", "Forging_LotB", "Forging_LotC"]

data = {
    "Inspection_DateTime":  dates,
    "Part_ID":              [f"RL-{2024000+i}" for i in range(N)],
    "Shift":                [random.choice(shifts) for _ in range(N)],
    "Operator_ID":          [random.choice(operators) for _ in range(N)],
    "Machine_ID":           [random.choice(machines) for _ in range(N)],
    "Material_Lot":         [random.choice(materials) for _ in range(N)],

    # ── Dimensional Parameters (mm) ──────────────────────────────────────
    # Nominal: 45.00 mm ±0.05
    "Rocker_Arm_Length":    np.random.normal(45.00, 0.018, N) + np.sin(np.arange(N)*0.05)*0.01,
    # Nominal: 12.00 mm ±0.03
    "Pivot_Bore_Diameter":  np.random.normal(12.00, 0.008, N),
    # Nominal: 8.50 mm ±0.025
    "Valve_Tip_Radius":     np.random.normal(8.50, 0.009, N),
    # Nominal: 6.00 mm ±0.02
    "Push_Rod_Socket_Depth":np.random.normal(6.00, 0.007, N),
    # Nominal: 3.20 mm ±0.015
    "Lash_Pad_Thickness":   np.random.normal(3.20, 0.005, N),

    # ── Surface & Form ───────────────────────────────────────────────────
    # Surface roughness Ra µm
    "Surface_Roughness_Ra": np.random.normal(0.80, 0.12, N),
    # Roundness µm
    "Bore_Roundness":       np.abs(np.random.normal(0.0, 3.5, N)),
    # Runout µm
    "Pivot_Runout":         np.abs(np.random.normal(0.0, 4.0, N)),

    # ── Mechanical ───────────────────────────────────────────────────────
    # Hardness HRC
    "Hardness_HRC":         np.random.normal(58.0, 1.2, N),
    # Weight grams
    "Weight_grams":         np.random.normal(185.0, 1.5, N),

    # ── Inspection outcome ───────────────────────────────────────────────
    "Visual_Defects":       np.random.choice(["None", "Scratch", "Burr", "Pit"], N,
                                              p=[0.88, 0.06, 0.04, 0.02]),
}

df = pd.DataFrame(data)

# ── Inject deliberate anomalies / process shift ──────────────────────────
# Machine MC-03 has tool wear after record 300 → length drift
mc03_idx = df[df["Machine_ID"] == "MC-03"].index.tolist()
for i, idx in enumerate(mc03_idx):
    if idx > 300:
        df.loc[idx, "Rocker_Arm_Length"] += 0.002 * (idx - 300) / N

# Night shift hardness dip
night_idx = df[df["Shift"] == "Night"].index.tolist()
df.loc[night_idx, "Hardness_HRC"] -= np.random.uniform(0.5, 1.5, len(night_idx))

# Inject 15 clear outliers
outlier_idx = np.random.choice(N, 15, replace=False)
df.loc[outlier_idx, "Rocker_Arm_Length"] += np.random.choice([-0.15, 0.15], 15)
df.loc[outlier_idx, "Bore_Roundness"]     += np.random.uniform(15, 30, 15)

# Pass/Fail judgment
spec = {
    "Rocker_Arm_Length":    (44.95, 45.05),
    "Pivot_Bore_Diameter":  (11.97, 12.03),
    "Hardness_HRC":         (55.0, 62.0),
    "Surface_Roughness_Ra": (0.4, 1.2),
}
def judge(row):
    for col, (lo, hi) in spec.items():
        if col in row and (row[col] < lo or row[col] > hi):
            return "REJECT"
    return "PASS"

df["Inspection_Result"] = df.apply(judge, axis=1)

df.to_csv("rocker_lever_inspection_sample.csv", index=False)
print(f"✅  Generated: rocker_lever_inspection_sample.csv  ({N} records)")
print(f"   Reject count: {(df['Inspection_Result']=='REJECT').sum()}")
print(f"   Pass count:   {(df['Inspection_Result']=='PASS').sum()}")
print(f"\n   Columns: {list(df.columns)}")