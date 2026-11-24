import os
import numpy as np
import pandas as pd

CSV_PATH = "intellirehab_unsupervised_features.csv"

NUMERIC_COLS = [
    "Range_of_Motion_deg",
    "Mean_Angular_Vel",
    "Max_Angular_Vel",
    "Movement_Jerk",
    "Joint_Symmetry_Index",
    "Rep_Variability",
]


def audit_features_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' nahi mila. Pehle feature extraction run karo.")

    df = pd.read_csv(csv_path)
    print("=== 1. DATASET STRUCTURE ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    print("\nData Types:\n", df.dtypes)

    print("\n=== 2. MISSING VALUES & DUPLICATES ===")
    print("Null count per column:\n", df.isna().sum())
    dup_feature_rows = df.duplicated(subset=[c for c in NUMERIC_COLS if c in df.columns]).sum()
    dup_rep_ids = df["Rep_ID"].duplicated().sum() if "Rep_ID" in df.columns else 0
    print(f"Duplicate feature rows: {dup_feature_rows} | Duplicate Rep_IDs: {dup_rep_ids}")

    present_num = [c for c in NUMERIC_COLS if c in df.columns]
    print("\n=== 3. FEATURE SUMMARY STATS ===")
    stats = df[present_num].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    print(stats[["count", "mean", "std", "min", "5%", "50%", "95%", "max"]].round(5))

    print("\n=== 4. IQR OUTLIER SCAN (AUDIT ONLY, NOT REMOVED) ===")
    for col in present_num:
        s = df[col].dropna()
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < low) | (s > high)).sum())
        print(f"{col:24s} -> IQR bounds [{low:.4f}, {high:.4f}] | Outliers: {n_out} ({n_out/len(s)*100:.1f}%)")

    print("\n=== 5. NAMING / UNIT AUDIT NOTE ===")
    print(
        "SkeletonUtils.py extracts 3D joint positions (x, y, z) for 25 joints—not joint angles in degrees.\n"
        "In Phase 2, we will rename/recompute these into accurate 3D kinematic features."
    )
    return df


if __name__ == "__main__":
    audit_features_csv()