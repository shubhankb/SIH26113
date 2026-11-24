import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

PHASE2_CSV = "phase2_kinematic_features.csv"

ALL_NUMERIC_FEATURES = [
    "n_frames",
    "spatial_rom_3d",
    "mean_speed_3d",
    "p95_speed_3d",
    "mean_accel_3d",
    "mean_jerk_3d",
    "norm_variability",
    "bilateral_motion_corr",
]

# Default pruned feature set dropping near-duplicate mean_accel_3d (r = 0.997 with mean_jerk_3d)
DEFAULT_PRUNED_FEATURES = [
    "n_frames",
    "spatial_rom_3d",
    "mean_speed_3d",
    "p95_speed_3d",
    "mean_jerk_3d",
    "norm_variability",
    "bilateral_motion_corr",
]


def select_non_redundant_features(
    df_num: pd.DataFrame,
    threshold: float = 0.95,
    preferred_order: list = None,
):
    cols = preferred_order if preferred_order else list(df_num.columns)
    cols = [c for c in cols if c in df_num.columns]
    corr = df_num[cols].corr().abs()

    kept = []
    dropped = []
    for col in cols:
        is_redundant = False
        for k_col in kept:
            if corr.loc[col, k_col] >= threshold:
                dropped.append((col, k_col, round(float(corr.loc[col, k_col]), 4)))
                is_redundant = True
                break
        if not is_redundant:
            kept.append(col)
    return kept, dropped


def run_phase3_preprocessing(
    input_csv: str = PHASE2_CSV,
    feature_cols: list = None,
    scaler_type: str = "robust",
    prune_collinear: bool = True,
    collinear_threshold: float = 0.95,
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"'{input_csv}' nahi mila. Pehle Phase 2 run karo.")

    df = pd.read_csv(input_csv)
    base_cols = feature_cols if feature_cols else [c for c in ALL_NUMERIC_FEATURES if c in df.columns]
    if not base_cols:
        raise ValueError("Koi valid numeric feature column nahi mila.")

    # 1. Impute missing/inf values using median
    X_raw = df[base_cols].replace([np.inf, -np.inf], np.nan).astype(float)
    nan_before = int(X_raw.isna().sum().sum())
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=base_cols, index=df.index)
    nan_after = int(X_imputed.isna().sum().sum())

    # 2. Collinearity check & optional pruning
    if prune_collinear:
        active_cols, dropped_pairs = select_non_redundant_features(
            X_imputed, threshold=collinear_threshold, preferred_order=DEFAULT_PRUNED_FEATURES
        )
    else:
        active_cols = list(base_cols)
        dropped_pairs = []

    X_selected = X_imputed[active_cols].copy()

    # 3. Compare StandardScaler vs RobustScaler on active features
    X_std = pd.DataFrame(
        StandardScaler().fit_transform(X_selected),
        columns=active_cols,
        index=df.index,
    )
    X_rob = pd.DataFrame(
        RobustScaler().fit_transform(X_selected),
        columns=active_cols,
        index=df.index,
    )

    comparison_df = pd.DataFrame({
        "Feature": active_cols,
        "StdScaler_MaxAbs": np.round(np.max(np.abs(X_std.values), axis=0), 2),
        "RobustScaler_MaxAbs": np.round(np.max(np.abs(X_rob.values), axis=0), 2),
        "RobustScaler_P95Abs": np.round(np.percentile(np.abs(X_rob.values), 95, axis=0), 2),
    })

    if scaler_type.lower() == "robust":
        X_scaled = X_rob
        fitted_scaler = RobustScaler().fit(X_selected)
    elif scaler_type.lower() == "standard":
        X_scaled = X_std
        fitted_scaler = StandardScaler().fit(X_selected)
    else:
        raise ValueError("scaler_type must be 'robust' or 'standard'.")

    return {
        "rep_ids": df["Rep_ID"] if "Rep_ID" in df.columns else pd.Series([f"REP_{i+1:04d}" for i in range(len(df))]),
        "active_features": active_cols,
        "dropped_collinear_pairs": dropped_pairs,
        "X_imputed": X_imputed,
        "X_selected": X_selected,
        "X_scaled": X_scaled,
        "X_std": X_std,
        "X_rob": X_rob,
        "scaler": fitted_scaler,
        "scaler_comparison": comparison_df,
        "nan_before": nan_before,
        "nan_after": nan_after,
    }


if __name__ == "__main__":
    out = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    print("=== PHASE 3 PREPROCESSING COMPLETE ===")
    print("Active features:", out["active_features"])
    print("Dropped collinear pairs (|r| >= 0.95):", out["dropped_collinear_pairs"])
    print("\n--- Scaler Comparison ---")
    print(out["scaler_comparison"].to_string(index=False))