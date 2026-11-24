import numpy as np
import pandas as pd
from phase3_preprocessing import run_phase3_preprocessing
from phase6_evaluation import run_phase6_evaluation

PROFILE_FEATURES = [
    "n_frames",
    "spatial_rom_3d",
    "mean_speed_3d",
    "p95_speed_3d",
    "mean_jerk_3d",
    "norm_variability",
    "bilateral_motion_corr",
]


def _describe_cluster_from_stats(
    cluster_med: pd.Series,
    pop_med: pd.Series,
    pop_iqr: pd.Series,
) -> str:
    """Generate data-driven description strictly from computed cluster vs population medians."""
    traits = []

    def ratio_tag(col: str, high_label: str, low_label: str, thresh: float = 0.25):
        if col not in cluster_med or col not in pop_med:
            return
        denom = pop_iqr.get(col, np.nan)
        if pd.isna(denom) or denom < 1e-12:
            denom = abs(pop_med[col]) if abs(pop_med[col]) > 1e-12 else 1.0
        delta = (cluster_med[col] - pop_med[col]) / denom
        if delta >= thresh:
            traits.append(high_label)
        elif delta <= -thresh:
            traits.append(low_label)

    ratio_tag("spatial_rom_3d", "wider 3D range of motion", "compact 3D range of motion")
    ratio_tag("mean_speed_3d", "faster mean speed", "slower mean speed")
    ratio_tag("mean_jerk_3d", "higher jerk / less smooth", "lower jerk / smoother")
    ratio_tag("norm_variability", "higher norm variability", "lower norm variability")
    ratio_tag("n_frames", "longer duration (more frames)", "shorter duration (fewer frames)")
    ratio_tag("bilateral_motion_corr", "stronger left-right correlation", "weaker/asymmetric left-right correlation")

    if not traits:
        return "Near-median baseline kinematics across measured features"
    return ", ".join(traits).capitalize()


def run_phase8_profiling(
    df_features: pd.DataFrame,
    labels: np.ndarray,
    profile_cols: list = None,
    include_noise: bool = False,
):
    cols = profile_cols if profile_cols else [c for c in PROFILE_FEATURES if c in df_features.columns]
    work = df_features[cols].copy()
    work["_cluster"] = labels

    if not include_noise:
        work = work[work["_cluster"] != -1].copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pop_med = work[cols].median()
    pop_iqr = (work[cols].quantile(0.75) - work[cols].quantile(0.25)).replace(0, np.nan)

    mean_df = work.groupby("_cluster")[cols].mean().round(5)
    median_df = work.groupby("_cluster")[cols].median().round(5)
    std_df = work.groupby("_cluster")[cols].std().round(5)
    counts = work.groupby("_cluster").size()

    summary_rows = []
    for cl_id in sorted(work["_cluster"].unique()):
        cl_med_row = work[work["_cluster"] == cl_id][cols].median()
        desc = _describe_cluster_from_stats(cl_med_row, pop_med, pop_iqr)
        summary_rows.append({
            "Cluster": int(cl_id),
            "Count": int(counts.loc[cl_id]),
            "Share_%": round(float(counts.loc[cl_id] / len(work) * 100.0), 2),
            "Observed_Characteristics": desc,
        })

    return pd.DataFrame(summary_rows), mean_df, median_df, std_df


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    _, fitted_labels, tuned = run_phase6_evaluation(prep["X_scaled"].values, random_state=42)

    # Profile K-Means clusters as an example
    km_lbl = fitted_labels["kmeans"]
    prof_summary, km_mean, km_med, _ = run_phase8_profiling(prep["X_selected"], km_lbl)
    print(f"=== PHASE 8 CLUSTER PROFILES (K-Means k={tuned['kmeans']['n_clusters']}) ===")
    print(prof_summary.to_string(index=False))
    print("\n--- Cluster Feature Medians ---")
    print(km_med.to_string())