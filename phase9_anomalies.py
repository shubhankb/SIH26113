import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from phase3_preprocessing import run_phase3_preprocessing
from phase6_evaluation import run_phase6_evaluation


def _top_feature_reasons(row_z: np.ndarray, feat_names: list, top_k: int = 2) -> str:
    """Explain which scaled features drive a point's deviation using |z| magnitude."""
    abs_z = np.abs(row_z)
    order = np.argsort(abs_z)[::-1][:top_k]
    parts = []
    for idx in order:
        sign = "+" if row_z[idx] >= 0 else "-"
        parts.append(f"{feat_names[idx]} ({sign}{abs_z[idx]:.1f}σ)")
    return ", ".join(parts)


def run_phase9_anomaly_detection(
    X_scaled: np.ndarray,
    feature_names: list,
    rep_ids: pd.Series = None,
    cluster_labels: np.ndarray = None,
    contamination: float = 0.07,
    lof_neighbors: int = 20,
    random_state: int = 42,
):
    n_samples, n_features = X_scaled.shape
    if n_samples < 6:
        raise ValueError("Anomaly detection ke liye kam se kam 6 samples chahiye.")

    ids = (
        rep_ids.reset_index(drop=True).astype(str)
        if rep_ids is not None
        else pd.Series([f"REP_{i+1:04d}" for i in range(n_samples)])
    )
    cl_lbl = (
        np.asarray(cluster_labels)
        if cluster_labels is not None
        else np.zeros(n_samples, dtype=int)
    )

    # 1. Isolation Forest (higher score => more anomalous)
    iso = IsolationForest(
        contamination=float(contamination),
        n_estimators=200,
        random_state=random_state,
    )
    iso_pred = iso.fit_predict(X_scaled)  # -1 = outlier, 1 = inlier
    iso_anom_score = -iso.score_samples(X_scaled)

    # 2. Local Outlier Factor (higher score => more anomalous)
    k_lof = min(int(lof_neighbors), max(2, n_samples - 1))
    lof = LocalOutlierFactor(n_neighbors=k_lof, contamination=float(contamination))
    lof_pred = lof.fit_predict(X_scaled)  # -1 = outlier, 1 = inlier
    lof_anom_score = -lof.negative_outlier_factor_

    # Normalize both continuous scores to [0, 1] for combined severity ranking
    def _minmax_01(arr: np.ndarray) -> np.ndarray:
        span = float(np.max(arr) - np.min(arr))
        if span < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - np.min(arr)) / span

    iso_norm = _minmax_01(iso_anom_score)
    lof_norm = _minmax_01(lof_anom_score)
    combined_severity = 0.5 * iso_norm + 0.5 * lof_norm

    # 3. Cluster-relative z-deviation and top feature explainability
    cluster_z = np.zeros_like(X_scaled, dtype=float)
    for c_val in np.unique(cl_lbl):
        mask = cl_lbl == c_val
        sub = X_scaled[mask]
        c_mean = np.mean(sub, axis=0)
        c_std = np.std(sub, axis=0)
        c_std_safe = np.where(c_std < 1e-8, 1.0, c_std)
        cluster_z[mask] = (sub - c_mean) / c_std_safe

    cluster_rel_dist = np.linalg.norm(cluster_z, axis=1)
    reasons = [_top_feature_reasons(cluster_z[i], feature_names, top_k=2) for i in range(n_samples)]

    both_flagged = (iso_pred == -1) & (lof_pred == -1)
    any_flagged = (iso_pred == -1) | (lof_pred == -1)

    status = np.where(
        both_flagged,
        "Consensus Anomaly (IF + LOF)",
        np.where(
            iso_pred == -1,
            "Flagged (Isolation Forest)",
            np.where(lof_pred == -1, "Flagged (LOF)", "Normal Pattern"),
        ),
    )

    df_out = pd.DataFrame({
        "Rep_ID": ids,
        "Cluster": cl_lbl,
        "Status": status,
        "Any_Flagged": any_flagged,
        "IF_Flag": iso_pred == -1,
        "LOF_Flag": lof_pred == -1,
        "Severity_Score": np.round(combined_severity, 4),
        "Cluster_Rel_Z_Norm": np.round(cluster_rel_dist, 3),
        "Top_Driving_Features": reasons,
    })

    df_out["Severity_Rank"] = (
        df_out["Severity_Score"].rank(ascending=False, method="min").astype(int)
    )

    flagged_sorted = (
        df_out[df_out["Any_Flagged"]]
        .sort_values(by=["Severity_Score", "Cluster_Rel_Z_Norm"], ascending=[False, False])
        .reset_index(drop=True)
    )

    summary = {
        "total_samples": int(n_samples),
        "if_flagged": int(np.sum(iso_pred == -1)),
        "lof_flagged": int(np.sum(lof_pred == -1)),
        "consensus_flagged": int(np.sum(both_flagged)),
        "any_flagged": int(np.sum(any_flagged)),
    }

    return df_out, flagged_sorted, summary


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    _, fitted_labels, _ = run_phase6_evaluation(prep["X_scaled"].values, random_state=42)

    anom_all, top_flagged, summ = run_phase9_anomaly_detection(
        X_scaled=prep["X_scaled"].values,
        feature_names=prep["active_features"],
        rep_ids=prep["rep_ids"],
        cluster_labels=fitted_labels["kmeans"],
        contamination=0.07,
        lof_neighbors=20,
        random_state=42,
    )

    print("=== PHASE 9 ANOMALY DETECTION SUMMARY ===")
    print(summ)
    print("\n--- Top 10 Most Severe Flagged Repetitions ---")
    cols_show = ["Severity_Rank", "Rep_ID", "Cluster", "Status", "Severity_Score", "Cluster_Rel_Z_Norm", "Top_Driving_Features"]
    print(top_flagged[cols_show].head(10).to_string(index=False))
    