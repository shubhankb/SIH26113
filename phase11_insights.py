import pandas as pd
from phase3_preprocessing import run_phase3_preprocessing
from phase6_evaluation import run_phase6_evaluation
from phase8_profiling import run_phase8_profiling
from phase9_anomalies import run_phase9_anomaly_detection
from phase10_reliability import run_phase10_reliability


def generate_ml_insights(
    n_rows: int,
    active_features: list,
    scorecard_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    anom_summary: dict,
    dbscan_noise_pct: float,
    seed_min_ari: float,
    key_driver_feature: str,
) -> list[str]:
    insights = []

    # 1. Dataset & unsupervised framing
    insights.append(
        f"Analyzed {n_rows} unlabeled rehabilitation repetitions across {len(active_features)} engineered 3D kinematic features ({', '.join(active_features)})."
    )

    # 2. Cluster count & profile summary
    if not profile_df.empty:
        n_pat = len(profile_df)
        insights.append(
            f"Unsupervised clustering identifies {n_pat} distinct movement pattern group(s) in the scaled feature space."
        )
        for _, r in profile_df.iterrows():
            insights.append(
                f"Cluster {int(r['Cluster'])} covers {int(r['Count'])} repetitions ({r['Share_%']}%) — {r['Observed_Characteristics'].lower()}."
            )

    # 3. DBSCAN noise & algorithm comparison takeaway
    if not scorecard_df.empty:
        best_sil_row = scorecard_df.dropna(subset=["Silhouette (↑)"]).sort_values("Silhouette (↑)", ascending=False).iloc[0]
        best_bal_row = scorecard_df.dropna(subset=["Balance"]).sort_values("Balance", ascending=False).iloc[0]
        insights.append(
            f"Model trade-off: {best_sil_row['Model']} achieves the highest silhouette ({best_sil_row['Silhouette (↑)']}), while {best_bal_row['Model']} provides the highest cluster balance ({best_bal_row['Balance']})."
        )
    insights.append(
        f"Density-based DBSCAN flags {dbscan_noise_pct:.2f}% of repetitions as noise."
    )

    # 4. Anomaly detection & stability takeaways
    insights.append(
        f"Anomaly detection flags {anom_summary.get('any_flagged', 0)} repetitions overall, including {anom_summary.get('consensus_flagged', 0)} consensus anomalies flagged by both Isolation Forest and LOF."
    )
    insights.append(
        f"Stability check shows minimum seed ARI of {seed_min_ari:.4f}, with '{key_driver_feature}' showing the strongest impact on the k=2 partition under ablation."
    )

    return insights


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    X_scaled_df = prep["X_scaled"]
    X_arr = X_scaled_df.values

    scorecard_df, fitted_labels, _ = run_phase6_evaluation(X_arr, random_state=42)
    prof_df, _, _, _ = run_phase8_profiling(prep["X_selected"], fitted_labels["kmeans"])
    _, _, anom_summ = run_phase9_anomaly_detection(
        X_scaled=X_arr,
        feature_names=prep["active_features"],
        rep_ids=prep["rep_ids"],
        cluster_labels=fitted_labels["kmeans"],
        contamination=0.07,
        lof_neighbors=20,
        random_state=42,
    )
    rel = run_phase10_reliability(X_scaled_df, n_clusters=2, random_state=42)

    db_row = scorecard_df[scorecard_df["Model"].str.startswith("DBSCAN")]
    db_noise = float(db_row["Noise_%"].iloc[0]) if not db_row.empty else 0.0

    min_ari = float(rel["seed_stability"]["ARI_vs_Seed42"].min())
    abl_non_full = rel["feature_ablation"][rel["feature_ablation"]["Remaining_Cols"] < len(prep["active_features"])]
    key_driver = str(abl_non_full.sort_values("ARI_vs_Full").iloc[0]["Dropped_Feature"]) if not abl_non_full.empty else "n_frames"

    lines = generate_ml_insights(
        n_rows=len(X_scaled_df),
        active_features=prep["active_features"],
        scorecard_df=scorecard_df,
        profile_df=prof_df,
        anom_summary=anom_summ,
        dbscan_noise_pct=db_noise,
        seed_min_ari=min_ari,
        key_driver_feature=key_driver,
    )

    print("=== PHASE 11 AUTOMATED ML INSIGHTS ===")
    for i, line in enumerate(lines, 1):
        print(f"{i}. {line}")