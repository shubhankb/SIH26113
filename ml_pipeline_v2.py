import pandas as pd
from phase3_preprocessing import run_phase3_preprocessing
from phase5_tuning import run_phase5_optimization
from phase6_evaluation import run_phase6_evaluation
from phase7_dim_reduction import run_phase7_dim_reduction
from phase8_profiling import run_phase8_profiling
from phase9_anomalies import run_phase9_anomaly_detection
from phase10_reliability import run_phase10_reliability
from phase11_insights import generate_ml_insights


def run_full_ml_pipeline(
    phase2_csv: str = "phase2_kinematic_features.csv",
    scaler_type: str = "robust",
    prune_collinear: bool = True,
    collinear_threshold: float = 0.95,
    contamination: float = 0.07,
    lof_neighbors: int = 20,
    random_state: int = 42,
):
    # Phase 3: Imputation, collinearity pruning & scaling
    prep = run_phase3_preprocessing(
        input_csv=phase2_csv,
        scaler_type=scaler_type,
        prune_collinear=prune_collinear,
        collinear_threshold=collinear_threshold,
    )
    X_scaled_df = prep["X_scaled"]
    X_arr = X_scaled_df.values

    # Phase 5: Hyperparameter tuning
    opt = run_phase5_optimization(X_arr, random_state=random_state)
    tuned_params = opt["best_params"]

    # Phase 6: Multi-model evaluation scorecard
    scorecard_df, fitted_labels, tuned_params = run_phase6_evaluation(
        X_arr, tuned_params=tuned_params, random_state=random_state
    )

    # Phase 7: Dimensionality reduction (PCA, t-SNE, UMAP) + distance fidelity
    dim_res = run_phase7_dim_reduction(X_arr, random_state=random_state)

    # Phase 8: Cluster profiling (K-Means & GMM profiles)
    km_lbl = fitted_labels["kmeans"]
    prof_summary, prof_mean, prof_med, prof_std = run_phase8_profiling(prep["X_selected"], km_lbl)

    # Phase 9: Enhanced anomaly detection (IF + LOF + severity + top driving features)
    anom_all, top_anomalies, anom_summary = run_phase9_anomaly_detection(
        X_scaled=X_arr,
        feature_names=prep["active_features"],
        rep_ids=prep["rep_ids"],
        cluster_labels=km_lbl,
        contamination=contamination,
        lof_neighbors=lof_neighbors,
        random_state=random_state,
    )

    # Phase 10: Stability & reliability experiments
    km_k = int(tuned_params["kmeans"]["n_clusters"])
    reliability = run_phase10_reliability(X_scaled_df, n_clusters=km_k, random_state=random_state)

    # Phase 11: Automated data-backed insights
    db_row = scorecard_df[scorecard_df["Model"].str.startswith("DBSCAN")]
    db_noise = float(db_row["Noise_%"].iloc[0]) if not db_row.empty else 0.0
    min_ari = float(reliability["seed_stability"]["ARI_vs_Seed42"].min())
    abl_non_full = reliability["feature_ablation"][
        reliability["feature_ablation"]["Remaining_Cols"] < len(prep["active_features"])
    ]
    key_driver = (
        str(abl_non_full.sort_values("ARI_vs_Full").iloc[0]["Dropped_Feature"])
        if not abl_non_full.empty
        else "n_frames"
    )

    insights = generate_ml_insights(
        n_rows=len(X_scaled_df),
        active_features=prep["active_features"],
        scorecard_df=scorecard_df,
        profile_df=prof_summary,
        anom_summary=anom_summary,
        dbscan_noise_pct=db_noise,
        seed_min_ari=min_ari,
        key_driver_feature=key_driver,
    )

    # Combined results table
    results_df = pd.DataFrame({
        "Rep_ID": prep["rep_ids"],
        "Cluster_KMeans": fitted_labels["kmeans"],
        "Cluster_Hierarchical": fitted_labels["hierarchical"],
        "Cluster_DBSCAN": fitted_labels["dbscan"],
        "Cluster_GMM": fitted_labels["gmm"],
        "Anomaly_Status": anom_all["Status"],
        "Anomaly_Severity": anom_all["Severity_Score"],
        "Top_Driving_Features": anom_all["Top_Driving_Features"],
    })
    results_df = pd.concat([results_df, dim_res["embeddings_df"]], axis=1)

    return {
        "prep": prep,
        "tuned_params": tuned_params,
        "tuning_tables": opt,
        "scorecard": scorecard_df,
        "scree": dim_res["scree_df"],
        "fidelity": dim_res["fidelity_df"],
        "profile_summary": prof_summary,
        "profile_median": prof_med,
        "anomalies_all": anom_all,
        "anomalies_top": top_anomalies,
        "anomaly_summary": anom_summary,
        "reliability": reliability,
        "insights": insights,
        "results_df": results_df,
    }


if __name__ == "__main__":
    bundle = run_full_ml_pipeline(random_state=42)
    print("=== PHASE 12 END-TO-END PIPELINE COMPLETE ===")
    print("Tuned Params:", bundle["tuned_params"])
    print("\n--- Automated Insights ---")
    for idx, item in enumerate(bundle["insights"], 1):
        print(f"  {idx}. {item}")