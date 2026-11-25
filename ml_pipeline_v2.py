import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from phase3_preprocessing import run_phase3_preprocessing
from phase5_tuning import run_phase5_optimization
from phase6_evaluation import run_phase6_evaluation
from phase7_dim_reduction import run_phase7_dim_reduction
from phase8_profiling import run_phase8_profiling
from phase9_anomalies import run_phase9_anomaly_detection
from phase10_reliability import run_phase10_reliability
from phase11_insights import generate_ml_insights

from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def fit_balanced_kmeans(
    X_raw_df: pd.DataFrame,
    scaler=None,
    k_candidates=(3, 4, 5),
    min_cluster_fraction: float = 0.03,
    max_cluster_fraction: float = 0.70,
    use_pca_dims: int = 4,
    random_state: int = 42,
):
    X_work = X_raw_df.copy()

    skewed_cols = [
        c
        for c in [
            "spatial_rom_3d",
            "mean_speed_3d",
            "p95_speed_3d",
            "mean_jerk_3d",
            "norm_variability",
        ]
        if c in X_work.columns
    ]
    for col in skewed_cols:
        s = pd.to_numeric(X_work[col], errors="coerce").clip(lower=0.0)
        X_work[col] = np.log1p(s)

    qt = QuantileTransformer(
        n_quantiles=min(500, len(X_work)),
        output_distribution="normal",
        random_state=random_state,
    )
    X_norm = qt.fit_transform(X_work)

    n_comps = min(use_pca_dims, X_norm.shape[1])
    X_proj = PCA(n_components=n_comps, random_state=random_state).fit_transform(X_norm)

    n_total = len(X_work)
    min_req = max(15, int(n_total * min_cluster_fraction))
    max_allowed = int(n_total * max_cluster_fraction)

    best_k = k_candidates[0]
    best_score = -1.0
    best_labels = None
    best_model = None

    # Pass 1: Strict balance (min_req <= count <= max_allowed for every cluster)
    for k in k_candidates:
        model = KMeans(n_clusters=k, n_init=25, random_state=random_state)
        labels = model.fit_predict(X_proj)
        counts = np.bincount(labels, minlength=k)
        if np.min(counts) < min_req or np.max(counts) > max_allowed:
            continue
        score = silhouette_score(X_proj, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = model

    # Pass 2: Fallback only on min_req if no k satisfies max_allowed
    if best_labels is None:
        for k in k_candidates:
            model = KMeans(n_clusters=k, n_init=25, random_state=random_state)
            labels = model.fit_predict(X_proj)
            counts = np.bincount(labels, minlength=k)
            if np.min(counts) < min_req:
                continue
            score = silhouette_score(X_proj, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_model = model

    if best_labels is None:
        best_model = KMeans(n_clusters=k_candidates[0], n_init=25, random_state=random_state)
        best_labels = best_model.fit_predict(X_proj)
        best_k = k_candidates[0]
        best_score = float(silhouette_score(X_proj, best_labels))

    return {
        "k": best_k,
        "labels": best_labels,
        "silhouette": float(best_score),
        "model": best_model,
        "X_scaled": X_proj,
    }
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

    # Balanced K-Means on selected features (log1p + scaling + min cluster size filter)
    active_scaler = prep.get("scaler")
    if active_scaler is None:
        active_scaler = (
            StandardScaler()
            if str(scaler_type).strip().lower() in {"std", "standard", "standardscaler"}
            else RobustScaler()
        )

    balanced_km = fit_balanced_kmeans(
        X_raw_df=prep["X_selected"],
        scaler=active_scaler,
        k_candidates=(3, 4, 5),
        min_cluster_fraction=0.02,
        random_state=random_state,
    )
    km_lbl = balanced_km["labels"]
    fitted_labels["kmeans"] = km_lbl
    if isinstance(tuned_params, dict) and isinstance(tuned_params.get("kmeans"), dict):
        tuned_params["kmeans"]["n_clusters"] = int(balanced_km["k"])

    # Phase 7: Dimensionality reduction (PCA, t-SNE, UMAP) + distance fidelity
    dim_res = run_phase7_dim_reduction(X_arr, random_state=random_state)

    # Phase 8: Cluster profiling (K-Means balanced labels)
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
    km_k = int(balanced_km["k"])
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