import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from phase3_preprocessing import run_phase3_preprocessing
from phase5_tuning import run_phase5_optimization


def _evaluate_labels(X_scaled: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float, int]:
    unique_all = set(labels)
    non_noise_clusters = sorted(c for c in unique_all if c != -1)
    n_clusters = len(non_noise_clusters)

    noise_mask = labels == -1
    noise_pct = round(float(np.mean(noise_mask) * 100.0), 2)

    valid_mask = ~noise_mask
    if n_clusters >= 2 and np.sum(valid_mask) >= 4 and len(set(labels[valid_mask])) >= 2:
        X_eval = X_scaled[valid_mask]
        y_eval = labels[valid_mask]
        sil = round(float(silhouette_score(X_eval, y_eval)), 4)
        dbi = round(float(davies_bouldin_score(X_eval, y_eval)), 4)
        chi = round(float(calinski_harabasz_score(X_eval, y_eval)), 2)
    else:
        sil, dbi, chi = np.nan, np.nan, np.nan

    if n_clusters >= 2 and np.sum(valid_mask) > 0:
        _, counts = np.unique(labels[valid_mask], return_counts=True)
        balance = round(float(np.min(counts) / np.max(counts)), 4) if np.max(counts) > 0 else 0.0
    elif n_clusters == 1:
        balance = 1.0
    else:
        balance = 0.0

    return sil, dbi, chi, balance, n_clusters, noise_pct


def run_phase6_evaluation(X_scaled: np.ndarray, tuned_params: dict = None, random_state: int = 42):
    if tuned_params is None:
        opt = run_phase5_optimization(X_scaled, random_state=random_state)
        tuned_params = opt["best_params"]

    km_k = int(tuned_params["kmeans"]["n_clusters"])
    hier_k = int(tuned_params["hierarchical"]["n_clusters"])
    hier_link = tuned_params["hierarchical"]["linkage"]
    db_eps = float(tuned_params["dbscan"]["eps"])
    db_ms = int(tuned_params["dbscan"]["min_samples"])
    gmm_comp = int(tuned_params["gmm"]["n_components"])
    gmm_cov = tuned_params["gmm"]["covariance_type"]

    scorecard = []
    fitted_labels = {}

    # 1. Tuned K-Means++
    t0 = time.perf_counter()
    km = KMeans(n_clusters=km_k, init="k-means++", n_init=25, max_iter=500, random_state=random_state)
    km_lbl = km.fit_predict(X_scaled)
    dt = time.perf_counter() - t0
    fitted_labels["kmeans"] = km_lbl
    sil, dbi, chi, bal, n_c, n_pct = _evaluate_labels(X_scaled, km_lbl)
    scorecard.append({
        "Model": f"K-Means++ (k={km_k})",
        "Silhouette (↑)": sil,
        "Davies-Bouldin (↓)": dbi,
        "Calinski-Harabasz (↑)": chi,
        "WCSS_Inertia": round(float(km.inertia_), 2),
        "Clusters": n_c,
        "Noise_%": n_pct,
        "Balance": bal,
        "Time_s": round(dt, 4),
    })

    # 2. Tuned Hierarchical Clustering
    t0 = time.perf_counter()
    agg = AgglomerativeClustering(n_clusters=hier_k, linkage=hier_link)
    agg_lbl = agg.fit_predict(X_scaled)
    dt = time.perf_counter() - t0
    fitted_labels["hierarchical"] = agg_lbl
    sil, dbi, chi, bal, n_c, n_pct = _evaluate_labels(X_scaled, agg_lbl)
    scorecard.append({
        "Model": f"Hierarchical ({hier_link}, k={hier_k})",
        "Silhouette (↑)": sil,
        "Davies-Bouldin (↓)": dbi,
        "Calinski-Harabasz (↑)": chi,
        "WCSS_Inertia": np.nan,
        "Clusters": n_c,
        "Noise_%": n_pct,
        "Balance": bal,
        "Time_s": round(dt, 4),
    })

    # 3. Tuned DBSCAN
    t0 = time.perf_counter()
    db = DBSCAN(eps=db_eps, min_samples=db_ms)
    db_lbl = db.fit_predict(X_scaled)
    dt = time.perf_counter() - t0
    fitted_labels["dbscan"] = db_lbl
    sil, dbi, chi, bal, n_c, n_pct = _evaluate_labels(X_scaled, db_lbl)
    scorecard.append({
        "Model": f"DBSCAN (eps={db_eps}, m={db_ms})",
        "Silhouette (↑)": sil,
        "Davies-Bouldin (↓)": dbi,
        "Calinski-Harabasz (↑)": chi,
        "WCSS_Inertia": np.nan,
        "Clusters": n_c,
        "Noise_%": n_pct,
        "Balance": bal,
        "Time_s": round(dt, 4),
    })

    # 4. Tuned GMM
    t0 = time.perf_counter()
    gmm = GaussianMixture(
        n_components=gmm_comp,
        covariance_type=gmm_cov,
        reg_covar=1e-5,
        n_init=10,
        random_state=random_state,
    )
    gmm.fit(X_scaled)
    gmm_lbl = gmm.predict(X_scaled)
    dt = time.perf_counter() - t0
    fitted_labels["gmm"] = gmm_lbl
    sil, dbi, chi, bal, n_c, n_pct = _evaluate_labels(X_scaled, gmm_lbl)
    scorecard.append({
        "Model": f"GMM ({gmm_cov}, c={gmm_comp})",
        "Silhouette (↑)": sil,
        "Davies-Bouldin (↓)": dbi,
        "Calinski-Harabasz (↑)": chi,
        "WCSS_Inertia": np.nan,
        "Clusters": n_c,
        "Noise_%": n_pct,
        "Balance": bal,
        "Time_s": round(dt, 4),
    })

    # 5. Spectral Clustering (kNN graph)
    if len(X_scaled) <= 3000:
        t0 = time.perf_counter()
        n_nbrs = min(15, max(2, len(X_scaled) - 1))
        spec = SpectralClustering(
            n_clusters=km_k,
            affinity="nearest_neighbors",
            n_neighbors=n_nbrs,
            assign_labels="kmeans",
            random_state=random_state,
        )
        spec_lbl = spec.fit_predict(X_scaled)
        dt = time.perf_counter() - t0
        fitted_labels["spectral"] = spec_lbl
        sil, dbi, chi, bal, n_c, n_pct = _evaluate_labels(X_scaled, spec_lbl)
        scorecard.append({
            "Model": f"Spectral (kNN, k={km_k})",
            "Silhouette (↑)": sil,
            "Davies-Bouldin (↓)": dbi,
            "Calinski-Harabasz (↑)": chi,
            "WCSS_Inertia": np.nan,
            "Clusters": n_c,
            "Noise_%": n_pct,
            "Balance": bal,
            "Time_s": round(dt, 4),
        })

    return pd.DataFrame(scorecard), fitted_labels, tuned_params


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    scorecard_df, _, _ = run_phase6_evaluation(prep["X_scaled"].values, random_state=42)
    print("=== PHASE 6 COMPARATIVE SCORECARD ===")
    print(scorecard_df.to_string(index=False))