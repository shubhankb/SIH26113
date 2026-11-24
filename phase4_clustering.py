import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from phase3_preprocessing import run_phase3_preprocessing


def estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int = 5, percentile: float = 90.0) -> float:
    """Data-driven eps estimate using k-th nearest neighbor distances (no labels used)."""
    k = min(max(2, min_samples), len(X_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    k_dist = np.sort(distances[:, k])
    eps_est = float(np.percentile(k_dist, percentile))
    return round(max(eps_est, 0.1), 4)


def _cluster_metrics_detail(labels: np.ndarray) -> tuple[dict, float]:
    non_noise = labels[labels != -1]
    if len(non_noise) == 0:
        return {}, 0.0
    unique, counts = np.unique(non_noise, return_counts=True)
    sizes = {str(int(k)): int(v) for k, v in zip(unique, counts)}
    balance = float(np.min(counts) / np.max(counts)) if len(counts) > 1 and np.max(counts) > 0 else 1.0
    return sizes, round(balance, 4)


def run_phase4_clustering_improved(
    X_scaled: np.ndarray,
    n_clusters: int = 3,
    eps: float = None,
    min_samples: int = 5,
    linkage: str = "ward",
    random_state: int = 42,
    include_spectral: bool = True,
    spectral_max_samples: int = 3000,
):
    if len(X_scaled) < 6:
        raise ValueError("Clustering ke liye kam se kam 6 samples chahiye.")

    effective_eps = eps if eps is not None else estimate_dbscan_eps(X_scaled, min_samples=min_samples)

    models = {}
    labels_dict = {}
    summary_rows = []

    # 1. Improved K-Means++ (higher n_init for better local-minima escape)
    t0 = time.perf_counter()
    km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=25, max_iter=500)
    km_labels = km.fit_predict(X_scaled)
    km_dt = time.perf_counter() - t0
    models["kmeans"] = km
    labels_dict["kmeans"] = km_labels
    km_sizes, km_bal = _cluster_metrics_detail(km_labels)
    summary_rows.append({
        "Model": "K-Means++ (n_init=25)",
        "Params": f"k={n_clusters}",
        "Clusters": int(len(km_sizes)),
        "Noise_Pct": 0.0,
        "Balance_Ratio": km_bal,
        "Runtime_sec": round(km_dt, 4),
        "Sizes": str(km_sizes),
    })

    # 2. Hierarchical / Agglomerative (configurable linkage, fallback to ward if non-Euclidean)
    t0 = time.perf_counter()
    eff_linkage = linkage if linkage in ("ward", "complete", "average", "single") else "ward"
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=eff_linkage)
    agg_labels = agg.fit_predict(X_scaled)
    agg_dt = time.perf_counter() - t0
    models["hierarchical"] = agg
    labels_dict["hierarchical"] = agg_labels
    agg_sizes, agg_bal = _cluster_metrics_detail(agg_labels)
    summary_rows.append({
        "Model": f"Hierarchical ({eff_linkage})",
        "Params": f"k={n_clusters}, link={eff_linkage}",
        "Clusters": int(len(agg_sizes)),
        "Noise_Pct": 0.0,
        "Balance_Ratio": agg_bal,
        "Runtime_sec": round(agg_dt, 4),
        "Sizes": str(agg_sizes),
    })

    # 3. Tuned-eps DBSCAN (prevents blind 1-cluster / high-noise collapse)
    t0 = time.perf_counter()
    db = DBSCAN(eps=effective_eps, min_samples=min_samples)
    db_labels = db.fit_predict(X_scaled)
    db_dt = time.perf_counter() - t0
    models["dbscan"] = db
    labels_dict["dbscan"] = db_labels
    db_sizes, db_bal = _cluster_metrics_detail(db_labels)
    noise_pct = round(float(np.mean(db_labels == -1) * 100.0), 2)
    summary_rows.append({
        "Model": "DBSCAN (k-NN eps)",
        "Params": f"eps={effective_eps}, min={min_samples}",
        "Clusters": int(len(db_sizes)),
        "Noise_Pct": noise_pct,
        "Balance_Ratio": db_bal,
        "Runtime_sec": round(db_dt, 4),
        "Sizes": str(db_sizes),
    })

    # 4. Gaussian Mixture Model (with BIC/AIC diagnostics & regularization)
    t0 = time.perf_counter()
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        reg_covar=1e-5,
        n_init=10,
        random_state=random_state,
    )
    gmm.fit(X_scaled)
    gmm_labels = gmm.predict(X_scaled)
    gmm_dt = time.perf_counter() - t0
    models["gmm"] = gmm
    labels_dict["gmm"] = gmm_labels
    gmm_sizes, gmm_bal = _cluster_metrics_detail(gmm_labels)
    summary_rows.append({
        "Model": f"GMM (BIC={round(float(gmm.bic(X_scaled)), 1)})",
        "Params": f"comp={n_clusters}, full",
        "Clusters": int(len(gmm_sizes)),
        "Noise_Pct": 0.0,
        "Balance_Ratio": gmm_bal,
        "Runtime_sec": round(gmm_dt, 4),
        "Sizes": str(gmm_sizes),
    })

    # 5. Spectral Clustering (nearest_neighbors affinity for local manifold structure)
    if include_spectral and len(X_scaled) <= spectral_max_samples:
        t0 = time.perf_counter()
        n_nbrs = min(15, max(2, len(X_scaled) - 1))
        spec = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_nbrs,
            assign_labels="kmeans",
            random_state=random_state,
        )
        spec_labels = spec.fit_predict(X_scaled)
        spec_dt = time.perf_counter() - t0
        models["spectral"] = spec
        labels_dict["spectral"] = spec_labels
        spec_sizes, spec_bal = _cluster_metrics_detail(spec_labels)
        summary_rows.append({
            "Model": "Spectral (kNN graph)",
            "Params": f"k={n_clusters}, nbrs={n_nbrs}",
            "Clusters": int(len(spec_sizes)),
            "Noise_Pct": 0.0,
            "Balance_Ratio": spec_bal,
            "Runtime_sec": round(spec_dt, 4),
            "Sizes": str(spec_sizes),
        })
    else:
        summary_rows.append({
            "Model": "Spectral (kNN graph)",
            "Params": "N/A",
            "Clusters": np.nan,
            "Noise_Pct": np.nan,
            "Balance_Ratio": np.nan,
            "Runtime_sec": np.nan,
            "Sizes": f"Skipped (n={len(X_scaled)} > {spectral_max_samples})",
        })

    return {
        "labels": labels_dict,
        "models": models,
        "effective_eps": effective_eps,
        "summary_df": pd.DataFrame(summary_rows),
    }


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    out = run_phase4_clustering_improved(
        prep["X_scaled"].values,
        n_clusters=3,
        eps=None,  # Auto-estimate from 90th percentile k-NN distance
        min_samples=5,
        linkage="ward",
        random_state=42,
    )
    print(f"=== PHASE 4 IMPROVED (Auto-estimated eps = {out['effective_eps']}) ===")
    print(out["summary_df"].to_string(index=False))