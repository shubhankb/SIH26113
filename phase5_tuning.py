import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from phase3_preprocessing import run_phase3_preprocessing


def tune_kmeans(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8, random_state: int = 42):
    rows = []
    best_k = k_min
    best_sil = -1.0

    for k in range(k_min, min(k_max, len(X_scaled) - 1) + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=400, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        dbi = float(davies_bouldin_score(X_scaled, labels))
        chi = float(calinski_harabasz_score(X_scaled, labels))
        rows.append({
            "K": k,
            "Inertia": round(float(km.inertia_), 2),
            "Silhouette": round(sil, 4),
            "Davies_Bouldin": round(dbi, 4),
            "Calinski_Harabasz": round(chi, 2),
        })
        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {"best_k": best_k, "best_silhouette": round(best_sil, 4), "table": pd.DataFrame(rows)}


def tune_hierarchical_linkage(X_scaled: np.ndarray, n_clusters: int = 3):
    linkages = ["ward", "complete", "average"]
    rows = []
    best_link = "ward"
    best_sil = -1.0

    for link in linkages:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        labels = agg.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        dbi = float(davies_bouldin_score(X_scaled, labels))
        rows.append({
            "Linkage": link,
            "Silhouette": round(sil, 4),
            "Davies_Bouldin": round(dbi, 4),
        })
        if sil > best_sil:
            best_sil = sil
            best_link = link

    return {"best_linkage": best_link, "best_silhouette": round(best_sil, 4), "table": pd.DataFrame(rows)}


def tune_dbscan(X_scaled: np.ndarray, min_samples_grid: list = None):
    if min_samples_grid is None:
        min_samples_grid = [5, 10]

    nn = NearestNeighbors(n_neighbors=min(11, len(X_scaled)))
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)
    base_q = np.percentile(dists[:, min(5, dists.shape[1] - 1)], [75, 85, 90, 94, 97])
    eps_grid = sorted(set(round(max(float(v), 0.15), 3) for v in base_q))

    rows = []
    best_cfg = {"eps": eps_grid[len(eps_grid) // 2], "min_samples": min_samples_grid[0], "score": -999.0}

    for ms in min_samples_grid:
        for eps_val in eps_grid:
            db = DBSCAN(eps=eps_val, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            noise_pct = float(np.mean(labels == -1) * 100.0)
            clusters_only = [c for c in set(labels) if c != -1]
            n_clust = len(clusters_only)

            sil = np.nan
            if n_clust >= 2 and np.sum(labels != -1) >= 4:
                mask = labels != -1
                if len(set(labels[mask])) >= 2:
                    sil = float(silhouette_score(X_scaled[mask], labels[mask]))

            # Prefer 2..6 clusters, moderate noise (<35%), and valid silhouette
            valid_penalty = 0.0
            if n_clust < 2 or n_clust > 6:
                valid_penalty -= 1.0
            if noise_pct > 35.0:
                valid_penalty -= 0.5
            score = (sil if not np.isnan(sil) else -0.5) + valid_penalty

            rows.append({
                "eps": eps_val,
                "min_samples": ms,
                "Clusters": n_clust,
                "Noise_Pct": round(noise_pct, 2),
                "Silhouette_NonNoise": round(sil, 4) if not np.isnan(sil) else np.nan,
                "Selection_Score": round(score, 4),
            })
            if score > best_cfg["score"]:
                best_cfg = {
                    "eps": eps_val,
                    "min_samples": ms,
                    "clusters": n_clust,
                    "noise_pct": round(noise_pct, 2),
                    "score": round(score, 4),
                }

    return {"best_params": best_cfg, "table": pd.DataFrame(rows)}


def tune_gmm(X_scaled: np.ndarray, components_range: range = range(2, 7), random_state: int = 42):
    cov_types = ["full", "diag", "tied"]
    rows = []
    best_cfg = None
    best_bic = np.inf

    for cov in cov_types:
        for n_comp in components_range:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=cov,
                reg_covar=1e-5,
                n_init=5,
                random_state=random_state,
            )
            gmm.fit(X_scaled)
            bic = float(gmm.bic(X_scaled))
            aic = float(gmm.aic(X_scaled))
            rows.append({
                "Covariance": cov,
                "Components": n_comp,
                "BIC": round(bic, 1),
                "AIC": round(aic, 1),
            })
            if bic < best_bic:
                best_bic = bic
                best_cfg = {"n_components": n_comp, "covariance_type": cov, "bic": round(bic, 1)}

    return {"best_params": best_cfg, "table": pd.DataFrame(rows)}


def run_phase5_optimization(X_scaled: np.ndarray, random_state: int = 42):
    km_res = tune_kmeans(X_scaled, k_min=2, k_max=7, random_state=random_state)
    link_res = tune_hierarchical_linkage(X_scaled, n_clusters=km_res["best_k"])
    db_res = tune_dbscan(X_scaled, min_samples_grid=[5, 10])
    gmm_res = tune_gmm(X_scaled, components_range=range(2, 7), random_state=random_state)

    best_params_summary = {
        "kmeans": {"n_clusters": km_res["best_k"], "silhouette": km_res["best_silhouette"]},
        "hierarchical": {
            "n_clusters": km_res["best_k"],
            "linkage": link_res["best_linkage"],
            "silhouette": link_res["best_silhouette"],
        },
        "dbscan": db_res["best_params"],
        "gmm": gmm_res["best_params"],
    }

    return {
        "best_params": best_params_summary,
        "kmeans_table": km_res["table"],
        "linkage_table": link_res["table"],
        "dbscan_table": db_res["table"],
        "gmm_table": gmm_res["table"],
    }


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    opt = run_phase5_optimization(prep["X_scaled"].values, random_state=42)

    print("=== PHASE 5 HYPERPARAMETER OPTIMIZATION RESULTS ===")
    for model_name, params in opt["best_params"].items():
        print(f"Best {model_name:14s}: {params}")

    print("\n--- K-Means K Search ---")
    print(opt["kmeans_table"].to_string(index=False))

    print("\n--- GMM Best 5 Configs by BIC ---")
    print(opt["gmm_table"].sort_values("BIC").head(5).to_string(index=False))