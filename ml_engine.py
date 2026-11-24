import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

import umap

FEATURE_COLS = [
    "Range_of_Motion_deg",
    "Mean_Angular_Vel",
    "Max_Angular_Vel",
    "Movement_Jerk",
    "Joint_Symmetry_Index",
    "Rep_Variability",
]


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 3,
    eps: float = 1.1,
    min_samples: int = 5,
    contamination: float = 0.07,
):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    work_df = df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    if "Rep_ID" not in work_df.columns:
        work_df.insert(0, "Rep_ID", [f"REP_{i+1:04d}" for i in range(len(work_df))])

    if len(work_df) < 5:
        raise ValueError("At least 5 valid rows are required to run the unsupervised pipeline.")

    X = work_df[FEATURE_COLS].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow Method (WCSS / Inertia for K = 1 to 7)
    max_k = min(8, len(work_df))
    elbow_rows = []
    for k in range(1, max_k):
        km_elbow = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km_elbow.fit(X_scaled)
        elbow_rows.append({"K": k, "WCSS_Inertia": round(float(km_elbow.inertia_), 2)})
    elbow_df = pd.DataFrame(elbow_rows)

    # 2. Clustering Algorithms (100% Unsupervised)
    k_eff = min(n_clusters, len(work_df) - 1) if len(work_df) > 1 else 1
    kmeans = KMeans(n_clusters=k_eff, init="k-means++", random_state=42, n_init=10)
    k_labels = kmeans.fit_predict(X_scaled)

    agg = AgglomerativeClustering(n_clusters=k_eff, linkage="ward")
    h_labels = agg.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    d_labels = dbscan.fit_predict(X_scaled)

    # 3. Internal Validation Metrics (No ground-truth labels used)
    def internal_metrics(labels: np.ndarray, name: str) -> dict:
        unique_clusters = set(labels) - {-1}
        n_found = len(unique_clusters)
        noise_pts = int(np.sum(labels == -1))

        if n_found < 2 or len(set(labels)) < 2:
            return {
                "Algorithm": name,
                "Clusters Found": n_found,
                "Noise Points": noise_pts,
                "Silhouette (Higher=Better)": np.nan,
                "Davies-Bouldin (Lower=Better)": np.nan,
                "Calinski-Harabasz (Higher=Better)": np.nan,
            }

        try:
            sil = round(float(silhouette_score(X_scaled, labels)), 3)
            dbi = round(float(davies_bouldin_score(X_scaled, labels)), 3)
            chi = round(float(calinski_harabasz_score(X_scaled, labels)), 2)
        except Exception:
            sil, dbi, chi = np.nan, np.nan, np.nan

        return {
            "Algorithm": name,
            "Clusters Found": n_found,
            "Noise Points": noise_pts,
            "Silhouette (Higher=Better)": sil,
            "Davies-Bouldin (Lower=Better)": dbi,
            "Calinski-Harabasz (Higher=Better)": chi,
        }

    metrics_df = pd.DataFrame([
        internal_metrics(k_labels, "K-Means (K-Means++)"),
        internal_metrics(h_labels, "Hierarchical (Ward)"),
        internal_metrics(d_labels, "DBSCAN (Density-Based)"),
    ])

    # 4. Dimensionality Reduction (PCA, t-SNE, UMAP)
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)

    perplexity = min(25, max(2, (len(work_df) - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_2d = tsne.fit_transform(X_scaled)

    n_neighbors = min(15, max(2, len(work_df) - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    umap_2d = reducer.fit_transform(X_scaled)

    # 5. Unsupervised Anomaly Detection (Isolation Forest)
    iso = IsolationForest(contamination=contamination, random_state=42)
    anom_pred = iso.fit_predict(X_scaled)

    results_df = work_df.copy()
    results_df["Cluster_KMeans"] = k_labels.astype(str)
    results_df["Cluster_Hierarchical"] = h_labels.astype(str)
    results_df["Cluster_DBSCAN"] = d_labels.astype(str)
    results_df["Anomaly_Flag"] = np.where(anom_pred == -1, "Unusual / Flagged", "Normal Pattern")

    results_df["PCA_1"], results_df["PCA_2"] = pca_2d[:, 0], pca_2d[:, 1]
    results_df["TSNE_1"], results_df["TSNE_2"] = tsne_2d[:, 0], tsne_2d[:, 1]
    results_df["UMAP_1"], results_df["UMAP_2"] = umap_2d[:, 0], umap_2d[:, 1]

    return results_df, metrics_df, elbow_df, pca.explained_variance_ratio_