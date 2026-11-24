import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from phase3_preprocessing import run_phase3_preprocessing


def _distance_preservation_corr(X_high: np.ndarray, X_2d: np.ndarray, sample_size: int = 600, seed: int = 42) -> float:
    n = len(X_high)
    if n < 10:
        return np.nan
    rng = np.random.default_rng(seed)
    m = min(sample_size, n)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    d_high = pdist(X_high[idx], metric="euclidean")
    d_low = pdist(X_2d[idx], metric="euclidean")
    if np.std(d_high) < 1e-12 or np.std(d_low) < 1e-12:
        return 0.0
    rho, _ = spearmanr(d_high, d_low)
    return round(float(rho), 4) if rho is not None and not np.isnan(rho) else 0.0


def run_phase7_dim_reduction(
    X_scaled: np.ndarray,
    tsne_perplexity: float = 30.0,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
):
    n_samples, n_features = X_scaled.shape

    # 1. Full PCA for Scree / Explained Variance + 2D projection
    pca_full = PCA(n_components=n_features, random_state=random_state)
    pca_full.fit(X_scaled)
    evr = pca_full.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    scree_df = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_features)],
        "Explained_Variance_Ratio": np.round(evr, 4),
        "Cumulative_Variance": np.round(cum_evr, 4),
    })

    pca_2d = pca_full.transform(X_scaled)[:, :2]

    # 2. Tuned t-SNE (PCA init + auto learning_rate for stability)
    perp = min(float(tsne_perplexity), max(5.0, (n_samples - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=random_state,
    )
    tsne_2d = tsne.fit_transform(X_scaled)

    # 3. Tuned UMAP
    nbrs = min(int(umap_n_neighbors), max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=nbrs,
        min_dist=float(umap_min_dist),
        metric="euclidean",
        random_state=random_state,
    )
    umap_2d = reducer.fit_transform(X_scaled)

    # 4. Space distortion / distance rank preservation metrics
    fidelity_df = pd.DataFrame([
        {
            "Method": "PCA (2D)",
            "Params": "n_components=2, linear",
            "Variance_or_Loss_Note": f"CumVar={round(float(cum_evr[1])*100, 1)}%",
            "Dist_Spearman_Rho": _distance_preservation_corr(X_scaled, pca_2d, seed=random_state),
        },
        {
            "Method": "t-SNE (2D)",
            "Params": f"perp={perp}, init=pca",
            "Variance_or_Loss_Note": f"KL_div={round(float(tsne.kl_divergence_), 4)}",
            "Dist_Spearman_Rho": _distance_preservation_corr(X_scaled, tsne_2d, seed=random_state),
        },
        {
            "Method": "UMAP (2D)",
            "Params": f"nbrs={nbrs}, min_dist={umap_min_dist}",
            "Variance_or_Loss_Note": "Non-linear manifold",
            "Dist_Spearman_Rho": _distance_preservation_corr(X_scaled, umap_2d, seed=random_state),
        },
    ])

    embeddings_df = pd.DataFrame({
        "PCA_1": pca_2d[:, 0],
        "PCA_2": pca_2d[:, 1],
        "TSNE_1": tsne_2d[:, 0],
        "TSNE_2": tsne_2d[:, 1],
        "UMAP_1": umap_2d[:, 0],
        "UMAP_2": umap_2d[:, 1],
    })

    return {
        "embeddings_df": embeddings_df,
        "scree_df": scree_df,
        "fidelity_df": fidelity_df,
        "pca_2d_variance_pct": round(float(cum_evr[1]) * 100.0, 2),
    }


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    out = run_phase7_dim_reduction(prep["X_scaled"].values, random_state=42)

    print("=== PHASE 7 SCREE / EXPLAINED VARIANCE ===")
    print(out["scree_df"].to_string(index=False))

    print("\n=== PHASE 7 2D EMBEDDING FIDELITY ===")
    print(out["fidelity_df"].to_string(index=False))