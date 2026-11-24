import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from phase3_preprocessing import run_phase3_preprocessing


def test_seed_stability(X_scaled: np.ndarray, n_clusters: int = 2, seeds: list = None) -> pd.DataFrame:
    if seeds is None:
        seeds = [42, 7, 123, 2026]

    label_sets = {}
    for s in seeds:
        km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, max_iter=400, random_state=s)
        label_sets[s] = km.fit_predict(X_scaled)

    ref_seed = seeds[0]
    ref_lbl = label_sets[ref_seed]
    rows = []
    for s in seeds:
        ari_vs_ref = float(adjusted_rand_score(ref_lbl, label_sets[s]))
        sil = float(silhouette_score(X_scaled, label_sets[s]))
        rows.append({
            "Seed": s,
            "ARI_vs_Seed42": round(ari_vs_ref, 4),
            "Silhouette": round(sil, 4),
        })
    return pd.DataFrame(rows)


def test_feature_ablation(
    X_scaled_df: pd.DataFrame,
    n_clusters: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    cols = list(X_scaled_df.columns)
    base_km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, max_iter=400, random_state=random_state)
    base_lbl = base_km.fit_predict(X_scaled_df.values)
    base_sil = float(silhouette_score(X_scaled_df.values, base_lbl))

    rows = [{
        "Dropped_Feature": "NONE (Full Active Set)",
        "Remaining_Cols": len(cols),
        "ARI_vs_Full": 1.0,
        "Silhouette": round(base_sil, 4),
        "Delta_Silhouette": 0.0,
    }]

    for col in cols:
        if len(cols) - 1 < 2:
            continue
        sub_cols = [c for c in cols if c != col]
        X_sub = X_scaled_df[sub_cols].values
        km_sub = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, max_iter=400, random_state=random_state)
        lbl_sub = km_sub.fit_predict(X_sub)
        sil_sub = float(silhouette_score(X_sub, lbl_sub))
        ari_sub = float(adjusted_rand_score(base_lbl, lbl_sub))
        rows.append({
            "Dropped_Feature": col,
            "Remaining_Cols": len(sub_cols),
            "ARI_vs_Full": round(ari_sub, 4),
            "Silhouette": round(sil_sub, 4),
            "Delta_Silhouette": round(sil_sub - base_sil, 4),
        })

    return pd.DataFrame(rows)


def test_noise_sensitivity(
    X_scaled: np.ndarray,
    n_clusters: int = 2,
    noise_sigmas: list = None,
    random_state: int = 42,
) -> pd.DataFrame:
    if noise_sigmas is None:
        noise_sigmas = [0.02, 0.05, 0.10]

    km_clean = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, max_iter=400, random_state=random_state)
    lbl_clean = km_clean.fit_predict(X_scaled)

    rng = np.random.default_rng(random_state)
    rows = []
    for sig in noise_sigmas:
        noisy = X_scaled + rng.normal(loc=0.0, scale=sig, size=X_scaled.shape)
        km_n = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20, max_iter=400, random_state=random_state)
        lbl_n = km_n.fit_predict(noisy)
        ari_n = float(adjusted_rand_score(lbl_clean, lbl_n))
        sil_n = float(silhouette_score(noisy, lbl_n))
        rows.append({
            "Noise_Sigma": sig,
            "ARI_vs_Clean": round(ari_n, 4),
            "Noisy_Silhouette": round(sil_n, 4),
        })

    return pd.DataFrame(rows)


def run_phase10_reliability(X_scaled_df: pd.DataFrame, n_clusters: int = 2, random_state: int = 42):
    X_arr = X_scaled_df.values
    return {
        "seed_stability": test_seed_stability(X_arr, n_clusters=n_clusters, seeds=[42, 7, 123, 2026]),
        "feature_ablation": test_feature_ablation(X_scaled_df, n_clusters=n_clusters, random_state=random_state),
        "noise_sensitivity": test_noise_sensitivity(X_arr, n_clusters=n_clusters, noise_sigmas=[0.02, 0.05, 0.10], random_state=random_state),
    }


if __name__ == "__main__":
    prep = run_phase3_preprocessing(scaler_type="robust", prune_collinear=True, collinear_threshold=0.95)
    rel = run_phase10_reliability(prep["X_scaled"], n_clusters=2, random_state=42)

    print("=== PHASE 10.1 SEED STABILITY ===")
    print(rel["seed_stability"].to_string(index=False))

    print("\n=== PHASE 10.2 FEATURE ABLATION ===")
    print(rel["feature_ablation"].to_string(index=False))

    print("\n=== PHASE 10.3 NOISE SENSITIVITY ===")
    print(rel["noise_sensitivity"].to_string(index=False))