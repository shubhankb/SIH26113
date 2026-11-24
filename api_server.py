from pathlib import Path
import sys
import threading
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "phase2_kinematic_features.csv"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline_v2 import run_full_ml_pipeline  # noqa: E402
from phase3_preprocessing import run_phase3_preprocessing  # noqa: E402

app = FastAPI(title="MovementAI INT396 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOCK = threading.Lock()
_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cleaned = df.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
    return cleaned.to_dict(orient="records")


def _get_bundle(
    scaler_type: str = "robust",
    prune_collinear: bool = True,
    contamination: float = 0.07,
    lof_neighbors: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    if not DEFAULT_CSV.exists():
        raise FileNotFoundError(f"Dataset not found at {DEFAULT_CSV}")

    key = (
        str(DEFAULT_CSV),
        scaler_type,
        bool(prune_collinear),
        round(float(contamination), 4),
        int(lof_neighbors),
        int(random_state),
    )

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

    bundle = run_full_ml_pipeline(
        phase2_csv=str(DEFAULT_CSV),
        scaler_type=scaler_type,
        prune_collinear=prune_collinear,
        collinear_threshold=0.95,
        contamination=float(contamination),
        lof_neighbors=int(lof_neighbors),
        random_state=int(random_state),
    )
    raw_prep = run_phase3_preprocessing(
        input_csv=str(DEFAULT_CSV),
        scaler_type=scaler_type,
        prune_collinear=False,
        collinear_threshold=0.95,
    )
    payload = {**bundle, "raw_prep": raw_prep}

    with _LOCK:
        _CACHE[key] = payload
    return payload


@app.get("/api/health")
def health():
    return {"status": "ok", "dataset_present": DEFAULT_CSV.exists()}


@app.get("/api/dashboard")
def dashboard(
    scaler_type: str = Query("robust"),
    contamination: float = Query(0.07, ge=0.01, le=0.20),
):
    try:
        b = _get_bundle(scaler_type=scaler_type, contamination=contamination)
        res = b["results_df"]
        prof = b["profile_summary"]
        anom = b["anomaly_summary"]

        cluster_dist = (
            res["Cluster_KMeans"]
            .value_counts()
            .reset_index()
            .rename(columns={"Cluster_KMeans": "cluster", "count": "count"})
        )
        anomaly_dist = (
            res["Anomaly_Status"]
            .value_counts()
            .reset_index()
            .rename(columns={"Anomaly_Status": "status", "count": "count"})
        )

        return {
            "total_records": int(len(res)),
            "active_features": b["prep"]["active_features"],
            "feature_count": int(len(b["prep"]["active_features"])),
            "cluster_count": int(len(prof)) if not prof.empty else 0,
            "consensus_anomalies": int(anom.get("consensus_flagged", 0)),
            "total_flagged": int(anom.get("any_flagged", 0)),
            "insights": b["insights"],
            "profiles": _to_records(prof),
            "cluster_distribution": _to_records(cluster_dist),
            "anomaly_distribution": _to_records(anomaly_dist),
            "top_anomalies": _to_records(b["anomalies_top"].head(8)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/dataset")
def dataset_overview(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    scaler_type: str = Query("robust"),
):
    try:
        b = _get_bundle(scaler_type=scaler_type)
        X_sel = b["prep"]["X_selected"]
        raw_X = b["raw_prep"]["X_selected"]
        merged = pd.concat(
            [
                b["results_df"][["Rep_ID", "Cluster_KMeans", "Anomaly_Status", "Anomaly_Severity"]],
                X_sel.reset_index(drop=True),
            ],
            axis=1,
        )
        start = (page - 1) * page_size
        end = start + page_size

        stats = X_sel.describe().T.reset_index().rename(columns={"index": "feature"})
        for c in ["count", "mean", "std", "min", "50%", "max"]:
            if c in stats.columns:
                stats[c] = stats[c].round(5)

        return {
            "rows": int(len(X_sel)),
            "columns": int(len(X_sel.columns)),
            "feature_names": list(X_sel.columns),
            "pruned_features": b["prep"].get("pruned_features", []),
            "missing_after_imputation": int(X_sel.isna().sum().sum()),
            "raw_columns_count": int(len(raw_X.columns)),
            "page": page,
            "page_size": page_size,
            "total_rows": int(len(merged)),
            "preview_rows": _to_records(merged.iloc[start:end]),
            "statistics": _to_records(stats),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/movement")
def movement_analysis(
    feature: str = Query("mean_jerk_3d"),
    bins: int = Query(15, ge=5, le=30),
    scaler_type: str = Query("robust"),
):
    try:
        b = _get_bundle(scaler_type=scaler_type)
        X_sel = b["prep"]["X_selected"]
        cols = list(X_sel.columns)
        target = feature if feature in cols else cols[0]

        vals = pd.to_numeric(X_sel[target], errors="coerce").dropna().values
        counts, edges = np.histogram(vals, bins=bins)
        histogram = [
            {
                "range": f"{edges[i]:.3f}–{edges[i+1]:.3f}",
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        corr = X_sel.corr(method="pearson").fillna(0.0).round(3)
        corr_rows = []
        for r_col in cols:
            row_obj = {"feature": r_col}
            for c_col in cols:
                row_obj[c_col] = float(corr.loc[r_col, c_col])
            corr_rows.append(row_obj)

        return {
            "features": cols,
            "selected_feature": target,
            "histogram": histogram,
            "correlation_rows": corr_rows,
            "cluster_medians": _to_records(b["profile_median"].reset_index()),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/clustering")
def clustering_viz(
    projection: str = Query("pca"),
    color_by: str = Query("Cluster_KMeans"),
    limit: int = Query(1000, ge=100, le=2577),
    scaler_type: str = Query("robust"),
):
    try:
        b = _get_bundle(scaler_type=scaler_type)
        df = b["results_df"].copy()
        if len(df) > limit:
            step = max(1, len(df) // limit)
            df = df.iloc[::step].head(limit)

        proj = projection.lower()
        x_col, y_col = (
            ("UMAP_1", "UMAP_2")
            if proj == "umap"
            else (("TSNE_1", "TSNE_2") if proj == "tsne" else ("PCA_1", "PCA_2"))
        )
        allowed = {"Cluster_KMeans", "Cluster_GMM", "Cluster_DBSCAN", "Cluster_Hierarchical", "Anomaly_Status"}
        c_col = color_by if color_by in allowed else "Cluster_KMeans"

        pts = [
            {
                "rep_id": str(r["Rep_ID"]),
                "x": float(r[x_col]),
                "y": float(r[y_col]),
                "group": str(r[c_col]),
                "anomaly_status": str(r["Anomaly_Status"]),
                "severity": float(r["Anomaly_Severity"]),
                "top_driving_features": str(r["Top_Driving_Features"]),
            }
            for _, r in df.iterrows()
            if np.isfinite(r[x_col]) and np.isfinite(r[y_col])
        ]

        return {
            "projection": proj,
            "x_label": x_col,
            "y_label": y_col,
            "color_by": c_col,
            "points": pts,
            "profiles": _to_records(b["profile_summary"]),
            "tuned_params": b["tuned_params"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/evaluation")
def model_evaluation(scaler_type: str = Query("robust")):
    try:
        b = _get_bundle(scaler_type=scaler_type)
        rel = b["reliability"]
        return {
            "scorecard": _to_records(b["scorecard"]),
            "scree": _to_records(b["scree"]),
            "fidelity": _to_records(b["fidelity"]),
            "seed_stability": _to_records(rel["seed_stability"]),
            "feature_ablation": _to_records(rel["feature_ablation"]),
            "anomalies_top": _to_records(b["anomalies_top"].head(20)),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/movement-3d")
def movement_3d(
    x_feature: str = Query("spatial_rom_3d"),
    y_feature: str = Query("mean_speed_3d"),
    z_feature: str = Query("mean_jerk_3d"),
    color_by: str = Query("Anomaly_Status"),
    limit: int = Query(1200, ge=100, le=2577),
    scaler_type: str = Query("robust"),
):
    try:
        b = _get_bundle(scaler_type=scaler_type)
        X_sel = b["prep"]["X_selected"].reset_index(drop=True)
        res = b["results_df"].reset_index(drop=True)
        cols = list(X_sel.columns)

        xf = x_feature if x_feature in cols else cols[0]
        yf = y_feature if y_feature in cols else cols[min(1, len(cols) - 1)]
        zf = z_feature if z_feature in cols else cols[min(2, len(cols) - 1)]

        merged = pd.concat(
            [
                res[["Rep_ID", "Cluster_KMeans", "Anomaly_Status", "Anomaly_Severity"]],
                X_sel[[xf, yf, zf]],
            ],
            axis=1,
        )

        if len(merged) > limit:
            step = max(1, len(merged) // limit)
            merged = merged.iloc[::step].head(limit)

        allowed_color = {"Anomaly_Status", "Cluster_KMeans"}
        c_col = color_by if color_by in allowed_color else "Anomaly_Status"

        points = []
        for _, r in merged.iterrows():
            xv, yv, zv = r[xf], r[yf], r[zf]
            if pd.notna(xv) and pd.notna(yv) and pd.notna(zv):
                points.append(
                    {
                        "rep_id": str(r["Rep_ID"]),
                        "x": float(xv),
                        "y": float(yv),
                        "z": float(zv),
                        "group": str(r[c_col]),
                        "cluster": str(r["Cluster_KMeans"]),
                        "anomaly_status": str(r["Anomaly_Status"]),
                        "severity": float(r["Anomaly_Severity"]) if pd.notna(r["Anomaly_Severity"]) else 0.0,
                    }
                )

        return {
            "features": cols,
            "x_feature": xf,
            "y_feature": yf,
            "z_feature": zf,
            "color_by": c_col,
            "points": points,
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())