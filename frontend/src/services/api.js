async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = "";
    try {
      const errBody = await res.json();
      detail = typeof errBody?.detail === "string" ? errBody.detail : JSON.stringify(errBody);
    } catch (_) {
      detail = await res.text().catch(() => "");
    }
    throw new Error(detail ? `Request failed (${res.status}): ${detail}` : `Request failed (${res.status})`);
  }
  return res.json();
}

export const getHealth = () => fetchJson("/api/health");

export const getDashboard = (scalerType = "robust") =>
  fetchJson(`/api/dashboard?scaler_type=${encodeURIComponent(scalerType)}`);

export const getDatasetOverview = (params = {}) => {
  const qs = new URLSearchParams({
    page: String(params.page || 1),
    page_size: String(params.page_size || 15),
    scaler_type: params.scaler_type || "robust",
  });
  return fetchJson(`/api/dataset?${qs.toString()}`);
};

export const getMovementAnalysis = (params = {}) => {
  const qs = new URLSearchParams({
    feature: params.feature || "mean_jerk_3d",
    bins: String(params.bins || 15),
    scaler_type: params.scaler_type || "robust",
  });
  return fetchJson(`/api/movement?${qs.toString()}`);
};

export const getMovement3D = (params = {}) => {
  const qs = new URLSearchParams({
    x_feature: params.x_feature || "spatial_rom_3d",
    y_feature: params.y_feature || "mean_speed_3d",
    z_feature: params.z_feature || "mean_jerk_3d",
    color_by: params.color_by || "Anomaly_Status",
    limit: String(params.limit || 1200),
    scaler_type: params.scaler_type || "robust",
  });
  return fetchJson(`/api/movement-3d?${qs.toString()}`);
};

export const getClustering = (params = {}) => {
  const qs = new URLSearchParams({
    projection: params.projection || "pca",
    color_by: params.color_by || "Cluster_KMeans",
    limit: String(params.limit || 1200),
    scaler_type: params.scaler_type || "robust",
  });
  return fetchJson(`/api/clustering?${qs.toString()}`);
};

export const getEvaluation = (arg = "robust") => {
  const scalerType =
    typeof arg === "string"
      ? arg
      : arg && typeof arg === "object" && arg.scaler_type
      ? arg.scaler_type
      : "robust";
  return fetchJson(`/api/evaluation?scaler_type=${encodeURIComponent(String(scalerType))}`);
};
export const api = {
  health: getHealth,
  getHealth,
  dashboard: getDashboard,
  getDashboard,
  dataset: getDatasetOverview,
  getDataset: getDatasetOverview,
  getDatasetOverview,
  movement: getMovementAnalysis,
  getMovement: getMovementAnalysis,
  getMovementAnalysis,
  movement3d: getMovement3D,
  getMovement3D,
  clustering: getClustering,
  getClustering,
  evaluation: getEvaluation,
  getEvaluation,
};

export default api;