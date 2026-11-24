const BASE = "/api";

async function getJson(path, params = {}) {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null) qs.set(k, String(v));
  });
  const query = qs.toString();
  const res = await fetch(`${BASE}${path}${query ? `?${query}` : ""}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  health: () => getJson("/health"),
  dashboard: (params) => getJson("/dashboard", params),
  dataset: (params) => getJson("/dataset", params),
  movement: (params) => getJson("/movement", params),
  clustering: (params) => getJson("/clustering", params),
  evaluation: (params) => getJson("/evaluation", params),
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