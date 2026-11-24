import React, { useEffect, useState } from "react";
import { BarMetricChart } from "../components/Charts";
import Movement3DPlot from "../components/Movement3DPlot";
import { ErrorBox, LoadingBox } from "../components/StatusBanner";
import { getMovement3D, getMovementAnalysis } from "../services/api";

export default function MovementPage({ scalerType }) {
  const [feature, setFeature] = useState("mean_jerk_3d");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [xFeature, setXFeature] = useState("spatial_rom_3d");
  const [yFeature, setYFeature] = useState("mean_speed_3d");
  const [zFeature, setZFeature] = useState("mean_jerk_3d");
  const [colorBy, setColorBy] = useState("Anomaly_Status");
  const [plot3D, setPlot3D] = useState(null);
  const [loading3D, setLoading3D] = useState(true);
  const [error3D, setError3D] = useState("");

  const load = (targetFeature = feature) => {
    setLoading(true);
    setError("");
    getMovementAnalysis({ feature: targetFeature, bins: 15, scaler_type: scalerType })
      .then((res) => {
        setData(res);
        if (res?.selected_feature && res.selected_feature !== targetFeature) {
          setFeature(res.selected_feature);
        }
      })
      .catch((err) => setError(err.message || "Failed to load movement analysis"))
      .finally(() => setLoading(false));
  };

  const load3D = (
    xf = xFeature,
    yf = yFeature,
    zf = zFeature,
    cb = colorBy
  ) => {
    setLoading3D(true);
    setError3D("");
    getMovement3D({
      x_feature: xf,
      y_feature: yf,
      z_feature: zf,
      color_by: cb,
      limit: 1200,
      scaler_type: scalerType,
    })
      .then((res) => {
        setPlot3D(res);
        if (Array.isArray(res?.features) && res.features.length >= 3) {
          if (res.x_feature) setXFeature(res.x_feature);
          if (res.y_feature) setYFeature(res.y_feature);
          if (res.z_feature) setZFeature(res.z_feature);
        }
      })
      .catch((err) => setError3D(err.message || "Failed to load 3D movement plot"))
      .finally(() => setLoading3D(false));
  };

  useEffect(() => {
    load(feature);
  }, [scalerType, feature]);

  useEffect(() => {
    load3D(xFeature, yFeature, zFeature, colorBy);
  }, [scalerType, xFeature, yFeature, zFeature, colorBy]);

  if (loading && !data) return <LoadingBox label="Analyzing movement kinematics..." />;
  if (error && !data) return <ErrorBox message={error} onRetry={() => load(feature)} />;

  const features = data?.features || [];
  const corrRows = data?.correlation_rows || [];
  const feature3DOptions = plot3D?.features?.length ? plot3D.features : features;

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Kinematic Feature Distribution
            </h3>
            <p className="text-xs text-slate-500">Histogram across 2,577 repetitions</p>
          </div>

          <select
            value={feature}
            onChange={(e) => setFeature(e.target.value)}
            className="border border-slate-300 rounded-xl px-3 py-2 text-sm text-slate-800 bg-white"
          >
            {features.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>

        <BarMetricChart
          data={data?.histogram || []}
          xKey="range"
          yKey="count"
          color="#0d9488"
          height={290}
        />
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              3D Movement Feature Space
            </h3>
            <p className="text-xs text-slate-500">
              Interactive 3D view across three kinematic dimensions (rotate, zoom, and hover repetitions)
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="text-xs text-slate-600 flex items-center gap-1.5">
              <span>X:</span>
              <select
                value={xFeature}
                onChange={(e) => setXFeature(e.target.value)}
                className="border border-slate-300 rounded-xl px-2.5 py-1.5 text-xs text-slate-800 bg-white"
              >
                {feature3DOptions.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs text-slate-600 flex items-center gap-1.5">
              <span>Y:</span>
              <select
                value={yFeature}
                onChange={(e) => setYFeature(e.target.value)}
                className="border border-slate-300 rounded-xl px-2.5 py-1.5 text-xs text-slate-800 bg-white"
              >
                {feature3DOptions.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs text-slate-600 flex items-center gap-1.5">
              <span>Z:</span>
              <select
                value={zFeature}
                onChange={(e) => setZFeature(e.target.value)}
                className="border border-slate-300 rounded-xl px-2.5 py-1.5 text-xs text-slate-800 bg-white"
              >
                {feature3DOptions.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-xs text-slate-600 flex items-center gap-1.5">
              <span>Color:</span>
              <select
                value={colorBy}
                onChange={(e) => setColorBy(e.target.value)}
                className="border border-slate-300 rounded-xl px-2.5 py-1.5 text-xs text-slate-800 bg-white"
              >
                <option value="Anomaly_Status">Anomaly_Status</option>
                <option value="Cluster_KMeans">Cluster_KMeans</option>
              </select>
            </label>
          </div>
        </div>

        {loading3D && !plot3D ? (
          <LoadingBox label="Rendering 3D movement space..." />
        ) : error3D && !plot3D ? (
          <ErrorBox message={error3D} onRetry={() => load3D()} />
        ) : (
          <Movement3DPlot
            points={plot3D?.points || []}
            xLabel={plot3D?.x_feature || xFeature}
            yLabel={plot3D?.y_feature || yFeature}
            zLabel={plot3D?.z_feature || zFeature}
          />
        )}
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm">
        <h3 className="text-base font-semibold text-slate-900 mb-3">
          Feature Correlation Matrix (Pearson)
        </h3>
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="text-left py-2.5 px-3 font-semibold">Feature</th>
                {features.map((col) => (
                  <th key={col} className="text-right py-2.5 px-3 font-semibold">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {corrRows.map((row) => (
                <tr key={row.feature} className="border-b border-slate-100">
                  <td className="py-2 px-3 font-medium text-slate-800">{row.feature}</td>
                  {features.map((col) => {
                    const val = Number(row[col] ?? 0);
                    const strong = col !== row.feature && Math.abs(val) >= 0.7;
                    return (
                      <td
                        key={col}
                        className={`py-2 px-3 text-right tabular-nums ${
                          strong ? "font-semibold text-teal-700 bg-teal-50/60" : "text-slate-700"
                        }`}
                      >
                        {val.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}