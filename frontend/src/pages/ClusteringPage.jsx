import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { ErrorBox, LoadingBox } from "../components/StatusBanner";
import { ProjectionScatter } from "../components/Charts";

export default function ClusteringPage({ scalerType }) {
  const [projection, setProjection] = useState("pca");
  const [colorBy, setColorBy] = useState("Cluster_KMeans");
  const [selected, setSelected] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.clustering({
        projection,
        color_by: colorBy,
        limit: 1000,
        scaler_type: scalerType,
      });
      setData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [projection, colorBy, scalerType]);

  if (loading) return <LoadingBox text="Loading cluster projections..." />;
  if (error) return <ErrorBox message={error} onRetry={load} />;
  if (!data) return null;

  return (
    <div className="space-y-5">
      <div className="bg-white border border-slate-200 rounded-lg p-4 flex flex-wrap gap-4">
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Projection</label>
          <select
            value={projection}
            onChange={(e) => setProjection(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-300 rounded-md bg-white"
          >
            <option value="pca">PCA</option>
            <option value="umap">UMAP</option>
            <option value="tsne">t-SNE</option>
          </select>
        </div>
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-1">Color By</label>
          <select
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value)}
            className="px-3 py-1.5 text-xs border border-slate-300 rounded-md bg-white"
          >
            <option value="Cluster_KMeans">Cluster_KMeans</option>
            <option value="Cluster_GMM">Cluster_GMM</option>
            <option value="Cluster_DBSCAN">Cluster_DBSCAN</option>
            <option value="Cluster_Hierarchical">Cluster_Hierarchical</option>
            <option value="Anomaly_Status">Anomaly_Status</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 bg-white border border-slate-200 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-slate-900 mb-2">
            {data.projection.toUpperCase()} Scatter ({data.color_by})
          </h2>
          <ProjectionScatter
            points={data.points}
            xLabel={data.x_label}
            yLabel={data.y_label}
            onSelect={setSelected}
            height={360}
          />
        </div>

        <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold text-slate-900">Point & Cluster Details</h2>
          {selected ? (
            <div className="p-3 rounded-md bg-slate-50 border border-slate-200 text-xs space-y-1">
              <div><span className="font-semibold">Rep ID:</span> {selected.rep_id}</div>
              <div><span className="font-semibold">Group:</span> {selected.group}</div>
              <div><span className="font-semibold">Anomaly:</span> {selected.anomaly_status}</div>
              <div><span className="font-semibold">Severity:</span> {selected.severity}</div>
              <div><span className="font-semibold">Drivers:</span> {selected.top_driving_features}</div>
            </div>
          ) : (
            <p className="text-xs text-slate-500">Click any scatter point to inspect repetition details.</p>
          )}

          <div className="space-y-2">
            {(data.profiles || []).map((p) => (
              <div key={p.Cluster} className="p-2.5 rounded border border-slate-200 text-xs">
                <div className="font-semibold text-slate-900">
                  Cluster {p.Cluster} — {p.Count} ({p["Share_%"]}%)
                </div>
                <div className="text-slate-600 mt-0.5">{p.Observed_Characteristics}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}