import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { ErrorBox, LoadingBox } from "../components/StatusBanner";
import { BarMetricChart } from "../components/Charts";

export default function EvaluationPage({ scalerType }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.evaluation({ scaler_type: scalerType });
      setData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [scalerType]);

  if (loading) return <LoadingBox text="Loading evaluation metrics..." />;
  if (error) return <ErrorBox message={error} onRetry={load} />;
  if (!data) return null;

  const scorecard = data.scorecard || [];
  const cols = scorecard.length ? Object.keys(scorecard[0]) : [];

  return (
    <div className="space-y-5">
      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-slate-900 mb-3">Comparative Clustering Scorecard</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-left">
                {cols.map((c) => (
                  <th key={c} className="p-2">{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {scorecard.map((row, i) => (
                <tr key={i} className="border-b border-slate-100">
                  {cols.map((c) => (
                    <td key={c} className="p-2">{row[c] === null || row[c] === undefined ? "—" : String(row[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">PCA Explained Variance Ratio</h2>
          <BarMetricChart data={data.scree || []} xKey="PC" yKey="Explained_Variance_Ratio" fill="#0d9488" height={240} />
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">Feature Ablation (Silhouette)</h2>
          <BarMetricChart data={data.feature_ablation || []} xKey="Dropped_Feature" yKey="Silhouette" fill="#059669" height={240} />
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-slate-900 mb-3">Top Flagged Anomalies (IF + LOF)</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-left">
                <th className="p-2">Rank</th>
                <th className="p-2">Rep_ID</th>
                <th className="p-2">Cluster</th>
                <th className="p-2">Status</th>
                <th className="p-2 text-right">Severity</th>
                <th className="p-2">Driving Features</th>
              </tr>
            </thead>
            <tbody>
              {(data.anomalies_top || []).slice(0, 12).map((a) => (
                <tr key={a.Rep_ID} className="border-b border-slate-100">
                  <td className="p-2">{a.Severity_Rank}</td>
                  <td className="p-2 font-medium">{a.Rep_ID}</td>
                  <td className="p-2">{a.Cluster}</td>
                  <td className="p-2">{a.Status}</td>
                  <td className="p-2 text-right">{a.Severity_Score}</td>
                  <td className="p-2">{a.Top_Driving_Features}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}