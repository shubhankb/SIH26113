import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import StatCard from "../components/StatCard";
import { ErrorBox, LoadingBox } from "../components/StatusBanner";

export default function DatasetPage({ scalerType }) {
  const [data, setData] = useState(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async (targetPage = page) => {
    setLoading(true);
    setError("");
    try {
      const res = await api.dataset({ page: targetPage, page_size: 15, scaler_type: scalerType });
      setData(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load(page);
  }, [page, scalerType]);

  if (loading) return <LoadingBox text="Loading dataset overview..." />;
  if (error) return <ErrorBox message={error} onRetry={() => load(page)} />;
  if (!data) return null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Repetitions (Rows)" value={data.rows.toLocaleString()} hint="phase2_kinematic_features.csv" />
        <StatCard title="Active Features" value={data.columns} hint={`Raw extracted: ${data.raw_columns_count}`} />
        <StatCard title="Pruned Collinear" value={(data.pruned_features || []).join(", ") || "None"} hint="|r| >= 0.95 threshold" />
        <StatCard title="Missing Values" value={data.missing_after_imputation} hint="After median imputation" />
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-slate-900 mb-3">Feature Summary Statistics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-left">
                <th className="p-2">Feature</th>
                <th className="p-2 text-right">Mean</th>
                <th className="p-2 text-right">Median</th>
                <th className="p-2 text-right">Std</th>
                <th className="p-2 text-right">Min</th>
                <th className="p-2 text-right">Max</th>
              </tr>
            </thead>
            <tbody>
              {(data.statistics || []).map((row) => (
                <tr key={row.feature} className="border-b border-slate-100">
                  <td className="p-2 font-medium">{row.feature}</td>
                  <td className="p-2 text-right">{row.mean}</td>
                  <td className="p-2 text-right">{row["50%"]}</td>
                  <td className="p-2 text-right">{row.std}</td>
                  <td className="p-2 text-right">{row.min}</td>
                  <td className="p-2 text-right">{row.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900">Records Preview</h2>
          <div className="flex items-center gap-2 text-xs">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-2.5 py-1 border border-slate-300 rounded disabled:opacity-40"
            >
              Prev
            </button>
            <span>Page {page}</span>
            <button
              type="button"
              disabled={page * 15 >= data.total_rows}
              onClick={() => setPage((p) => p + 1)}
              className="px-2.5 py-1 border border-slate-300 rounded disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-left">
                <th className="p-2">Rep_ID</th>
                <th className="p-2">Cluster</th>
                <th className="p-2">Anomaly_Status</th>
                {(data.feature_names || []).map((f) => (
                  <th key={f} className="p-2 text-right">{f}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.preview_rows || []).map((r, i) => (
                <tr key={r.Rep_ID || i} className="border-b border-slate-100">
                  <td className="p-2 font-medium">{r.Rep_ID}</td>
                  <td className="p-2">{r.Cluster_KMeans}</td>
                  <td className="p-2">{r.Anomaly_Status}</td>
                  {(data.feature_names || []).map((f) => (
                    <td key={f} className="p-2 text-right">
                      {typeof r[f] === "number" ? r[f].toFixed(4) : r[f]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}