import React, { useEffect, useState } from "react";
import { Activity, AlertTriangle, GitMerge, Layers } from "lucide-react";
import { api } from "../services/api";
import StatCard from "../components/StatCard";
import { ErrorBox, LoadingBox } from "../components/StatusBanner";
import { BarMetricChart } from "../components/Charts";

export default function DashboardPage({ scalerType }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.dashboard({ scaler_type: scalerType });
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

  if (loading) return <LoadingBox text="Loading dashboard metrics..." />;
  if (error) return <ErrorBox message={error} onRetry={load} />;
  if (!data) return null;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          title="Movement Records"
          value={data.total_records.toLocaleString()}
          hint="Unlabeled 3D repetitions"
          icon={Activity}
        />
        <StatCard
          title="Engineered Features"
          value={data.feature_count}
          hint={data.active_features.join(", ")}
          icon={GitMerge}
        />
        <StatCard
          title="Detected Clusters"
          value={data.cluster_count}
          hint="K-Means profile groups"
          icon={Layers}
        />
        <StatCard
          title="Consensus Anomalies"
          value={data.consensus_anomalies}
          hint={`${data.total_flagged} total flagged (IF + LOF)`}
          icon={AlertTriangle}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">Cluster Distribution</h2>
          <BarMetricChart data={data.cluster_distribution} xKey="cluster" yKey="count" fill="#0d9488" />
        </div>
        <div className="bg-white border border-slate-200 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-slate-900 mb-3">Anomaly Detection Breakdown</h2>
          <BarMetricChart data={data.anomaly_distribution} xKey="status" yKey="count" fill="#059669" />
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-lg p-4">
        <h2 className="text-sm font-semibold text-slate-900 mb-3">Pipeline Findings</h2>
        <ul className="space-y-2 text-xs text-slate-700">
          {(data.insights || []).map((line, i) => (
            <li key={i} className="p-2.5 rounded-md bg-slate-50 border border-slate-200/70">
              {line}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}