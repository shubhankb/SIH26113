import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const COLORS = ["#0d9488", "#059669", "#0284c7", "#d97706", "#64748b", "#475569"];

export function BarMetricChart({ data = [], xKey, yKey, fill = "#0d9488", height = 250 }) {
  if (!data.length) return <div className="text-xs text-slate-500 py-8 text-center">No chart data available.</div>;
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey={xKey} tick={{ fontSize: 11, fill: "#475569" }} />
          <YAxis tick={{ fontSize: 11, fill: "#475569" }} />
          <Tooltip contentStyle={{ fontSize: "12px", borderRadius: "6px", borderColor: "#cbd5e1" }} />
          <Bar dataKey={yKey} fill={fill} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ProjectionScatter({ points = [], xLabel = "x", yLabel = "y", onSelect, height = 340 }) {
  if (!points.length) return <div className="text-xs text-slate-500 py-8 text-center">No projection points available.</div>;
  const groups = Array.from(new Set(points.map((p) => p.group)));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 10, right: 14, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis type="number" dataKey="x" name={xLabel} tick={{ fontSize: 11, fill: "#475569" }} />
          <YAxis type="number" dataKey="y" name={yLabel} tick={{ fontSize: 11, fill: "#475569" }} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
            contentStyle={{ fontSize: "12px", borderRadius: "6px", borderColor: "#cbd5e1" }}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          {groups.map((grp, i) => (
            <Scatter
              key={grp}
              name={`${grp}`}
              data={points.filter((p) => p.group === grp)}
              fill={COLORS[i % COLORS.length]}
              onClick={(entry) => onSelect && onSelect(entry?.payload || entry)}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}