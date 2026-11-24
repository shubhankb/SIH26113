import React, { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";

const PALETTE = ["#0d9488", "#dc2626", "#2563eb", "#d97706", "#7c3aed"];

export default function Movement3DPlot({
  points = [],
  xLabel = "spatial_rom_3d",
  yLabel = "mean_speed_3d",
  zLabel = "mean_jerk_3d",
}) {
  const plotRef = useRef(null);
  const [plotError, setPlotError] = useState("");

  useEffect(() => {
    const el = plotRef.current;
    const plotlyLib = Plotly?.react ? Plotly : window?.Plotly;
    if (!el) return;

    if (!plotlyLib?.react) {
      setPlotError("Plotly 3D library failed to initialize in browser.");
      return;
    }

    try {
      setPlotError("");
      const grouped = new Map();
      for (const pt of Array.isArray(points) ? points : []) {
        const key = String(pt?.group ?? "Unknown");
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(pt);
      }

      const traces = Array.from(grouped.entries()).map(([groupName, rows], idx) => ({
        name: groupName,
        type: "scatter3d",
        mode: "markers",
        x: rows.map((r) => Number(r.x)),
        y: rows.map((r) => Number(r.y)),
        z: rows.map((r) => Number(r.z)),
        text: rows.map(
          (r) =>
            `${r.rep_id || "Rep"} | Cluster ${r.cluster ?? "-"} | ${r.anomaly_status ?? "-"}`
        ),
        hovertemplate:
          `<b>%{text}</b><br>` +
          `${xLabel}: %{x:.4f}<br>` +
          `${yLabel}: %{y:.4f}<br>` +
          `${zLabel}: %{z:.4f}<extra></extra>`,
        marker: {
          size: 3.5,
          opacity: 0.8,
          color: PALETTE[idx % PALETTE.length],
        },
      }));

      const layout = {
        autosize: true,
        height: 460,
        margin: { l: 0, r: 0, t: 24, b: 0 },
        legend: { orientation: "h", y: 1.05 },
        scene: {
          xaxis: { title: { text: xLabel } },
          yaxis: { title: { text: yLabel } },
          zaxis: { title: { text: zLabel } },
        },
        paper_bgcolor: "#ffffff",
      };

      plotlyLib.react(el, traces, layout, { responsive: true, displaylogo: false });
    } catch (err) {
      setPlotError(err?.message || "3D plot rendering error");
    }

    return () => {
      try {
        if (el && plotlyLib?.purge) plotlyLib.purge(el);
      } catch (_) {}
    };
  }, [points, xLabel, yLabel, zLabel]);

  if (plotError) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
        3D Plot Note: {plotError}
      </div>
    );
  }

  return <div ref={plotRef} className="w-full h-[460px]" />;
}