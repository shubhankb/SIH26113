import React, { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

const PALETTE = ["#0d9488", "#dc2626", "#2563eb", "#d97706", "#7c3aed"];

export default function Movement3DPlot({
  points = [],
  xLabel = "spatial_rom_3d",
  yLabel = "mean_speed_3d",
  zLabel = "mean_jerk_3d",
}) {
  const plotRef = useRef(null);

  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;

    const grouped = new Map();
    for (const pt of points) {
      const key = pt.group ?? "Unknown";
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(pt);
    }

    const traces = Array.from(grouped.entries()).map(([groupName, rows], idx) => ({
      name: String(groupName),
      type: "scatter3d",
      mode: "markers",
      x: rows.map((r) => r.x),
      y: rows.map((r) => r.y),
      z: rows.map((r) => r.z),
      text: rows.map(
        (r) =>
          `${r.rep_id} | Cluster ${r.cluster} | ${r.anomaly_status} (sev: ${r.severity.toFixed(2)})`
      ),
      hovertemplate:
        `<b>%{text}</b><br>` +
        `${xLabel}: %{x:.4f}<br>` +
        `${yLabel}: %{y:.4f}<br>` +
        `${zLabel}: %{z:.4f}<extra></extra>`,
      marker: {
        size: 3.5,
        opacity: 0.78,
        color: PALETTE[idx % PALETTE.length],
      },
    }));

    const layout = {
      autosize: true,
      height: 460,
      margin: { l: 0, r: 0, t: 24, b: 0 },
      legend: { orientation: "h", y: 1.05 },
      scene: {
        xaxis: { title: xLabel },
        yaxis: { title: yLabel },
        zaxis: { title: zLabel },
      },
      paper_bgcolor: "#ffffff",
    };

    Plotly.react(el, traces, layout, { responsive: true, displaylogo: false });

    return () => {
      if (el) Plotly.purge(el);
    };
  }, [points, xLabel, yLabel, zLabel]);

  return <div ref={plotRef} className="w-full h-[460px]" />;
}