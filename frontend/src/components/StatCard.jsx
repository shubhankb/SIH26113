import React from "react";

export default function StatCard({ title, value, hint, icon: Icon }) {
  return (
    <div className="bg-white border border-slate-200 border-t-2 border-t-emerald-600 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-500">{title}</span>
        {Icon && <Icon className="w-4 h-4 text-teal-700" />}
      </div>
      <div className="mt-1.5 text-2xl font-semibold text-slate-900">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}