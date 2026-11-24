// frontend/src/App.jsx
import React, { useState } from "react";
import { motion } from "framer-motion";
import { Activity, BarChart2, CheckSquare, Database, Layers, Menu } from "lucide-react";
import DashboardPage from "./pages/DashboardPage";
import DatasetPage from "./pages/DatasetPage";
import MovementPage from "./pages/MovementPage";
import ClusteringPage from "./pages/ClusteringPage";
import EvaluationPage from "./pages/EvaluationPage";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: Activity },
  { id: "dataset", label: "Dataset Overview", icon: Database },
  { id: "movement", label: "Movement Analysis", icon: BarChart2 },
  { id: "clustering", label: "Cluster Visualization", icon: Layers },
  { id: "evaluation", label: "Model Evaluation", icon: CheckSquare },
];

export default function App() {
  const [page, setPage] = useState("dashboard");
  const [scalerType, setScalerType] = useState("robust");
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const currentNav = NAV.find((n) => n.id === page) || NAV[0];

  return (
    <div className="min-h-screen flex bg-[#f8fafb] text-slate-900">
      <aside
        className={`${
          sidebarOpen ? "w-60" : "w-16"
        } shrink-0 bg-white border-r border-slate-200 transition-all duration-150 flex flex-col justify-between`}
      >
        <div>
          <div className="h-14 px-4 border-b border-slate-200 flex items-center justify-between">
            {sidebarOpen && (
              <div>
                <div className="text-sm font-semibold text-slate-900">MovementAI</div>
                <div className="text-[11px] text-teal-700">INT396 Unsupervised ML</div>
              </div>
            )}
            <button
              type="button"
              onClick={() => setSidebarOpen((s) => !s)}
              className="p-1.5 rounded-md text-slate-500 hover:bg-slate-100"
              aria-label="Toggle navigation"
            >
              <Menu className="w-4 h-4" />
            </button>
          </div>

          <nav className="p-2 space-y-1">
            {NAV.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => setPage(id)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium text-left transition-colors ${
                  page === id
                    ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <Icon className="w-4 h-4 shrink-0" />
                {sidebarOpen && <span className="truncate">{label}</span>}
              </button>
            ))}
          </nav>
        </div>

        {sidebarOpen && (
          <div className="p-3 border-t border-slate-200">
            <label className="block text-[11px] font-medium text-slate-500 mb-1">Feature Scaler</label>
            <select
              value={scalerType}
              onChange={(e) => setScalerType(e.target.value)}
              className="w-full px-2.5 py-1.5 text-xs border border-slate-300 rounded-md bg-white"
            >
              <option value="robust">RobustScaler</option>
              <option value="standard">StandardScaler</option>
            </select>
          </div>
        )}
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        <header className="h-14 bg-white border-b border-slate-200 px-6 flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-900">{currentNav.label}</div>
          <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-600" />
            FastAPI Connected
          </span>
        </header>

        <motion.main
          key={page}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.16 }}
          className="p-6 max-w-7xl w-full mx-auto"
        >
          {page === "dashboard" && <DashboardPage scalerType={scalerType} />}
          {page === "dataset" && <DatasetPage scalerType={scalerType} />}
          {page === "movement" && <MovementPage scalerType={scalerType} />}
          {page === "clustering" && <ClusteringPage scalerType={scalerType} />}
          {page === "evaluation" && <EvaluationPage scalerType={scalerType} />}
        </motion.main>
      </div>
    </div>
  );
}