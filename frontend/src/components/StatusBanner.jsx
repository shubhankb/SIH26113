import React from "react";
import { AlertCircle, Loader2, RotateCcw } from "lucide-react";

export function LoadingBox({ text = "Loading pipeline output..." }) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-8 flex items-center justify-center gap-2.5 text-sm text-slate-600">
      <Loader2 className="w-4 h-4 animate-spin text-teal-600" />
      <span>{text}</span>
    </div>
  );
}

export function ErrorBox({ message, onRetry }) {
  return (
    <div className="bg-white border border-rose-200 rounded-lg p-5 text-sm text-rose-800 flex items-center justify-between gap-4">
      <div className="flex items-center gap-2">
        <AlertCircle className="w-4 h-4 shrink-0" />
        <span>{message}</span>
      </div>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-900 text-white hover:bg-slate-800"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Retry
        </button>
      )}
    </div>
  );
}