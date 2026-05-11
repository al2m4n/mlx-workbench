import { Loader2, X } from "lucide-react";

import type { ModelLoadState } from "../hooks/useModelLoadStream";

export default function LoadProgress({
  state,
  onCancel,
}: {
  state: ModelLoadState;
  onCancel?: () => void;
}) {
  if (!state.loading && !state.error) return null;

  if (state.error) {
    return <div className="mt-2 text-xs text-red-400">{state.error}</div>;
  }

  const cancel =
    onCancel && state.loading ? (
      <button
        onClick={onCancel}
        title="Cancel (download continues in background)"
        className="shrink-0 p-1 rounded text-ink-400 hover:text-red-300 hover:bg-ink-800 transition"
      >
        <X size={12} />
      </button>
    ) : null;

  if (state.phase === "loading_weights") {
    return (
      <div className="mt-3 flex items-center gap-2 text-xs font-mono text-ink-300">
        <Loader2 size={12} className="animate-spin text-accent-400" />
        <span className="flex-1">Loading weights…</span>
        {cancel}
      </div>
    );
  }

  const percent = Math.max(0, Math.min(100, state.percent ?? 0));
  return (
    <div className="mt-3 space-y-1">
      <div className="flex items-center gap-2 text-[11px] font-mono text-ink-400">
        <span className="flex-1 truncate">
          {state.file || "preparing download…"}
        </span>
        <span className="shrink-0">
          {state.phase === "downloading" ? `${percent.toFixed(1)}%` : ""}
        </span>
        {cancel}
      </div>
      <div className="h-1 w-full rounded-full bg-ink-800 overflow-hidden">
        <div
          className="h-full bg-accent-500 transition-[width] duration-150"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
