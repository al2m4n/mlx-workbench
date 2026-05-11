import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";

import {
  fetchMetrics,
  listLoadedModels,
  type InferenceMetric,
} from "../api/client";

function fmtTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

function formatExtra(m: InferenceMetric): string {
  if (m.modality === "audio" && m.extra) {
    const dur = m.extra.audio_duration_s as number | undefined;
    const lang = m.extra.language as string | undefined;
    const rtf = m.extra.realtime_factor as number | undefined;
    return [
      lang,
      dur ? `${dur.toFixed(1)}s` : null,
      rtf ? `${rtf.toFixed(1)}× rt` : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }
  if (m.modality === "video" && m.extra) {
    return (m.extra.media_kind as string) ?? "";
  }
  return m.finish_reason ?? "—";
}

export default function MetricsPage() {
  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: listLoadedModels,
  });
  const [filter, setFilter] = useState<string>("");

  const { data: metrics = [], isFetching } = useQuery({
    queryKey: ["metrics", filter],
    queryFn: () => fetchMetrics(filter || undefined, 200),
    refetchInterval: 3000,
  });

  const summary = useMemo(() => {
    if (metrics.length === 0) return null;
    const tps = metrics
      .map((m) => m.generation_tps)
      .filter((v) => v > 0);
    const lat = metrics.map((m) => m.latency_ms);
    const avg = (xs: number[]) =>
      xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
    return {
      count: metrics.length,
      avgTps: avg(tps),
      avgLatency: avg(lat),
      totalTokens: metrics.reduce((s, m) => s + m.generation_tokens, 0),
    };
  }, [metrics]);

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-mono text-ink-100">Inference metrics</h1>
          <p className="text-sm text-ink-400 mt-1">
            Per-request stats from the in-process buffer (last 1000).
          </p>
        </div>
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1 text-sm font-mono focus:outline-none focus:border-accent-500"
        >
          <option value="">all models</option>
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.model_id}
            </option>
          ))}
        </select>
      </header>

      {summary && (
        <div className="grid grid-cols-4 gap-3">
          <Stat label="Requests" value={summary.count.toString()} />
          <Stat
            label="Avg gen tps"
            value={summary.avgTps.toFixed(1)}
            unit="tok/s"
          />
          <Stat
            label="Avg latency"
            value={summary.avgLatency.toFixed(0)}
            unit="ms"
          />
          <Stat
            label="Tokens generated"
            value={summary.totalTokens.toString()}
          />
        </div>
      )}

      <section className="rounded-lg border border-ink-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-ink-900 text-[11px] uppercase tracking-wider text-ink-400">
            <tr>
              <th className="text-left px-3 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Mod</th>
              <th className="text-left px-3 py-2 font-medium">Model</th>
              <th className="text-right px-3 py-2 font-medium">Prompt tok</th>
              <th className="text-right px-3 py-2 font-medium">Gen tok</th>
              <th className="text-right px-3 py-2 font-medium">Prompt tps</th>
              <th className="text-right px-3 py-2 font-medium">Gen tps</th>
              <th className="text-right px-3 py-2 font-medium">Latency</th>
              <th className="text-right px-3 py-2 font-medium">Peak MB</th>
              <th className="text-left px-3 py-2 font-medium">Notes</th>
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {metrics.length === 0 && (
              <tr>
                <td
                  colSpan={10}
                  className="text-center py-8 text-ink-500 italic"
                >
                  {isFetching ? "loading…" : "no metrics yet"}
                </td>
              </tr>
            )}
            {metrics.map((m, i) => (
              <tr
                key={i}
                className="border-t border-ink-800 hover:bg-ink-900/40"
              >
                <td className="px-3 py-1.5 text-ink-400">{fmtTime(m.ts)}</td>
                <td className="px-3 py-1.5">
                  <span
                    className={clsx(
                      "px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider",
                      m.modality === "llm" && "bg-accent-500/20 text-accent-300",
                      m.modality === "audio" && "bg-amber-500/20 text-amber-300",
                      m.modality === "video" && "bg-emerald-500/20 text-emerald-300",
                    )}
                  >
                    {m.modality}
                  </span>
                </td>
                <td className="px-3 py-1.5 truncate max-w-[260px]">
                  {m.model_key.replace(/^(llm|audio|video)::/, "")}
                </td>
                <td className="px-3 py-1.5 text-right">{m.prompt_tokens || "—"}</td>
                <td className="px-3 py-1.5 text-right">{m.generation_tokens || "—"}</td>
                <td className="px-3 py-1.5 text-right">
                  {m.prompt_tps > 0 ? m.prompt_tps.toFixed(1) : "—"}
                </td>
                <td className="px-3 py-1.5 text-right text-accent-400">
                  {m.generation_tps > 0 ? m.generation_tps.toFixed(1) : "—"}
                </td>
                <td className="px-3 py-1.5 text-right">
                  {m.latency_ms.toFixed(0)}
                </td>
                <td className="px-3 py-1.5 text-right">
                  {m.peak_memory_mb > 0 ? m.peak_memory_mb.toFixed(0) : "—"}
                </td>
                <td className="px-3 py-1.5 text-ink-400">
                  {formatExtra(m)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function Stat({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit?: string;
}) {
  return (
    <div className="rounded-lg border border-ink-800 bg-ink-900/40 px-4 py-3">
      <div className="text-[11px] uppercase tracking-widest text-ink-400">
        {label}
      </div>
      <div className="mt-1 font-mono">
        <span className="text-2xl text-ink-100">{value}</span>
        {unit && <span className="ml-1 text-xs text-ink-400">{unit}</span>}
      </div>
    </div>
  );
}
