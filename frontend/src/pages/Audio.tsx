import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Mic, Upload, Plus } from "lucide-react";
import clsx from "clsx";

import {
  listLoadedModels,
  transcribe,
  type TranscribeResult,
} from "../api/client";
import LoadProgress from "../components/LoadProgress";
import { useModelLoadStream } from "../hooks/useModelLoadStream";

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function AudioPage() {
  const qc = useQueryClient();
  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: listLoadedModels,
    refetchInterval: 5000,
  });
  const audioModels = useMemo(
    () => models.filter((m) => m.modality === "audio"),
    [models],
  );

  const [selected, setSelected] = useState<string>("");
  const effectiveSelected = selected || audioModels[0]?.key || "";

  const [showLoad, setShowLoad] = useState(false);
  const [newModelId, setNewModelId] = useState("mlx-community/whisper-tiny");

  const { state: loadState, load: loadAudio, cancel: cancelLoad } =
    useModelLoadStream("audio");

  async function handleLoad(id: string) {
    const key = await loadAudio(id);
    if (key) {
      setSelected(key);
      setShowLoad(false);
    }
  }

  const [file, setFile] = useState<File | null>(null);
  const [language, setLanguage] = useState<string>("");
  const [task, setTask] = useState<"transcribe" | "translate">("transcribe");
  const [wordTs, setWordTs] = useState(false);
  const [result, setResult] = useState<TranscribeResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const transMut = useMutation({
    mutationFn: () =>
      transcribe({
        model_key: effectiveSelected,
        file: file!,
        language: language.trim() || undefined,
        task,
        word_timestamps: wordTs,
      }),
    onSuccess: (data) => {
      setResult(data);
      qc.invalidateQueries({ queryKey: ["metrics"] });
    },
  });

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-mono text-ink-100">Audio · Whisper</h1>
          <p className="text-sm text-ink-400 mt-1">
            Speech-to-text via mlx-whisper.
          </p>
        </div>
      </header>

      <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
        <div className="text-xs uppercase tracking-widest text-ink-400">
          Model
        </div>
        <div className="flex items-center gap-2">
          <select
            value={effectiveSelected}
            onChange={(e) => setSelected(e.target.value)}
            disabled={audioModels.length === 0}
            className="flex-1 bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-accent-500 disabled:opacity-50"
          >
            {audioModels.length === 0 && (
              <option value="">no audio model loaded</option>
            )}
            {audioModels.map((m) => (
              <option key={m.key} value={m.key}>
                {m.model_id}
              </option>
            ))}
          </select>
          <button
            onClick={() => setShowLoad((v) => !v)}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md text-ink-200 hover:bg-ink-800 transition"
          >
            <Plus size={14} />
            Load
          </button>
        </div>
        {showLoad && (
          <div className="pt-2 border-t border-ink-800">
            <div className="flex gap-2">
              <input
                value={newModelId}
                onChange={(e) => setNewModelId(e.target.value)}
                placeholder="HuggingFace whisper-mlx model id"
                className="flex-1 bg-ink-800 border border-ink-700 rounded-md px-3 py-1.5 text-sm font-mono focus:outline-none focus:border-accent-500"
              />
              <button
                onClick={() => handleLoad(newModelId.trim())}
                disabled={!newModelId.trim() || loadState.loading}
                className="px-3 py-1.5 text-sm rounded-md bg-accent-500 text-ink-950 font-medium hover:bg-accent-400 transition disabled:opacity-50"
              >
                {loadState.loading ? "Loading…" : "Load"}
              </button>
            </div>
            <LoadProgress state={loadState} onCancel={cancelLoad} />
          </div>
        )}
        <div className="text-[11px] text-ink-500">
          Try{" "}
          <code className="font-mono text-ink-300">
            mlx-community/whisper-tiny
          </code>{" "}
          or{" "}
          <code className="font-mono text-ink-300">
            mlx-community/whisper-large-v3-turbo
          </code>
          .
        </div>
      </section>

      <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
        <div className="text-xs uppercase tracking-widest text-ink-400">
          Input
        </div>
        <div
          className={clsx(
            "rounded-md border-2 border-dashed border-ink-700 p-6 text-center cursor-pointer transition",
            file ? "bg-ink-800/40" : "hover:border-accent-500/60",
          )}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files?.[0];
            if (f) setFile(f);
          }}
        >
          <Upload size={20} className="mx-auto text-ink-400 mb-2" />
          {file ? (
            <div className="space-y-1">
              <div className="text-sm font-mono text-ink-100">{file.name}</div>
              <div className="text-[11px] text-ink-500">
                {(file.size / 1024).toFixed(1)} KB · {file.type || "audio"}
              </div>
            </div>
          ) : (
            <div className="text-sm text-ink-300">
              Drop or click to choose an audio file
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*,video/*"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>

        <div className="grid grid-cols-3 gap-3">
          <label className="text-xs text-ink-300">
            <div className="mb-1">Language (auto if blank)</div>
            <input
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              placeholder="en, de, ja, …"
              className="w-full bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-accent-500"
            />
          </label>
          <label className="text-xs text-ink-300">
            <div className="mb-1">Task</div>
            <select
              value={task}
              onChange={(e) =>
                setTask(e.target.value as "transcribe" | "translate")
              }
              className="w-full bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-accent-500"
            >
              <option value="transcribe">Transcribe</option>
              <option value="translate">Translate to English</option>
            </select>
          </label>
          <label className="text-xs text-ink-300 flex items-end">
            <span className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={wordTs}
                onChange={(e) => setWordTs(e.target.checked)}
                className="accent-accent-500"
              />
              Word timestamps
            </span>
          </label>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => transMut.mutate()}
            disabled={!file || !effectiveSelected || transMut.isPending}
            className="flex items-center gap-2 px-4 py-2 rounded-md bg-accent-500 text-ink-950 font-medium hover:bg-accent-400 transition disabled:opacity-50"
          >
            <Mic size={16} />
            {transMut.isPending ? "Transcribing…" : "Transcribe"}
          </button>
          {transMut.isError && (
            <div className="text-xs text-red-400">
              {(transMut.error as Error).message}
            </div>
          )}
        </div>
      </section>

      {result && (
        <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-widest text-ink-400">
              Result
            </div>
            <div className="flex gap-4 text-[11px] font-mono text-ink-400">
              <span>{result.language ?? "—"}</span>
              <span>{result.duration_s.toFixed(1)}s audio</span>
              <span>{result.latency_ms.toFixed(0)} ms</span>
              {result.duration_s > 0 && result.latency_ms > 0 && (
                <span>
                  {(result.duration_s / (result.latency_ms / 1000)).toFixed(1)}×
                  realtime
                </span>
              )}
            </div>
          </div>
          <div className="text-sm leading-relaxed whitespace-pre-wrap text-ink-100">
            {result.text || (
              <span className="italic text-ink-500">no speech detected</span>
            )}
          </div>
          {result.segments.length > 1 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-ink-400 hover:text-ink-200">
                Segments ({result.segments.length})
              </summary>
              <div className="mt-2 space-y-1 font-mono">
                {result.segments.map((s, i) => (
                  <div key={i} className="flex gap-3">
                    <span className="text-ink-500 shrink-0 w-24">
                      {fmtTime(s.start)} → {fmtTime(s.end)}
                    </span>
                    <span className="text-ink-200">{s.text}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </section>
      )}
    </div>
  );
}
