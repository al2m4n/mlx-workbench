import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Image as ImageIcon, Plus, Upload, Wand2 } from "lucide-react";
import clsx from "clsx";

import {
  listLoadedModels,
  visionGenerate,
  type VisionResult,
} from "../api/client";
import LoadProgress from "../components/LoadProgress";
import { useModelLoadStream } from "../hooks/useModelLoadStream";

const IMAGE_EXTS = /\.(png|jpe?g|webp|gif|bmp)$/i;
const VIDEO_EXTS = /\.(mp4|mov|webm|mkv|avi|m4v)$/i;

export default function VideoPage() {
  const qc = useQueryClient();
  const { data: models = [] } = useQuery({
    queryKey: ["models"],
    queryFn: listLoadedModels,
    refetchInterval: 5000,
  });
  const visionModels = useMemo(
    () => models.filter((m) => m.modality === "video"),
    [models],
  );

  const [selected, setSelected] = useState<string>("");
  const effective = selected || visionModels[0]?.key || "";

  const [showLoad, setShowLoad] = useState(false);
  const [newModelId, setNewModelId] = useState(
    "mlx-community/SmolVLM-256M-Instruct-bf16",
  );

  const { state: loadState, load: loadVision, cancel: cancelLoad } =
    useModelLoadStream("video");

  async function handleLoad(id: string) {
    const key = await loadVision(id);
    if (key) {
      setSelected(key);
      setShowLoad(false);
    }
  }

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("Describe this in detail.");
  const [maxTokens, setMaxTokens] = useState(256);
  const [result, setResult] = useState<VisionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const isImage = file && IMAGE_EXTS.test(file.name);
  const isVideo = file && VIDEO_EXTS.test(file.name);

  const genMut = useMutation({
    mutationFn: () =>
      visionGenerate({
        model_key: effective,
        prompt,
        file: file!,
        max_tokens: maxTokens,
      }),
    onSuccess: (data) => {
      setResult(data);
      qc.invalidateQueries({ queryKey: ["metrics"] });
    },
  });

  return (
    <div className="h-full overflow-y-auto p-6 space-y-5">
      <header>
        <h1 className="text-xl font-mono text-ink-100">Vision · VLM</h1>
        <p className="text-sm text-ink-400 mt-1">
          Image and video understanding via mlx-vlm.
        </p>
      </header>

      <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
        <div className="text-xs uppercase tracking-widest text-ink-400">
          Model
        </div>
        <div className="flex items-center gap-2">
          <select
            value={effective}
            onChange={(e) => setSelected(e.target.value)}
            disabled={visionModels.length === 0}
            className="flex-1 bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-accent-500 disabled:opacity-50"
          >
            {visionModels.length === 0 && (
              <option value="">no vision model loaded</option>
            )}
            {visionModels.map((m) => (
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
                placeholder="mlx-community/<vlm-model-id>"
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
            mlx-community/SmolVLM-256M-Instruct-bf16
          </code>{" "}
          (small) or{" "}
          <code className="font-mono text-ink-300">
            mlx-community/Qwen2-VL-2B-Instruct-4bit
          </code>{" "}
          (better).
        </div>
      </section>

      <section className="grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
          <div className="text-xs uppercase tracking-widest text-ink-400">
            Media
          </div>
          <div
            className={clsx(
              "rounded-md border-2 border-dashed border-ink-700 p-6 text-center cursor-pointer transition min-h-[180px] flex items-center justify-center",
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
            {previewUrl && isImage ? (
              <img
                src={previewUrl}
                alt={file?.name}
                className="max-h-[260px] rounded"
              />
            ) : previewUrl && isVideo ? (
              <video
                src={previewUrl}
                controls
                className="max-h-[260px] rounded"
              />
            ) : (
              <div className="space-y-2 text-ink-400">
                <Upload size={20} className="mx-auto" />
                <div className="text-sm">
                  Drop or click to choose an image or video
                </div>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*,video/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>
          {file && (
            <div className="text-[11px] font-mono text-ink-500 truncate">
              {file.name} · {(file.size / 1024).toFixed(1)} KB
            </div>
          )}
        </div>

        <div className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
          <div className="text-xs uppercase tracking-widest text-ink-400">
            Prompt
          </div>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={5}
            className="w-full bg-ink-800 border border-ink-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent-500 resize-none"
          />
          <label className="block text-xs text-ink-300">
            <div className="mb-1 flex justify-between">
              <span>Max tokens</span>
              <span className="font-mono text-ink-200">{maxTokens}</span>
            </div>
            <input
              type="range"
              min={64}
              max={2048}
              step={64}
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-accent-500"
            />
          </label>
          <button
            onClick={() => genMut.mutate()}
            disabled={!file || !effective || !prompt.trim() || genMut.isPending}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-accent-500 text-ink-950 font-medium hover:bg-accent-400 transition disabled:opacity-50"
          >
            <Wand2 size={16} />
            {genMut.isPending ? "Generating…" : "Generate"}
          </button>
          {genMut.isError && (
            <div className="text-xs text-red-400">
              {(genMut.error as Error).message}
            </div>
          )}
        </div>
      </section>

      {result && (
        <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-widest text-ink-400 flex items-center gap-2">
              <ImageIcon size={12} />
              Response
            </div>
            <div className="flex gap-4 text-[11px] font-mono text-ink-400">
              <span>{result.media_kind}</span>
              <span>{result.generation_tps.toFixed(1)} tok/s</span>
              <span>{result.generation_tokens} tok</span>
              <span>{result.latency_ms.toFixed(0)} ms</span>
              <span>{result.peak_memory_mb.toFixed(0)} MB peak</span>
            </div>
          </div>
          <div className="text-sm leading-relaxed whitespace-pre-wrap text-ink-100">
            {result.text}
          </div>
        </section>
      )}
    </div>
  );
}
