import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Trash2, Cpu, RefreshCw } from "lucide-react";

import {
  fetchHealthInfo,
  listLoadedModels,
  unloadModel,
} from "../api/client";
import LoadProgress from "../components/LoadProgress";
import { useModelLoadStream } from "../hooks/useModelLoadStream";

export default function ModelsPage() {
  const qc = useQueryClient();
  const { data: models = [], isFetching } = useQuery({
    queryKey: ["models"],
    queryFn: listLoadedModels,
    refetchInterval: 5000,
  });
  const { data: info } = useQuery({
    queryKey: ["health-info"],
    queryFn: fetchHealthInfo,
  });

  const [newModelId, setNewModelId] = useState(
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
  );

  const { state: loadState, load: loadLLM, cancel: cancelLoad } =
    useModelLoadStream("llm");

  const unloadMut = useMutation({
    mutationFn: (key: string) => unloadModel(key),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["models"] }),
  });

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-mono text-ink-100">Loaded models</h1>
          <p className="text-sm text-ink-400 mt-1">
            Models held in memory by the inference server.
          </p>
        </div>
        {info && (
          <div className="text-xs font-mono text-ink-400 text-right">
            <div>mlx {info.mlx_version}</div>
            <div>{info.processor}</div>
          </div>
        )}
      </header>

      <section className="rounded-lg border border-ink-800 bg-ink-900/40 p-4">
        <div className="text-xs uppercase tracking-widest text-ink-400 mb-2">
          Load LLM
        </div>
        <div className="flex gap-2">
          <input
            value={newModelId}
            onChange={(e) => setNewModelId(e.target.value)}
            placeholder="mlx-community/..."
            className="flex-1 bg-ink-800 border border-ink-700 rounded-md px-3 py-2 text-sm font-mono focus:outline-none focus:border-accent-500"
          />
          <button
            onClick={() => loadLLM(newModelId.trim())}
            disabled={!newModelId.trim() || loadState.loading}
            className="px-4 py-2 text-sm rounded-md bg-accent-500 text-ink-950 font-medium hover:bg-accent-400 transition disabled:opacity-50"
          >
            {loadState.loading ? "Loading…" : "Load"}
          </button>
        </div>
        <LoadProgress state={loadState} onCancel={cancelLoad} />
        <div className="mt-2 text-[11px] text-ink-500">
          MLX-format weights only. Browse{" "}
          <a
            href="https://huggingface.co/mlx-community"
            target="_blank"
            rel="noreferrer"
            className="text-accent-400 hover:underline"
          >
            mlx-community
          </a>{" "}
          on the Hub.
        </div>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="text-xs uppercase tracking-widest text-ink-400">
            In memory ({models.length})
          </div>
          <RefreshCw
            size={12}
            className={isFetching ? "text-accent-400 animate-spin" : "text-ink-600"}
          />
        </div>
        {models.length === 0 ? (
          <div className="text-sm text-ink-500 italic px-4 py-6 rounded-lg border border-dashed border-ink-800">
            No models loaded.
          </div>
        ) : (
          <ul className="space-y-2">
            {models.map((m) => (
              <li
                key={m.key}
                className="flex items-center gap-3 rounded-lg border border-ink-800 bg-ink-900/40 px-4 py-3"
              >
                <Cpu size={16} className="text-accent-400 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-mono truncate">{m.model_id}</div>
                  <div className="text-[11px] font-mono text-ink-500 uppercase tracking-wider">
                    {m.modality}
                  </div>
                </div>
                <button
                  onClick={() => unloadMut.mutate(m.key)}
                  disabled={unloadMut.isPending}
                  className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md text-ink-300 hover:bg-ink-800 hover:text-red-300 transition disabled:opacity-50"
                >
                  <Trash2 size={14} />
                  Unload
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
