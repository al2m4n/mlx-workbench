import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Send,
  StopCircle,
  Plus,
  Trash2,
  Settings2,
  ChevronRight,
  ChevronDown,
  Brain,
} from "lucide-react";
import clsx from "clsx";

import { listLoadedModels, type ChatMessage } from "../api/client";
import { streamChat } from "../api/sse";
import { HarmonyParser } from "../api/harmony";
import LoadProgress from "../components/LoadProgress";
import { useModelLoadStream } from "../hooks/useModelLoadStream";
import { useStore, type ChatTurn } from "../store";

function nid(): string {
  return Math.random().toString(36).slice(2, 10);
}

export default function ChatPage() {
  const qc = useQueryClient();

  const {
    selectedModelKey,
    setSelectedModelKey,
    systemPrompt,
    setSystemPrompt,
    temperature,
    topP,
    maxTokens,
    setSampler,
    turns,
    appendTurn,
    patchTurn,
    clearTurns,
  } = useStore();

  const { data: models = [], isLoading: modelsLoading } = useQuery({
    queryKey: ["models"],
    queryFn: listLoadedModels,
    refetchInterval: 5000,
  });

  const llmModels = useMemo(
    () => models.filter((m) => m.modality === "llm"),
    [models],
  );

  // Auto-select the first loaded model if nothing selected
  useEffect(() => {
    if (!selectedModelKey && llmModels.length > 0) {
      setSelectedModelKey(llmModels[0].key);
    }
    if (selectedModelKey && !llmModels.find((m) => m.key === selectedModelKey)) {
      setSelectedModelKey(llmModels[0]?.key ?? null);
    }
  }, [llmModels, selectedModelKey, setSelectedModelKey]);

  const [input, setInput] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [showLoadForm, setShowLoadForm] = useState(false);
  const [newModelId, setNewModelId] = useState(
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
  );

  const abortRef = useRef<AbortController | null>(null);
  const [streaming, setStreaming] = useState(false);

  const { state: loadState, load: loadLLM, cancel: cancelLoad } =
    useModelLoadStream("llm");

  async function handleLoad(id: string) {
    const key = await loadLLM(id);
    if (key) {
      setSelectedModelKey(key);
      setShowLoadForm(false);
    }
  }

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns]);

  async function send() {
    if (!selectedModelKey || !input.trim() || streaming) return;

    const userTurn: ChatTurn = {
      id: nid(),
      role: "user",
      content: input.trim(),
    };
    const assistantTurn: ChatTurn = {
      id: nid(),
      role: "assistant",
      content: "",
      analysisOpen: true,
      pending: true,
    };
    appendTurn(userTurn);
    appendTurn(assistantTurn);
    setInput("");

    const messages: ChatMessage[] = [
      ...(systemPrompt.trim()
        ? [{ role: "system" as const, content: systemPrompt }]
        : []),
      ...turns
        .filter((t) => !t.pending && !t.error)
        .map((t) => ({ role: t.role, content: t.content })),
      { role: "user", content: userTurn.content },
    ];

    const ac = new AbortController();
    abortRef.current = ac;
    setStreaming(true);

    const parser = new HarmonyParser();
    let acc = "";
    let analysis = "";
    const apply = (segments: { channel: string; text: string }[]) => {
      for (const seg of segments) {
        if (seg.channel === "final") acc += seg.text;
        else if (seg.channel === "analysis") analysis += seg.text;
      }
      patchTurn(assistantTurn.id, {
        content: acc,
        analysis: analysis || undefined,
      });
    };

    try {
      await streamChat(
        {
          model_key: selectedModelKey,
          messages,
          max_tokens: maxTokens,
          temperature,
          top_p: topP,
        },
        {
          onToken: (text) => apply(parser.feed(text)),
          onDone: (meta) => {
            apply(parser.flush());
            patchTurn(assistantTurn.id, {
              content: acc,
              analysis: analysis || undefined,
              analysisOpen: false,
              meta,
              pending: false,
            });
          },
          onError: (msg) => {
            apply(parser.flush());
            patchTurn(assistantTurn.id, {
              content: acc,
              analysis: analysis || undefined,
              error: msg,
              pending: false,
            });
          },
        },
        ac.signal,
      );
    } finally {
      setStreaming(false);
      abortRef.current = null;
      qc.invalidateQueries({ queryKey: ["metrics"] });
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-ink-800 bg-ink-900/50 px-5 py-3 flex items-center gap-3">
        <div className="text-xs text-ink-400 uppercase tracking-widest">Model</div>
        <select
          value={selectedModelKey ?? ""}
          onChange={(e) => setSelectedModelKey(e.target.value || null)}
          disabled={modelsLoading || llmModels.length === 0}
          className="bg-ink-800 border border-ink-700 rounded-md px-2 py-1 text-sm font-mono min-w-[280px] focus:outline-none focus:border-accent-500 disabled:opacity-50"
        >
          {llmModels.length === 0 && <option value="">no LLM loaded</option>}
          {llmModels.map((m) => (
            <option key={m.key} value={m.key}>
              {m.model_id}
            </option>
          ))}
        </select>

        <button
          onClick={() => setShowLoadForm((v) => !v)}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md text-ink-200 hover:bg-ink-800 transition"
        >
          <Plus size={14} />
          Load
        </button>

        <button
          onClick={() => setShowSettings((v) => !v)}
          className={clsx(
            "ml-auto flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition",
            showSettings
              ? "bg-ink-800 text-ink-100"
              : "text-ink-300 hover:bg-ink-800/60",
          )}
        >
          <Settings2 size={14} />
          Sampler
        </button>

        <button
          onClick={clearTurns}
          disabled={turns.length === 0 || streaming}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md text-ink-300 hover:bg-ink-800/60 transition disabled:opacity-40"
        >
          <Trash2 size={14} />
          Clear
        </button>
      </div>

      {showLoadForm && (
        <div className="border-b border-ink-800 bg-ink-900/30 px-5 py-3">
          <div className="flex items-center gap-2">
            <input
              value={newModelId}
              onChange={(e) => setNewModelId(e.target.value)}
              placeholder="HuggingFace model id (e.g. mlx-community/...)"
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

      {showSettings && (
        <div className="border-b border-ink-800 bg-ink-900/30 px-5 py-3 grid grid-cols-2 gap-x-6 gap-y-3">
          <label className="text-xs text-ink-300">
            <div className="mb-1.5 flex justify-between">
              <span>System prompt</span>
            </div>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              rows={3}
              className="w-full bg-ink-800 border border-ink-700 rounded-md px-2 py-1.5 text-sm focus:outline-none focus:border-accent-500"
            />
          </label>
          <div className="space-y-3">
            <SliderRow
              label="Temperature"
              value={temperature}
              min={0}
              max={2}
              step={0.05}
              onChange={(v) => setSampler({ temperature: v })}
            />
            <SliderRow
              label="Top-p"
              value={topP}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => setSampler({ topP: v })}
            />
            <SliderRow
              label="Max tokens"
              value={maxTokens}
              min={64}
              max={8192}
              step={64}
              onChange={(v) => setSampler({ maxTokens: v })}
              format={(v) => v.toString()}
            />
          </div>
        </div>
      )}

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-5 py-6 space-y-5">
        {turns.length === 0 && (
          <div className="h-full min-h-[40vh] flex items-center justify-center text-center text-ink-400">
            <div className="space-y-2">
              <div className="text-sm">
                {selectedModelKey
                  ? "Send a message to start"
                  : "Load a model to start"}
              </div>
              {!selectedModelKey && (
                <div className="text-xs text-ink-500">
                  Try{" "}
                  <code className="font-mono text-ink-300">
                    mlx-community/Llama-3.2-1B-Instruct-4bit
                  </code>
                </div>
              )}
            </div>
          </div>
        )}
        {turns.map((t) => (
          <Turn key={t.id} turn={t} />
        ))}
      </div>

      {/* Composer */}
      <div className="border-t border-ink-800 bg-ink-900/50 px-5 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={
              selectedModelKey ? "Message…" : "Load a model first"
            }
            disabled={!selectedModelKey}
            rows={2}
            className="flex-1 bg-ink-800 border border-ink-700 rounded-md px-3 py-2 text-sm focus:outline-none focus:border-accent-500 resize-none disabled:opacity-50"
          />
          {streaming ? (
            <button
              onClick={stop}
              className="px-3 py-2 rounded-md bg-ink-700 text-ink-100 hover:bg-ink-600 transition flex items-center gap-1.5"
            >
              <StopCircle size={16} />
              Stop
            </button>
          ) : (
            <button
              onClick={send}
              disabled={!selectedModelKey || !input.trim()}
              className="px-3 py-2 rounded-md bg-accent-500 text-ink-950 font-medium hover:bg-accent-400 transition disabled:opacity-40 flex items-center gap-1.5"
            >
              <Send size={16} />
              Send
            </button>
          )}
        </div>
        <div className="mt-1.5 text-[11px] text-ink-500 font-mono">
          ⏎ send · ⇧⏎ newline
        </div>
      </div>
    </div>
  );
}

function Turn({ turn }: { turn: ChatTurn }) {
  const patchTurn = useStore((s) => s.patchTurn);
  const isUser = turn.role === "user";
  const hasAnalysis = !isUser && !!turn.analysis;
  // While streaming, show the placeholder only if we have neither final
  // content nor an in-progress analysis section.
  const showPlaceholder =
    !isUser && !turn.content && !turn.analysis && turn.pending;

  return (
    <div className={clsx("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={clsx(
          "max-w-[85%] rounded-lg px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-accent-500 text-ink-950 whitespace-pre-wrap"
            : "bg-ink-800 text-ink-100 border border-ink-700",
        )}
      >
        {hasAnalysis && (
          <div className="mb-2 -mx-1 rounded-md border border-ink-700/70 bg-ink-900/40">
            <button
              onClick={() =>
                patchTurn(turn.id, { analysisOpen: !turn.analysisOpen })
              }
              className="w-full flex items-center gap-1.5 px-2 py-1 text-[11px] uppercase tracking-widest text-ink-400 hover:text-ink-200 transition"
            >
              {turn.analysisOpen ? (
                <ChevronDown size={12} />
              ) : (
                <ChevronRight size={12} />
              )}
              <Brain size={12} />
              <span>Thinking</span>
              {turn.pending && (
                <span className="ml-1 text-ink-500 normal-case tracking-normal">
                  · live
                </span>
              )}
            </button>
            {turn.analysisOpen && (
              <div className="px-3 pb-2 pt-1 text-[12.5px] text-ink-400 italic whitespace-pre-wrap font-mono leading-snug">
                {turn.analysis}
              </div>
            )}
          </div>
        )}

        {!isUser ? (
          <div className="whitespace-pre-wrap">
            {turn.content ||
              (showPlaceholder && (
                <span className="text-ink-400 italic">…</span>
              ))}
          </div>
        ) : (
          turn.content
        )}

        {turn.error && (
          <div className="mt-2 text-xs text-red-300 border-t border-red-900/40 pt-2">
            {turn.error}
          </div>
        )}
        {turn.meta && !isUser && (
          <div className="mt-3 pt-2 border-t border-ink-700 flex flex-wrap gap-x-4 gap-y-1 text-[11px] font-mono text-ink-400">
            {typeof turn.meta.generation_tps === "number" && (
              <span>{turn.meta.generation_tps.toFixed(1)} tok/s</span>
            )}
            {typeof turn.meta.generation_tokens === "number" && (
              <span>{turn.meta.generation_tokens} tok</span>
            )}
            <span>{turn.meta.latency_ms.toFixed(0)} ms</span>
            {typeof turn.meta.peak_memory_mb === "number" && (
              <span>{turn.meta.peak_memory_mb.toFixed(0)} MB peak</span>
            )}
            {turn.meta.finish_reason && (
              <span className="text-ink-500">{turn.meta.finish_reason}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="block text-xs text-ink-300">
      <div className="mb-1 flex justify-between">
        <span>{label}</span>
        <span className="font-mono text-ink-200">
          {format ? format(value) : value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent-500"
      />
    </label>
  );
}
