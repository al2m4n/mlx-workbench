import axios from "axios";

import { streamSSE } from "./sse";

export const api = axios.create({
  baseURL: "/api",
  timeout: 0,
});

export type LoadModality = "llm" | "audio" | "video";

export type LoadProgressEvent =
  | { type: "phase"; phase: "downloading" | "loading_weights" }
  | {
      type: "progress";
      file: string;
      downloaded: number;
      total: number;
      percent: number;
    }
  | { type: "done"; model_key: string; cached: boolean }
  | { type: "error"; message: string };

export type LoadedModel = {
  key: string;
  model_id: string;
  modality: "llm" | "audio" | "video";
};

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type DonePayload = {
  model_key: string;
  latency_ms: number;
  prompt_tokens?: number;
  generation_tokens?: number;
  prompt_tps?: number;
  generation_tps?: number;
  peak_memory_mb?: number;
  finish_reason?: string | null;
};

export type InferenceMetric = {
  ts: number;
  modality: string;
  model_key: string;
  prompt_tokens: number;
  generation_tokens: number;
  prompt_tps: number;
  generation_tps: number;
  latency_ms: number;
  peak_memory_mb: number;
  finish_reason: string | null;
  extra: Record<string, unknown> | null;
};

export type AudioSegment = { start: number; end: number; text: string };

export type TranscribeResult = {
  text: string;
  language: string | null;
  duration_s: number;
  latency_ms: number;
  segments: AudioSegment[];
};

export type VisionResult = {
  text: string;
  model_key: string;
  latency_ms: number;
  prompt_tokens: number;
  generation_tokens: number;
  prompt_tps: number;
  generation_tps: number;
  peak_memory_mb: number;
  media_kind: "image" | "video";
};

export async function listLoadedModels(): Promise<LoadedModel[]> {
  const { data } = await api.get<LoadedModel[]>("/models/");
  return data;
}

export async function loadLLM(model_id: string): Promise<{ model_key: string }> {
  const { data } = await api.post("/llm/load", { model_id });
  return data;
}

export async function unloadModel(key: string): Promise<void> {
  await api.delete(`/models/${encodeURIComponent(key)}`);
}

export async function fetchMetrics(
  model_key?: string,
  limit = 100,
): Promise<InferenceMetric[]> {
  const { data } = await api.get<InferenceMetric[]>("/llm/metrics", {
    params: { model_key, limit },
  });
  return data;
}

export async function loadAudio(model_id: string): Promise<{ model_key: string }> {
  const { data } = await api.post("/audio/load", { model_id });
  return data;
}

export async function transcribe(opts: {
  model_key: string;
  file: File;
  language?: string;
  task?: "transcribe" | "translate";
  word_timestamps?: boolean;
}): Promise<TranscribeResult> {
  const fd = new FormData();
  fd.append("model_key", opts.model_key);
  fd.append("file", opts.file);
  if (opts.language) fd.append("language", opts.language);
  fd.append("task", opts.task ?? "transcribe");
  fd.append("word_timestamps", String(opts.word_timestamps ?? false));
  const { data } = await api.post<TranscribeResult>("/audio/transcribe", fd);
  return data;
}

export async function loadVision(model_id: string): Promise<{ model_key: string }> {
  const { data } = await api.post("/video/load", { model_id });
  return data;
}

export async function visionGenerate(opts: {
  model_key: string;
  prompt: string;
  file: File;
  max_tokens?: number;
  temperature?: number;
}): Promise<VisionResult> {
  const fd = new FormData();
  fd.append("model_key", opts.model_key);
  fd.append("prompt", opts.prompt);
  fd.append("file", opts.file);
  fd.append("max_tokens", String(opts.max_tokens ?? 512));
  fd.append("temperature", String(opts.temperature ?? 0.4));
  const { data } = await api.post<VisionResult>("/video/generate", fd);
  return data;
}

export async function fetchHealthInfo() {
  const { data } = await api.get("/health/info");
  return data as {
    python: string;
    platform: string;
    processor: string;
    mlx_version: string;
  };
}

const LOAD_PATHS: Record<LoadModality, string> = {
  llm: "/api/llm/load-stream",
  audio: "/api/audio/load-stream",
  video: "/api/video/load-stream",
};

export async function streamLoad(
  modality: LoadModality,
  model_id: string,
  onEvent: (ev: LoadProgressEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await streamSSE(
    LOAD_PATHS[modality],
    { model_id },
    {
      onEvent: ({ event, data }) => {
        if (event === "phase" && data && typeof data === "object" && "phase" in data) {
          const phase = (data as { phase: string }).phase;
          if (phase === "downloading" || phase === "loading_weights") {
            onEvent({ type: "phase", phase });
          }
        } else if (event === "progress" && data && typeof data === "object") {
          const d = data as {
            file?: string;
            downloaded?: number;
            total?: number;
            percent?: number;
          };
          onEvent({
            type: "progress",
            file: d.file ?? "",
            downloaded: d.downloaded ?? 0,
            total: d.total ?? 0,
            percent: d.percent ?? 0,
          });
        } else if (event === "done" && data && typeof data === "object") {
          const d = data as { model_key?: string; cached?: boolean };
          onEvent({
            type: "done",
            model_key: d.model_key ?? "",
            cached: d.cached ?? false,
          });
        } else if (event === "error" && data && typeof data === "object") {
          onEvent({
            type: "error",
            message: (data as { message?: string }).message ?? "Unknown error",
          });
        }
      },
      onError: (message) => onEvent({ type: "error", message }),
    },
    signal,
  );
}
