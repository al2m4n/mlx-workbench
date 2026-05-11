import { create } from "zustand";

import type { DonePayload, LoadModality } from "../api/client";

export type LoadPhase = "downloading" | "loading_weights";

export type ModelLoadState = {
  loading: boolean;
  modelId?: string;
  phase?: LoadPhase;
  file?: string;
  percent?: number;
  error?: string;
};

const IDLE_LOAD: ModelLoadState = { loading: false };

export type ChatTurn = {
  id: string;
  role: "user" | "assistant";
  content: string;
  analysis?: string;
  analysisOpen?: boolean;
  meta?: DonePayload;
  pending?: boolean;
  error?: string;
};

type Store = {
  selectedModelKey: string | null;
  setSelectedModelKey: (key: string | null) => void;

  systemPrompt: string;
  setSystemPrompt: (s: string) => void;

  temperature: number;
  topP: number;
  maxTokens: number;
  setSampler: (s: { temperature?: number; topP?: number; maxTokens?: number }) => void;

  turns: ChatTurn[];
  appendTurn: (t: ChatTurn) => void;
  patchTurn: (id: string, patch: Partial<ChatTurn>) => void;
  clearTurns: () => void;

  loads: Record<LoadModality, ModelLoadState>;
  patchLoad: (modality: LoadModality, patch: Partial<ModelLoadState>) => void;
  resetLoad: (modality: LoadModality) => void;
};

export const useStore = create<Store>((set) => ({
  selectedModelKey: null,
  setSelectedModelKey: (key) => set({ selectedModelKey: key }),

  systemPrompt: "You are a helpful assistant running locally via MLX on Apple Silicon.",
  setSystemPrompt: (s) => set({ systemPrompt: s }),

  temperature: 0.7,
  topP: 0.9,
  maxTokens: 1024,
  setSampler: (s) =>
    set((state) => ({
      temperature: s.temperature ?? state.temperature,
      topP: s.topP ?? state.topP,
      maxTokens: s.maxTokens ?? state.maxTokens,
    })),

  turns: [],
  appendTurn: (t) => set((state) => ({ turns: [...state.turns, t] })),
  patchTurn: (id, patch) =>
    set((state) => ({
      turns: state.turns.map((t) => (t.id === id ? { ...t, ...patch } : t)),
    })),
  clearTurns: () => set({ turns: [] }),

  loads: { llm: IDLE_LOAD, audio: IDLE_LOAD, video: IDLE_LOAD },
  patchLoad: (modality, patch) =>
    set((state) => ({
      loads: {
        ...state.loads,
        [modality]: { ...state.loads[modality], ...patch },
      },
    })),
  resetLoad: (modality) =>
    set((state) => ({ loads: { ...state.loads, [modality]: IDLE_LOAD } })),
}));
