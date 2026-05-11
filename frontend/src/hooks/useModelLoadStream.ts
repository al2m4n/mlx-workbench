import { useCallback } from "react";
import { useQueryClient, type QueryClient } from "@tanstack/react-query";

import {
  streamLoad,
  type LoadModality,
  type LoadProgressEvent,
} from "../api/client";
import { useStore, type ModelLoadState } from "../store";

export type { ModelLoadState } from "../store";

// Module-scoped state so loads survive page navigation. Each modality has at
// most one in-flight load (the backend serializes everything through a single
// asyncio lock anyway).
type ActiveLoad = {
  controller: AbortController;
  promise: Promise<string | null>;
};
const active: Partial<Record<LoadModality, ActiveLoad>> = {};

function runLoad(
  modality: LoadModality,
  model_id: string,
  qc: QueryClient,
): Promise<string | null> {
  active[modality]?.controller.abort();

  const ac = new AbortController();
  const { patchLoad, resetLoad } = useStore.getState();

  patchLoad(modality, {
    loading: true,
    modelId: model_id,
    phase: undefined,
    file: undefined,
    percent: undefined,
    error: undefined,
  });

  let resultKey: string | null = null;
  let errorMsg: string | null = null;

  const handle = (ev: LoadProgressEvent) => {
    if (ev.type === "phase") {
      const patch: Partial<ModelLoadState> = { phase: ev.phase };
      if (ev.phase === "loading_weights") {
        patch.percent = undefined;
        patch.file = undefined;
      }
      patchLoad(modality, patch);
    } else if (ev.type === "progress") {
      patchLoad(modality, {
        phase: "downloading",
        file: ev.file,
        percent: ev.percent,
      });
    } else if (ev.type === "done") {
      resultKey = ev.model_key;
      qc.invalidateQueries({ queryKey: ["models"] });
      resetLoad(modality);
    } else if (ev.type === "error") {
      errorMsg = ev.message;
      patchLoad(modality, { loading: false, error: ev.message });
    }
  };

  const promise = (async () => {
    try {
      await streamLoad(modality, model_id, handle, ac.signal);
    } finally {
      if (active[modality]?.controller === ac) delete active[modality];
    }
    return errorMsg ? null : resultKey;
  })();

  active[modality] = { controller: ac, promise };
  return promise;
}

function cancelLoad(modality: LoadModality): void {
  active[modality]?.controller.abort();
  delete active[modality];
  useStore.getState().resetLoad(modality);
}

export function useModelLoadStream(modality: LoadModality) {
  const qc = useQueryClient();
  const state = useStore((s) => s.loads[modality]) as ModelLoadState;

  const load = useCallback(
    (model_id: string) => runLoad(modality, model_id, qc),
    [modality, qc],
  );
  const cancel = useCallback(() => cancelLoad(modality), [modality]);

  return { state, load, cancel };
}
