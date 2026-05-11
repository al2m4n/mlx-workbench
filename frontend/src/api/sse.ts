/**
 * POST-based SSE client.
 *
 * The native EventSource API only supports GET, but we want to send chat
 * payloads in the request body. This helper wraps fetch + ReadableStream and
 * parses the SSE wire format (`event: ...` / `data: ...` blocks separated by
 * blank lines).
 */

import type { ChatMessage, DonePayload } from "./client";

export type SSEEvent = { event: string; data: unknown };

export type SSEHandlers = {
  onEvent: (ev: SSEEvent) => void;
  onError: (message: string) => void;
};

export async function streamSSE(
  url: string,
  body: unknown,
  handlers: SSEHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    handlers.onError(`HTTP ${res.status}: ${text}`);
    return;
  }
  if (!res.body) {
    handlers.onError("No response body");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      // SSE allows CRLF, LF, or CR as line endings (sse-starlette emits CRLF).
      // Normalise to LF so the `\n\n` block separator below always matches.
      buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");

      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        dispatch(block, handlers);
      }
    }
    if (buffer.trim()) dispatch(buffer, handlers);
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      handlers.onError((err as Error).message);
    }
  }
}

function dispatch(block: string, handlers: SSEHandlers): void {
  let event = "message";
  const dataLines: string[] = [];
  for (const raw of block.split("\n")) {
    const line = raw.trimEnd();
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (dataLines.length === 0) return;
  const raw = dataLines.join("\n");
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return;
  }
  handlers.onEvent({ event, data });
}

export type ChatStreamRequest = {
  model_key: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
};

export type ChatStreamHandlers = {
  onToken: (text: string) => void;
  onDone: (payload: DonePayload) => void;
  onError: (message: string) => void;
};

export async function streamChat(
  body: ChatStreamRequest,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await streamSSE(
    "/api/llm/chat",
    { ...body, stream: true },
    {
      onEvent: ({ event, data }) => {
        if (event === "token" && data && typeof data === "object" && "text" in data) {
          handlers.onToken((data as { text: string }).text);
        } else if (event === "done") {
          handlers.onDone(data as DonePayload);
        } else if (
          event === "error" &&
          data &&
          typeof data === "object" &&
          "message" in data
        ) {
          handlers.onError((data as { message: string }).message);
        }
      },
      onError: handlers.onError,
    },
    signal,
  );
}
