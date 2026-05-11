/**
 * Streaming parser for OpenAI's harmony chat format (used by gpt-oss models).
 *
 * The model emits control tokens that delimit reasoning vs. user-facing text:
 *
 *   <|channel|>analysis<|message|>...thinking...<|end|>
 *   <|start|>assistant<|channel|>final<|message|>...reply...<|return|>
 *
 * Other reasoning models (DeepSeek-R1 with <think>...</think>) use different
 * tokens — extend `MARKERS` to support them.
 *
 * For non-harmony models the stream contains no markers and every chunk is
 * emitted as `final` unchanged.
 */

export type Channel = "final" | "analysis" | "other";

export type Segment = { channel: Channel; text: string };

const MARKERS: { match: string; next: Channel }[] = [
  { match: "<|start|>assistant<|channel|>analysis<|message|>", next: "analysis" },
  { match: "<|start|>assistant<|channel|>final<|message|>", next: "final" },
  { match: "<|start|>assistant<|channel|>commentary<|message|>", next: "other" },
  { match: "<|channel|>analysis<|message|>", next: "analysis" },
  { match: "<|channel|>final<|message|>", next: "final" },
  { match: "<|channel|>commentary<|message|>", next: "other" },
];

// Markers we silently consume without changing channel.
const PASSTHRU = ["<|end|>", "<|return|>", "<|start|>assistant", "<|start|>"];

const ALL_MARKERS = [...MARKERS.map((m) => m.match), ...PASSTHRU];

export class HarmonyParser {
  private state: Channel = "final";
  private pending = "";

  feed(text: string): Segment[] {
    const buf = this.pending + text;
    this.pending = "";

    const segments: Segment[] = [];
    let i = 0;

    while (i < buf.length) {
      const idx = buf.indexOf("<|", i);
      if (idx === -1) {
        if (i < buf.length) {
          segments.push({ channel: this.state, text: buf.slice(i) });
        }
        return segments;
      }

      if (idx > i) {
        segments.push({ channel: this.state, text: buf.slice(i, idx) });
      }

      const tail = buf.slice(idx);

      // State-changing marker?
      const m = MARKERS.find((mk) => tail.startsWith(mk.match));
      if (m) {
        this.state = m.next;
        i = idx + m.match.length;
        continue;
      }

      // Pass-through marker?
      const p = PASSTHRU.find((pk) => tail.startsWith(pk));
      if (p) {
        i = idx + p.length;
        continue;
      }

      // Possible partial marker straddling chunk boundary — hold for next feed.
      if (ALL_MARKERS.some((mk) => mk.startsWith(tail))) {
        this.pending = tail;
        return segments;
      }

      // Just a literal "<|" in content; emit and move on.
      segments.push({ channel: this.state, text: "<|" });
      i = idx + 2;
    }

    return segments;
  }

  /** Emit any leftover buffered text — call at end of stream. */
  flush(): Segment[] {
    if (!this.pending) return [];
    const out: Segment = { channel: this.state, text: this.pending };
    this.pending = "";
    return [out];
  }
}
