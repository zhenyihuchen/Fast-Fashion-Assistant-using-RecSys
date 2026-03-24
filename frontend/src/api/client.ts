import type { ParsedConstraints, ProductRow } from "../types";

// When running via Vite proxy the base is empty; override for standalone builds.
const BASE = import.meta.env.VITE_API_BASE ?? "";

// ---------------------------------------------------------------------------
// SSE event shapes
// ---------------------------------------------------------------------------

export type ProgressEvent = {
  type: "progress";
  step: number;
  total: number;
  message: string;
};

export type NeedsOccasionEvent = {
  type: "needs_occasion";
  parsed: ParsedConstraints;
};

export type ResultsEvent = {
  type: "results";
  parsed: ParsedConstraints;
  rows_by_model: Record<string, ProductRow[]>;
  summary: string;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type DoneEvent = { type: "done" };

export type SearchEvent =
  | ProgressEvent
  | NeedsOccasionEvent
  | ResultsEvent
  | ErrorEvent
  | DoneEvent;

// ---------------------------------------------------------------------------
// Streaming search — yields typed events one by one
// ---------------------------------------------------------------------------

export async function* search(query: string): AsyncGenerator<SearchEvent> {
  const response = await fetch(`${BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by blank lines
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      let eventType = "";
      let dataLine = "";

      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) eventType = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLine = line.slice(6).trim();
      }

      if (!eventType || !dataLine) continue;

      try {
        const data = JSON.parse(dataLine);
        yield { type: eventType, ...data } as SearchEvent;
      } catch {
        console.error("Failed to parse SSE frame:", frame);
      }
    }
  }
}


export async function generateSessionTitle(conversation: string): Promise<string> {
  const response = await fetch(`${BASE}/api/session-title`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ conversation }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const data = await response.json();
  return typeof data.title === "string" && data.title.trim() ? data.title.trim() : "New chat";
}
