import { useCallback, useEffect, useRef, useState } from "react";
import { generateSessionTitle, search } from "./api/client";
import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import type { ChatMsg, Session } from "./types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function uid() {
  return Math.random().toString(36).slice(2);
}

const INTRO_TEXT =
  "Hey, I'm Zara's virtual shopping assistant. I can help you find your next purchase faster than traditional searching. What do you want to buy today, and for what occasion?";
const STORAGE_KEY = "fashion-assistant-sessions";

function makeIntroMsg(): ChatMsg {
  return { id: uid(), role: "assistant", type: "text", content: INTRO_TEXT };
}

function makeSession(label: string): Session {
  return {
    id: uid(),
    label,
    messages: [makeIntroMsg()],
    pendingQuery: null,
    awaitingOccasion: false,
  };
}

function conversationText(messages: ChatMsg[]): string {
  return messages
    .flatMap((msg) => {
      switch (msg.type) {
        case "text":
          return msg.content ? [`${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`] : [];
        case "reco":
          return msg.summary ? [`Assistant: ${msg.summary}`] : [];
        case "occasion_prompt":
          return ["Assistant: Please clarify the occasion you are shopping for."];
        case "no_results":
          return ["Assistant: No matching results were found for this request."];
        default:
          return [];
      }
    })
    .join("\n");
}

function loadStoredState(): { sessions: Session[]; currentId: string } {
  const fallbackSessions = [makeSession("Session 1")];
  const fallback = { sessions: fallbackSessions, currentId: fallbackSessions[0].id };

  if (typeof window === "undefined") {
    return fallback;
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return fallback;
    }
    const parsed = JSON.parse(raw) as {
      sessions?: Session[];
      currentId?: string;
    };
    const sessions = Array.isArray(parsed.sessions) ? parsed.sessions : [];
    if (sessions.length === 0) {
      return fallback;
    }
    const currentId = sessions.some((session) => session.id === parsed.currentId)
      ? (parsed.currentId as string)
      : sessions[0].id;
    return { sessions, currentId };
  } catch {
    return fallback;
  }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [sessions, setSessions] = useState<Session[]>(() => loadStoredState().sessions);
  const [currentId, setCurrentId] = useState<string>(() => loadStoredState().currentId);
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const titleRequestRef = useRef<Record<string, number>>({});
  const current =
    sessions.find((s) => s.id === currentId) ??
    sessions[0] ??
    makeSession("Session 1");

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [current.messages.length]);

  // Persist sessions and selection locally so they survive refreshes.
  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ sessions, currentId })
    );
  }, [sessions, currentId]);

  // ---------------------------------------------------------------------------
  // Session helpers
  // ---------------------------------------------------------------------------

  const updateSession = useCallback(
    (id: string, updater: (s: Session) => Session) => {
      setSessions((prev) => prev.map((s) => (s.id === id ? updater(s) : s)));
    },
    []
  );

  const pushMsg = useCallback(
    (sessionId: string, msg: ChatMsg) => {
      updateSession(sessionId, (s) => ({
        ...s,
        messages: [...s.messages, msg],
      }));
    },
    [updateSession]
  );

  const replaceLastMsg = useCallback(
    (sessionId: string, msg: ChatMsg) => {
      updateSession(sessionId, (s) => ({
        ...s,
        messages: [...s.messages.slice(0, -1), msg],
      }));
    },
    [updateSession]
  );

  const refreshSessionTitle = useCallback(
    async (sessionId: string, messages: ChatMsg[]) => {
      const text = conversationText(messages);
      if (!text.trim()) return;

      const requestId = (titleRequestRef.current[sessionId] ?? 0) + 1;
      titleRequestRef.current[sessionId] = requestId;

      try {
        const title = await generateSessionTitle(text);
        if (titleRequestRef.current[sessionId] !== requestId) return;
        updateSession(sessionId, (s) => ({ ...s, label: title || s.label }));
      } catch {
        // Keep the existing label if summarization fails.
      }
    },
    [updateSession]
  );

  // ---------------------------------------------------------------------------
  // Pipeline
  // ---------------------------------------------------------------------------

  const runSearch = useCallback(
    async (sessionId: string, query: string) => {
      setLoading(true);

      // Show a loading placeholder
      const loadingId = uid();
      const loadingMsg: ChatMsg = {
        id: loadingId,
        role: "assistant",
        type: "loading",
        step: 0,
        total: 6,
        message: "Starting…",
      };
      pushMsg(sessionId, loadingMsg);

      try {
        for await (const event of search(query)) {
          switch (event.type) {
            case "progress":
              replaceLastMsg(sessionId, {
                id: loadingId,
                role: "assistant",
                type: "loading",
                step: event.step,
                total: event.total,
                message: event.message,
              });
              break;

            case "needs_occasion":
              {
                const nextMsg: ChatMsg = {
                  id: uid(),
                  role: "assistant",
                  type: "occasion_prompt",
                  parsed: event.parsed,
                };
                replaceLastMsg(sessionId, nextMsg);
                const existing = sessions.find((s) => s.id === sessionId);
                if (existing) {
                  void refreshSessionTitle(sessionId, [...existing.messages.slice(0, -1), nextMsg]);
                }
              }
              updateSession(sessionId, (s) => ({
                ...s,
                awaitingOccasion: true,
                pendingQuery: query,
              }));
              break;

            case "results": {
              const hasResults = Object.values(event.rows_by_model).some(
                (rows) => rows.length > 0
              );

              if (hasResults) {
                const nextMsg: ChatMsg = {
                  id: uid(),
                  role: "assistant",
                  type: "reco",
                  summary: event.summary,
                  parsed: event.parsed,
                  rowsByModel: event.rows_by_model,
                };
                replaceLastMsg(sessionId, nextMsg);
                const existing = sessions.find((s) => s.id === sessionId);
                if (existing) {
                  void refreshSessionTitle(sessionId, [...existing.messages.slice(0, -1), nextMsg]);
                }
              } else {
                const nextMsg: ChatMsg = {
                  id: uid(),
                  role: "assistant",
                  type: "no_results",
                  parsed: event.parsed,
                };
                replaceLastMsg(sessionId, nextMsg);
                const existing = sessions.find((s) => s.id === sessionId);
                if (existing) {
                  void refreshSessionTitle(sessionId, [...existing.messages.slice(0, -1), nextMsg]);
                }
              }
              // Reset occasion state
              updateSession(sessionId, (s) => ({
                ...s,
                awaitingOccasion: false,
                pendingQuery: null,
              }));
              break;
            }

            case "error":
              {
                const nextMsg: ChatMsg = {
                  id: uid(),
                  role: "assistant",
                  type: "text",
                  content: `Something went wrong: ${event.message}`,
                };
                replaceLastMsg(sessionId, nextMsg);
                const existing = sessions.find((s) => s.id === sessionId);
                if (existing) {
                  void refreshSessionTitle(sessionId, [...existing.messages.slice(0, -1), nextMsg]);
                }
              }
              break;

            case "done":
              break;
          }
        }
      } catch (err) {
        replaceLastMsg(sessionId, {
          id: uid(),
          role: "assistant",
          type: "text",
          content: `Connection error: ${err instanceof Error ? err.message : String(err)}`,
        });
      } finally {
        setLoading(false);
      }
    },
    [pushMsg, refreshSessionTitle, replaceLastMsg, sessions, updateSession]
  );

  // ---------------------------------------------------------------------------
  // Submit handler
  // ---------------------------------------------------------------------------

  const handleSubmit = useCallback(
    (text: string) => {
      if (loading) return;

      // Add user message
      const userMsg: ChatMsg = { id: uid(), role: "user", type: "text", content: text };
      const nextMessages = [...current.messages, userMsg];
      pushMsg(currentId, userMsg);
      void refreshSessionTitle(currentId, nextMessages);

      // Build effective query
      let effectiveQuery = text;
      if (current.awaitingOccasion && current.pendingQuery) {
        effectiveQuery = `${current.pendingQuery}. Occasion: ${text}`;
        // Reset awaiting state immediately (SSE will confirm)
        updateSession(currentId, (s) => ({
          ...s,
          awaitingOccasion: false,
          pendingQuery: null,
        }));
      }

      runSearch(currentId, effectiveQuery);
    },
    [loading, currentId, current, pushMsg, refreshSessionTitle, updateSession, runSearch]
  );

  // ---------------------------------------------------------------------------
  // Session actions
  // ---------------------------------------------------------------------------

  const handleNewSession = () => {
    const label = `Session ${sessions.length + 1}`;
    const s = makeSession(label);
    setSessions((prev) => [...prev, s]);
    setCurrentId(s.id);
  };

  const handleClear = () => {
    updateSession(currentId, (s) => ({
      ...s,
      messages: [makeIntroMsg()],
      pendingQuery: null,
      awaitingOccasion: false,
    }));
  };

  const placeholder = current.awaitingOccasion
    ? "e.g. party night out, date night, work/office, wedding guest…"
    : "Tell me what you want to buy…";

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        sessions={sessions}
        currentId={currentId}
        onSelect={setCurrentId}
        onNew={handleNewSession}
        onClear={handleClear}
      />

      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="shrink-0 flex items-center justify-between px-8 py-4 border-b border-ink-100">
          <div>
            <h2 className="text-sm font-semibold text-ink-900">{current.label}</h2>
            <p className="text-xs text-ink-400 mt-0.5">
              {loading ? "Processing your request…" : "Ready"}
            </p>
          </div>
          {/* Model badge strip */}
          <div className="flex gap-2">
            {["CLIP", "FashionCLIP"].map((m) => (
              <span
                key={m}
                className="px-2.5 py-1 rounded-full border border-ink-200 text-[10px] font-semibold tracking-widest uppercase text-ink-500"
              >
                {m}
              </span>
            ))}
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
          {current.messages.map((msg) => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 px-8 py-4 border-t border-ink-100 bg-white">
          <ChatInput
            onSubmit={handleSubmit}
            disabled={loading}
            placeholder={placeholder}
          />
          <p className="text-center text-[10px] text-ink-300 mt-2 tracking-wide">
            Powered by CLIP · FashionCLIP · Groq
          </p>
        </div>
      </div>
    </div>
  );
}
