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
const DEFAULT_SESSION_LABEL = "New Chat";

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

function isGenericSessionLabel(label: string): boolean {
  return label === DEFAULT_SESSION_LABEL || /^Session \d+$/.test(label.trim());
}

function firstUserMessage(messages: ChatMsg[]): string | null {
  const first = messages.find(
    (msg): msg is Extract<ChatMsg, { type: "text" }> =>
      msg.type === "text" && msg.role === "user" && Boolean(msg.content.trim())
  );
  return first?.content.trim() || null;
}

function loadStoredState(): { sessions: Session[]; currentId: string } {
  const fallbackSessions = [makeSession(DEFAULT_SESSION_LABEL)];
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
    const sessions = Array.isArray(parsed.sessions)
      ? parsed.sessions.map((session) => ({
          ...session,
          label: /^Session \d+$/.test(session.label) ? DEFAULT_SESSION_LABEL : session.label,
        }))
      : [];
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
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const retitledSessionsRef = useRef<Record<string, boolean>>({});
  const current =
    sessions.find((s) => s.id === currentId) ??
    sessions[0] ??
    makeSession(DEFAULT_SESSION_LABEL);

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
    async (sessionId: string, firstMessage: string) => {
      try {
        const title = await generateSessionTitle(firstMessage);
        updateSession(sessionId, (s) => ({ ...s, label: title || s.label }));
      } catch {
        // Keep the default label if title generation fails.
      }
    },
    [updateSession]
  );

  useEffect(() => {
    for (const session of sessions) {
      if (retitledSessionsRef.current[session.id]) continue;
      if (!isGenericSessionLabel(session.label)) continue;
      const firstMessage = firstUserMessage(session.messages);
      if (!firstMessage) continue;
      retitledSessionsRef.current[session.id] = true;
      void refreshSessionTitle(session.id, firstMessage);
    }
  }, [sessions, refreshSessionTitle]);

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
              replaceLastMsg(sessionId, {
                id: uid(),
                role: "assistant",
                type: "occasion_prompt",
                parsed: event.parsed,
              });
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
                replaceLastMsg(sessionId, {
                  id: uid(),
                  role: "assistant",
                  type: "reco",
                  summary: event.summary,
                  parsed: event.parsed,
                  rowsByModel: event.rows_by_model,
                });
              } else {
                replaceLastMsg(sessionId, {
                  id: uid(),
                  role: "assistant",
                  type: "no_results",
                  parsed: event.parsed,
                });
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
              replaceLastMsg(sessionId, {
                id: uid(),
                role: "assistant",
                type: "text",
                content: `Something went wrong: ${event.message}`,
              });
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
    [pushMsg, replaceLastMsg, updateSession]
  );

  // ---------------------------------------------------------------------------
  // Submit handler
  // ---------------------------------------------------------------------------

  const handleSubmit = useCallback(
    (text: string) => {
      if (loading) return;

      const hasUserMessage = current.messages.some(
        (msg) => msg.role === "user" && msg.type === "text" && Boolean(msg.content.trim())
      );

      // Add user message
      pushMsg(currentId, { id: uid(), role: "user", type: "text", content: text });
      if (!hasUserMessage) {
        void refreshSessionTitle(currentId, text);
      }

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
    const label = DEFAULT_SESSION_LABEL;
    const s = makeSession(label);
    setSessions((prev) => [...prev, s]);
    setCurrentId(s.id);
  };

  const handleDeleteSession = useCallback(
    (sessionId: string) => {
      setSessions((prev) => {
        const remaining = prev.filter((session) => session.id !== sessionId);
        if (remaining.length === 0) {
          const next = makeSession(DEFAULT_SESSION_LABEL);
          setCurrentId(next.id);
          return [next];
        }
        if (sessionId === currentId) {
          const currentIndex = prev.findIndex((session) => session.id === sessionId);
          const nextActive = remaining[Math.max(0, Math.min(currentIndex, remaining.length - 1))];
          setCurrentId(nextActive.id);
        }
        return remaining;
      });
      delete retitledSessionsRef.current[sessionId];
    },
    [currentId]
  );

  const handleClear = () => {
    updateSession(currentId, (s) => ({
      ...s,
      label: DEFAULT_SESSION_LABEL,
      messages: [makeIntroMsg()],
      pendingQuery: null,
      awaitingOccasion: false,
    }));
    delete retitledSessionsRef.current[currentId];
  };

  const placeholder = current.awaitingOccasion
    ? "e.g. party night out, date night, work/office, wedding guest…"
    : "Tell me what you want to buy…";

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Mobile overlay backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-20 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar — always visible on md+, slide-in overlay on mobile */}
      <div className={[
        "fixed inset-y-0 left-0 z-30 md:static md:z-auto md:flex md:shrink-0 transition-transform duration-200",
        sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
      ].join(" ")}>
        <Sidebar
          sessions={sessions}
          currentId={currentId}
          onSelect={(id) => { setCurrentId(id); setSidebarOpen(false); }}
          onDelete={handleDeleteSession}
          onNew={handleNewSession}
          onClear={handleClear}
          onClose={() => setSidebarOpen(false)}
        />
      </div>

      {/* Main chat area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="shrink-0 flex items-center justify-between px-4 md:px-8 py-4 border-b border-ink-100">
          <div className="flex items-center gap-3">
            {/* Hamburger — mobile only */}
            <button
              className="md:hidden p-1 rounded text-ink-600 hover:text-ink-900"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open menu"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor">
                <rect y="3" width="20" height="2" rx="1"/>
                <rect y="9" width="20" height="2" rx="1"/>
                <rect y="15" width="20" height="2" rx="1"/>
              </svg>
            </button>
            <div>
              <h2 className="text-sm font-semibold text-ink-900">{current.label}</h2>
              <p className="text-xs text-ink-400 mt-0.5">
                {loading ? "Processing your request…" : "Ready"}
              </p>
            </div>
          </div>
          {/* Model badge strip */}
          <div className="flex gap-2">
            {["CLIP", "FashionCLIP"].map((m) => (
              <span
                key={m}
                className="hidden sm:inline px-2.5 py-1 rounded-full border border-ink-200 text-[10px] font-semibold tracking-widest uppercase text-ink-500"
              >
                {m}
              </span>
            ))}
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
          {current.messages.map((msg) => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="shrink-0 px-4 md:px-8 py-4 border-t border-ink-100 bg-white">
          <ChatInput
            onSubmit={handleSubmit}
            disabled={loading}
            placeholder={placeholder}
          />
          <p className="text-center text-[10px] text-ink-300 mt-2 tracking-wide">
            Powered by CLIP · FashionCLIP
          </p>
        </div>
      </div>
    </div>
  );
}
