import { Plus, Trash2 } from "lucide-react";
import type { Session } from "../types";

interface Props {
  sessions: Session[];
  currentId: string;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  onClear: () => void;
}

export default function Sidebar({ sessions, currentId, onSelect, onDelete, onNew, onClear }: Props) {
  return (
    <aside className="flex flex-col h-full w-60 bg-ink-900 text-white shrink-0">
      {/* Brand */}
      <div className="px-6 pt-8 pb-6 border-b border-ink-800">
        <p className="text-[10px] tracking-[0.25em] uppercase text-ink-400 mb-1">Zara</p>
        <h1 className="text-lg font-semibold leading-tight">Fashion Assistant</h1>
      </div>

      {/* Sessions list */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-0.5">
        <p className="px-3 mb-2 text-[10px] tracking-widest uppercase text-ink-400">Sessions</p>
        {sessions.map((s) => (
          <div
            key={s.id}
            className={[
              "group flex items-center gap-2 rounded transition-colors",
              s.id === currentId ? "bg-white/10" : "hover:bg-white/5",
            ].join(" ")}
          >
            <button
              onClick={() => onSelect(s.id)}
              className={[
                "flex-1 min-w-0 text-left px-3 py-2 rounded text-sm transition-colors truncate",
                s.id === currentId
                  ? "text-white font-medium"
                  : "text-ink-400 group-hover:text-white",
              ].join(" ")}
              title={s.label}
            >
              {s.label}
            </button>
            <button
              onClick={() => onDelete(s.id)}
              className="shrink-0 mr-2 text-ink-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
              aria-label={`Delete ${s.label}`}
              title="Delete session"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="px-3 py-4 border-t border-ink-800 space-y-1">
        <button
          onClick={onNew}
          className="flex items-center gap-2 w-full px-3 py-2 rounded text-sm text-ink-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <Plus size={14} />
          New session
        </button>
        <button
          onClick={onClear}
          className="flex items-center gap-2 w-full px-3 py-2 rounded text-sm text-ink-400 hover:text-white hover:bg-white/5 transition-colors"
        >
          <Trash2 size={14} />
          Clear conversation
        </button>
      </div>
    </aside>
  );
}
