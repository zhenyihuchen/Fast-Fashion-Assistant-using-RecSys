import ProgressIndicator from "./ProgressIndicator";
import ConstraintsViewer from "./ConstraintsViewer";
import ModelResults from "./ModelResults";
import type { ChatMsg } from "../types";

interface Props {
  msg: ChatMsg;
}

/* Typing dots shown while backend processes */
function TypingDots() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-ink-400 animate-pulse-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
}

/* Wrapper for assistant bubble */
function AssistantBubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-7 h-7 rounded-full bg-ink-900 flex items-center justify-center shrink-0 mt-0.5 text-white text-[10px] font-bold select-none">
        Z
      </div>
      <div className="flex-1 max-w-3xl space-y-3">{children}</div>
    </div>
  );
}

/* Wrapper for user bubble */
function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end animate-fade-in">
      <div className="max-w-md px-4 py-2.5 bg-ink-900 text-white rounded-2xl rounded-tr-sm text-sm leading-relaxed">
        {content}
      </div>
    </div>
  );
}

export default function ChatMessage({ msg }: Props) {
  if (msg.role === "user") {
    return <UserBubble content={(msg as { content?: string }).content ?? ""} />;
  }

  // Assistant messages
  switch (msg.type) {
    case "text":
      return (
        <AssistantBubble>
          <p className="text-sm text-ink-800 leading-relaxed">{msg.content}</p>
        </AssistantBubble>
      );

    case "loading":
      return (
        <AssistantBubble>
          {msg.step > 0 ? (
            <ProgressIndicator step={msg.step} total={msg.total} message={msg.message} />
          ) : (
            <TypingDots />
          )}
        </AssistantBubble>
      );

    case "occasion_prompt":
      return (
        <AssistantBubble>
          <p className="text-sm text-ink-800 leading-relaxed">
            Thanks! What occasion is this for?
            <span className="block mt-1 text-ink-500 text-xs">
              e.g. party night out, date night, work/office, wedding guest…
            </span>
          </p>
          {msg.parsed && (
            <div className="mt-2">
              <ConstraintsViewer parsed={msg.parsed} />
            </div>
          )}
        </AssistantBubble>
      );

    case "no_results":
      return (
        <AssistantBubble>
          <p className="text-sm text-ink-800 leading-relaxed">
            I couldn't find matches for that request. Want to adjust the style, colour, or budget?
          </p>
          {msg.parsed && (
            <div className="mt-2">
              <ConstraintsViewer parsed={msg.parsed} />
            </div>
          )}
        </AssistantBubble>
      );

    case "reco":
      return (
        <AssistantBubble>
          {/* Summary text */}
          <p className="text-sm text-ink-800 leading-relaxed">{msg.summary}</p>

          {/* Constraints */}
          <ConstraintsViewer parsed={msg.parsed} />

          {/* Results by model (tabs) */}
          <ModelResults rowsByModel={msg.rowsByModel} />
        </AssistantBubble>
      );

    default:
      return null;
  }
}
