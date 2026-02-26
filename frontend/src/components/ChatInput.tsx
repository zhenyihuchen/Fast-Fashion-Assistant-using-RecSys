import { useRef, useState } from "react";
import { ArrowUp } from "lucide-react";

interface Props {
  onSubmit: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput({ onSubmit, disabled, placeholder }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    // Auto-grow
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  return (
    <div className="flex items-end gap-3 bg-white border border-ink-200 rounded-2xl px-4 py-3 shadow-sm focus-within:border-ink-400 transition-colors">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        placeholder={placeholder ?? "Tell me what you want to buy…"}
        className="flex-1 resize-none bg-transparent text-sm text-ink-900 placeholder:text-ink-400 outline-none leading-relaxed max-h-40 disabled:opacity-50"
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-xl bg-ink-900 text-white hover:bg-ink-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        aria-label="Send"
      >
        <ArrowUp size={14} strokeWidth={2.5} />
      </button>
    </div>
  );
}
