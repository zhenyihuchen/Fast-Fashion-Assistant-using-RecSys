import { useRef, useState } from "react";
import { ArrowUp, Mic, Square } from "lucide-react";

interface Props {
  onSubmit: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

type RecordState = "idle" | "recording" | "transcribing";

export default function ChatInput({ onSubmit, disabled, placeholder }: Props) {
  const [value, setValue] = useState("");
  const [recordState, setRecordState] = useState<RecordState>("idle");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

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
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      alert("Your browser does not support microphone access.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecordState("transcribing");

        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "recording.webm");

        try {
          const res = await fetch("/api/transcribe", {
            method: "POST",
            body: formData,
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          setValue(data.text ?? "");
          requestAnimationFrame(() => {
            if (textareaRef.current) {
              textareaRef.current.style.height = "auto";
              textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
              textareaRef.current.focus();
            }
          });
        } catch {
          alert("Transcription failed. Please try again.");
        } finally {
          setRecordState("idle");
        }
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setRecordState("recording");
    } catch {
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    mediaRecorderRef.current = null;
  };

  const handleMicClick = () => {
    if (recordState === "recording") {
      stopRecording();
    } else if (recordState === "idle") {
      startRecording();
    }
  };

  const micDisabled = disabled || recordState === "transcribing";

  return (
    <div className="flex items-end gap-3 bg-white border border-ink-200 rounded-2xl px-4 py-3 shadow-sm focus-within:border-ink-400 transition-colors">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        disabled={disabled || recordState !== "idle"}
        placeholder={
          recordState === "recording"
            ? "Listening…"
            : recordState === "transcribing"
            ? "Transcribing…"
            : (placeholder ?? "Tell me what you want to buy…")
        }
        className="flex-1 resize-none bg-transparent text-sm text-ink-900 placeholder:text-ink-400 outline-none leading-relaxed max-h-40 disabled:opacity-50"
      />

      {/* Mic button */}
      <button
        onClick={handleMicClick}
        disabled={micDisabled}
        title={recordState === "recording" ? "Stop recording" : "Voice input"}
        className={`shrink-0 w-8 h-8 flex items-center justify-center rounded-xl transition-colors disabled:opacity-30 disabled:cursor-not-allowed
          ${recordState === "recording"
            ? "bg-red-500 text-white animate-pulse hover:bg-red-600"
            : "border border-ink-200 text-ink-500 hover:bg-ink-50"
          }`}
        aria-label={recordState === "recording" ? "Stop recording" : "Voice input"}
      >
        {recordState === "recording" ? (
          <Square size={12} strokeWidth={2.5} fill="currentColor" />
        ) : (
          <Mic size={14} strokeWidth={2} />
        )}
      </button>

      {/* Send button */}
      <button
        onClick={submit}
        disabled={disabled || !value.trim() || recordState !== "idle"}
        className="shrink-0 w-8 h-8 flex items-center justify-center rounded-xl bg-ink-900 text-white hover:bg-ink-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        aria-label="Send"
      >
        <ArrowUp size={14} strokeWidth={2.5} />
      </button>
    </div>
  );
}
