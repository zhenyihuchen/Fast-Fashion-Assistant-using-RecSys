const STEP_LABELS = [
  "Parsing query",
  "Retrieving candidates",
  "Loading catalog",
  "Scoring & ranking",
  "Generating explanations",
  "Finalising",
];

interface Props {
  step: number;   // 1-based current step
  total: number;
  message: string;
}

export default function ProgressIndicator({ step, total, message }: Props) {
  const pct = Math.round((step / total) * 100);

  return (
    <div className="w-full max-w-lg">
      {/* Step pills */}
      <div className="flex gap-1 mb-3">
        {STEP_LABELS.map((label, i) => {
          const idx = i + 1;
          const done = idx < step;
          const active = idx === step;
          return (
            <div
              key={label}
              title={label}
              className={[
                "h-1 flex-1 rounded-full transition-all duration-500",
                done   ? "bg-ink-900"  : "",
                active ? "bg-ink-600 animate-pulse" : "",
                !done && !active ? "bg-ink-100" : "",
              ].join(" ")}
            />
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="w-full h-0.5 bg-ink-100 rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-ink-900 transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Status text */}
      <p className="text-xs text-ink-600">{message}</p>
    </div>
  );
}
