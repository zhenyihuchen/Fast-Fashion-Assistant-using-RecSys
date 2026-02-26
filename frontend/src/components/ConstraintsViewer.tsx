import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { ParsedConstraints } from "../types";

interface Props {
  parsed: ParsedConstraints;
}

export default function ConstraintsViewer({ parsed }: Props) {
  const [open, setOpen] = useState(false);

  const c = parsed.constraints ?? {};
  const o = parsed.occasion ?? {};

  const chips: string[] = [
    ...(c.categories ?? []),
    ...(c.colors ?? []),
    ...(c.fit ?? []),
    ...(c.price_min != null ? [`≥ €${c.price_min}`] : []),
    ...(c.price_max != null ? [`≤ €${c.price_max}`] : []),
    ...(o.target ? [o.target] : []),
  ];

  return (
    <div className="border border-ink-100 rounded-lg overflow-hidden text-sm">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center justify-between w-full px-4 py-3 bg-ink-50 hover:bg-ink-100 transition-colors"
      >
        <span className="font-medium text-ink-700">Parsed constraints</span>
        {open ? (
          <ChevronDown size={14} className="text-ink-400" />
        ) : (
          <ChevronRight size={14} className="text-ink-400" />
        )}
      </button>

      {open && (
        <div className="px-4 py-3 bg-white space-y-3">
          {/* Quick chips */}
          {chips.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {chips.map((chip) => (
                <span
                  key={chip}
                  className="px-2 py-0.5 bg-ink-900 text-white text-[11px] rounded-full tracking-wide"
                >
                  {chip}
                </span>
              ))}
            </div>
          )}

          {/* Raw JSON */}
          <pre className="text-[11px] text-ink-600 bg-ink-50 rounded p-3 overflow-auto max-h-52 leading-relaxed">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
