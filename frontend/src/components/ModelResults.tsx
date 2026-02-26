import { useState } from "react";
import ProductCard from "./ProductCard";
import type { ProductRow } from "../types";

const MODEL_LABELS: Record<string, string> = {
  clip: "CLIP",
  fashion_clip: "FashionCLIP",
};

interface Props {
  rowsByModel: Record<string, ProductRow[]>;
}

export default function ModelResults({ rowsByModel }: Props) {
  const modelNames = Object.keys(rowsByModel).filter((m) => rowsByModel[m].length > 0);
  const [activeTab, setActiveTab] = useState<string>(modelNames[0] ?? "");

  if (modelNames.length === 0) return null;

  // "compare" tab is available when both models have results
  const showCompare = modelNames.length > 1;
  const tabs = [...modelNames, ...(showCompare ? ["compare"] : [])];

  return (
    <div className="w-full animate-fade-in">
      {/* Tab bar */}
      <div className="flex gap-0 border-b border-ink-100 mb-6">
        {tabs.map((tab) => {
          const label =
            tab === "compare"
              ? "Side-by-side"
              : (MODEL_LABELS[tab] ?? tab.toUpperCase());
          const active = activeTab === tab;
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={[
                "px-4 py-2.5 text-xs font-semibold tracking-wider uppercase transition-colors border-b-2 -mb-px",
                active
                  ? "border-ink-900 text-ink-900"
                  : "border-transparent text-ink-400 hover:text-ink-700",
              ].join(" ")}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      {activeTab === "compare" && showCompare ? (
        /* Side-by-side: two columns */
        <div className="grid grid-cols-2 gap-6">
          {modelNames.map((model) => (
            <div key={model}>
              <h3 className="text-[10px] tracking-widest uppercase font-semibold text-ink-400 mb-4">
                {MODEL_LABELS[model] ?? model.toUpperCase()} results
              </h3>
              <div className="flex flex-col gap-4">
                {rowsByModel[model].map((p, i) => (
                  <ProductCard key={p.row_id} product={p} rank={i + 1} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Single model: responsive grid */
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {(rowsByModel[activeTab] ?? []).map((p, i) => (
            <ProductCard key={p.row_id} product={p} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
