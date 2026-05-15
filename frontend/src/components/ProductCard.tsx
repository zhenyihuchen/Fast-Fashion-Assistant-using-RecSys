import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import type { ProductRow } from "../types";

interface Props {
  product: ProductRow;
  rank: number;
}

export default function ProductCard({ product, rank }: Props) {
  const [imgError, setImgError] = useState(false);
  const [explanationOpen, setExplanationOpen] = useState(false);

  // const scorePercent = ((product.final_display / 10) * 100).toFixed(0);
  const hasLongExplanation = Boolean(product.explanation && product.explanation.length > 140);

  return (
    <article className="group flex flex-col bg-white border border-ink-100 rounded-xl overflow-hidden hover:shadow-md hover:border-ink-300 transition-all duration-200 animate-fade-in">
      {/* Image */}
      <div className="relative bg-ink-50 aspect-[3/4] overflow-hidden">
        {/* Rank badge */}
        <span className="absolute top-3 left-3 z-10 w-6 h-6 flex items-center justify-center bg-ink-900 text-white text-[11px] font-semibold rounded-full">
          {rank}
        </span>

        {product.image_url && !imgError ? (
          <img
            src={product.image_url}
            alt={product.product_name}
            onError={() => setImgError(true)}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-ink-200">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
              <path d="M20.38 3.46 16 2a4 4 0 0 1-8 0L3.62 3.46a2 2 0 0 0-1.34 2.23l.58 3.57a1 1 0 0 0 .99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 0 0 2-2V10h2.15a1 1 0 0 0 .99-.84l.58-3.57a2 2 0 0 0-1.34-2.23z" />
            </svg>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="flex flex-col flex-1 p-4 gap-3">
        {/* Category + meta */}
        <div>
          <p className="text-[10px] tracking-widest uppercase text-ink-400 mb-1">
            {[product.product_category, product.color].filter(Boolean).join(" · ")}
          </p>
          <h3 className="font-semibold text-sm leading-snug line-clamp-2">
            {product.product_name || "—"}
          </h3>
          {product.price && (
            <p className="text-sm text-ink-600 mt-1">{product.price}</p>
          )}
        </div>

        {/* Score bar */}
        {/* <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-widest text-ink-400">Match score</span>
            <span className="text-[11px] font-semibold text-ink-700">
              {product.final_display.toFixed(1)}/10
            </span>
          </div>
          <div className="w-full h-0.5 bg-ink-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-ink-900 rounded-full transition-all duration-700"
              style={{ width: `${scorePercent}%` }}
            />
          </div>
          <div className="flex gap-3 mt-1.5">
            <span className="text-[10px] text-ink-400">
              Relevance&nbsp;<span className="text-ink-600">{product.relevance_display.toFixed(1)}/10</span>
            </span>
            <span className="text-[10px] text-ink-400">
              Occasion&nbsp;<span className="text-ink-600">{product.occasion_display.toFixed(1)}/10</span>
            </span>
          </div>
        </div> */}

        {/* Explanation */}
        {product.explanation && (
          <div className="space-y-2">
            <p
              className={[
                "text-xs text-ink-600 leading-relaxed",
                explanationOpen ? "" : "line-clamp-3",
              ].join(" ")}
            >
              {product.explanation}
            </p>
            {hasLongExplanation && (
              <button
                type="button"
                onClick={() => setExplanationOpen((open) => !open)}
                aria-expanded={explanationOpen}
                className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wide text-ink-700 hover:text-ink-950 transition-colors"
              >
                {explanationOpen ? (
                  <>
                    Show less
                    <ChevronUp size={12} />
                  </>
                ) : (
                  <>
                    Show full explanation
                    <ChevronDown size={12} />
                  </>
                )}
              </button>
            )}
          </div>
        )}

        {/* CTA */}
        {product.product_url && (
          <a
            href={product.product_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-auto flex items-center justify-center gap-1.5 w-full py-2 rounded-lg border border-ink-900 text-[11px] font-semibold tracking-wide uppercase text-ink-900 hover:bg-ink-900 hover:text-white transition-colors duration-150"
          >
            View on Zara
            <ExternalLink size={11} />
          </a>
        )}
      </div>
    </article>
  );
}
