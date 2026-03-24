// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export interface ParsedConstraints {
  normalized?: string;
  constraints?: {
    categories?: string[];
    colors?: string[];
    price_min?: number | null;
    price_max?: number | null;
    fit?: string[];
  };
  occasion?: {
    mode?: string;
    target?: string | null;
    score?: number | null;
  };
  confidence?: {
    overall?: number;
    occasion?: number;
  };
}

export interface ProductRow {
  model_name: string;
  row_id: string;
  relevance_score: number;
  occasion_score: number;
  final_score: number;
  relevance_display: number;
  occasion_display: number;
  final_display: number;
  product_name: string;
  product_description: string;
  product_url: string;
  price: string;
  color: string;
  product_category: string;
  image_url: string;
  explanation: string;
}

// ---------------------------------------------------------------------------
// Chat message types
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "assistant";

interface BaseMsg {
  id: string;
  role: MessageRole;
}

export interface TextMsg extends BaseMsg {
  type: "text";
  content: string;
}

export interface LoadingMsg extends BaseMsg {
  type: "loading";
  step: number;
  total: number;
  message: string;
}

export interface RecoMsg extends BaseMsg {
  type: "reco";
  summary: string;
  parsed: ParsedConstraints;
  rowsByModel: Record<string, ProductRow[]>;
}

export interface OccasionPromptMsg extends BaseMsg {
  type: "occasion_prompt";
  parsed: ParsedConstraints;
}

export interface NoResultsMsg extends BaseMsg {
  type: "no_results";
  parsed: ParsedConstraints;
}

export type ChatMsg =
  | TextMsg
  | LoadingMsg
  | RecoMsg
  | OccasionPromptMsg
  | NoResultsMsg;

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export interface Session {
  id: string;
  label: string;
  messages: ChatMsg[];
  pendingQuery: string | null;
  awaitingOccasion: boolean;
}
