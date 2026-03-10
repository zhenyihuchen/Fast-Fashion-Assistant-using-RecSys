"""Streamlit evaluation dashboard.

Run with (from project root, myenv activated):
    streamlit run evaluation/dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fashion RecSys — Evaluation",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────

ITEM_RUBRICS = [
    "item_relevance",
    "occasion_appropriateness",
    "explanation_quality",
]
ITEM_LABELS = {
    "item_relevance": "Item Relevance",
    "occasion_appropriateness": "Occasion Appropriateness",
    "explanation_quality": "Explanation Quality",
    "set_answer_relevance": "Set Answer Relevance",
}
PARSER_RUBRICS = [
    "parser_completeness",
    "parser_no_hallucination",
    "parser_occasion_detection",
]
PARSER_LABELS = {
    "parser_completeness": "Completeness",
    "parser_no_hallucination": "No Hallucination",
    "parser_occasion_detection": "Occasion Detection",
}
MODEL_COLORS = {"CLIP": "#3B82F6", "FashionCLIP": "#F59E0B", "Tie": "#9CA3AF"}

# ── Load results ──────────────────────────────────────────────────────────────

result_files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True) if RESULTS_DIR.exists() else []

with st.sidebar:
    st.title("RecSys Evaluation")
    st.markdown("---")

    if not result_files:
        st.warning("No results found.\n\nRun the evaluator:\n```\npython -m evaluation.run_eval\n```")
        st.stop()

    selected_file = st.selectbox(
        "Results run",
        result_files,
        format_func=lambda p: p.stem.replace("eval_", ""),
    )
    data = json.loads(selected_file.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    per_query = data.get("per_query", [])

    st.caption(f"**Timestamp:** {data.get('run_timestamp', '—')}")
    st.caption(f"**Queries:** {data.get('total_queries', 0)}")
    st.markdown("---")
    st.caption("CLIP — blue  ·  FashionCLIP — orange")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _agg_mean(model_key: str, rubric: str) -> float | None:
    return (agg.get(model_key) or {}).get(rubric, {}).get("mean")

def _agg_std(model_key: str, rubric: str) -> float | None:
    return (agg.get(model_key) or {}).get(rubric, {}).get("std")

def _model_overall_avg(model_key: str) -> float | None:
    scores = [_agg_mean(model_key, r) for r in ITEM_RUBRICS + ["set_answer_relevance"]]
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 2) if valid else None

def _score_badge(score: int | None) -> str:
    if score is None or score < 0:
        return "—"
    return f"{score}/5"


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_comparison, tab_parser, tab_explorer = st.tabs([
    "Overview",
    "Model Comparison",
    "Parser Analysis",
    "Query Explorer",
])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────

with tab_overview:
    st.header("Evaluation Overview")

    clip_avg = _model_overall_avg("clip")
    fc_avg = _model_overall_avg("fashion_clip")
    cm = agg.get("cross_model", {})
    total = cm.get("total", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("CLIP overall avg", f"{clip_avg:.2f} / 5" if clip_avg else "—")
    with c2:
        st.metric("FashionCLIP overall avg", f"{fc_avg:.2f} / 5" if fc_avg else "—")
    with c3:
        delta = round(fc_avg - clip_avg, 2) if clip_avg and fc_avg else None
        st.metric("FashionCLIP vs CLIP", f"{delta:+.2f}" if delta is not None else "—")
    with c4:
        st.metric("FashionCLIP wins", f"{cm.get('fashionclip_wins', 0)} / {total}")
    with c5:
        st.metric("CLIP wins", f"{cm.get('clip_wins', 0)} / {total}")

    st.divider()
    col_radar, col_pie = st.columns([3, 2])

    with col_radar:
        st.subheader("Rubric Scores — Radar Chart")
        categories = [ITEM_LABELS.get(r, r) for r in ITEM_RUBRICS]
        clip_vals = [_agg_mean("clip", r) or 0 for r in ITEM_RUBRICS]
        fc_vals = [_agg_mean("fashion_clip", r) or 0 for r in ITEM_RUBRICS]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=clip_vals, theta=categories, fill="toself",
            name="CLIP", line_color=MODEL_COLORS["CLIP"],
        ))
        fig.add_trace(go.Scatterpolar(
            r=fc_vals, theta=categories, fill="toself",
            name="FashionCLIP", line_color=MODEL_COLORS["FashionCLIP"],
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    with col_pie:
        st.subheader("Cross-Model Preference")
        labels = ["FashionCLIP", "CLIP", "Tie"]
        values = [cm.get("fashionclip_wins", 0), cm.get("clip_wins", 0), cm.get("ties", 0)]
        fig_pie = px.pie(
            names=labels, values=values,
            color=labels,
            color_discrete_map={"CLIP": MODEL_COLORS["CLIP"], "FashionCLIP": MODEL_COLORS["FashionCLIP"], "Tie": MODEL_COLORS["Tie"]},
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, width="stretch")

    # Summary table
    st.subheader("Rubric Summary Table")
    rows = []
    for r in ITEM_RUBRICS + ["set_answer_relevance"]:
        label = ITEM_LABELS.get(r, r)
        cm_val = _agg_mean("clip", r)
        fc_val = _agg_mean("fashion_clip", r)
        delta = round(fc_val - cm_val, 2) if cm_val is not None and fc_val is not None else None
        rows.append({
            "Rubric": label,
            "CLIP mean": f"{cm_val:.2f}" if cm_val is not None else "—",
            "CLIP std": f"±{_agg_std('clip', r):.2f}" if _agg_std("clip", r) is not None else "—",
            "FashionCLIP mean": f"{fc_val:.2f}" if fc_val is not None else "—",
            "FashionCLIP std": f"±{_agg_std('fashion_clip', r):.2f}" if _agg_std("fashion_clip", r) is not None else "—",
            "Δ (FC − CLIP)": f"{delta:+.2f}" if delta is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

# ── TAB 2: Model Comparison ───────────────────────────────────────────────────

with tab_comparison:
    st.header("CLIP vs FashionCLIP — Detailed Comparison")

    # Grouped bar chart for all rubrics
    rubric_df_rows = []
    for r in ITEM_RUBRICS + ["set_answer_relevance"]:
        label = ITEM_LABELS.get(r, r)
        for model_key, model_label in [("clip", "CLIP"), ("fashion_clip", "FashionCLIP")]:
            mean = _agg_mean(model_key, r)
            std = _agg_std(model_key, r)
            if mean is not None:
                rubric_df_rows.append({
                    "Rubric": label,
                    "Model": model_label,
                    "Mean Score": mean,
                    "Std": std or 0,
                })

    if rubric_df_rows:
        df_rubrics = pd.DataFrame(rubric_df_rows)
        fig_bar = px.bar(
            df_rubrics, x="Rubric", y="Mean Score", color="Model", barmode="group",
            error_y="Std",
            color_discrete_map=MODEL_COLORS,
            range_y=[0, 5],
            height=420,
        )
        fig_bar.update_layout(xaxis_tickangle=-20, margin=dict(t=20, b=80))
        st.plotly_chart(fig_bar, width="stretch")
    else:
        st.info("No rubric data available.")

    st.divider()

    # Per-query scores: one rubric selector
    st.subheader("Per-Query Score Distribution")
    selected_rubric = st.selectbox(
        "Rubric",
        ITEM_RUBRICS + ["set_answer_relevance"],
        format_func=lambda r: ITEM_LABELS.get(r, r),
        key="comparison_rubric",
    )

    scatter_rows = []
    for q in per_query:
        ev = q.get("eval", {})
        qid = q.get("id", "")
        query_short = q.get("query", "")[:50]

        if selected_rubric == "set_answer_relevance":
            clip_s = (ev.get("clip", {}).get("set") or {}).get(selected_rubric, {}).get("score")
            fc_s = (ev.get("fashion_clip", {}).get("set") or {}).get(selected_rubric, {}).get("score")
        else:
            clip_items = (ev.get("clip", {}).get("items") or [])
            fc_items = (ev.get("fashion_clip", {}).get("items") or [])
            clip_vals = [i.get(selected_rubric, {}).get("score") for i in clip_items if i.get(selected_rubric, {}).get("score") is not None]
            fc_vals = [i.get(selected_rubric, {}).get("score") for i in fc_items if i.get(selected_rubric, {}).get("score") is not None]
            clip_s = round(sum(clip_vals) / len(clip_vals), 2) if clip_vals else None
            fc_s = round(sum(fc_vals) / len(fc_vals), 2) if fc_vals else None

        if clip_s is not None or fc_s is not None:
            scatter_rows.append({"Query": f"{qid}: {query_short}", "CLIP": clip_s, "FashionCLIP": fc_s})

    if scatter_rows:
        df_scatter = pd.DataFrame(scatter_rows)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Bar(
            x=df_scatter["Query"], y=df_scatter["CLIP"],
            name="CLIP", marker_color=MODEL_COLORS["CLIP"],
        ))
        fig_scatter.add_trace(go.Bar(
            x=df_scatter["Query"], y=df_scatter["FashionCLIP"],
            name="FashionCLIP", marker_color=MODEL_COLORS["FashionCLIP"],
        ))
        fig_scatter.update_layout(
            barmode="group", xaxis_tickangle=-40,
            yaxis=dict(range=[0, 5], title="Score"),
            height=380, margin=dict(t=10, b=120),
        )
        st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("No per-query data for this rubric.")

# ── TAB 3: Parser Analysis ────────────────────────────────────────────────────

with tab_parser:
    st.header("Query Parser Evaluation")

    # Aggregate bar chart
    parser_agg = agg.get("parser", {})
    p_rows = [
        {
            "Rubric": PARSER_LABELS.get(r, r),
            "Mean": parser_agg.get(r, {}).get("mean"),
            "Std": parser_agg.get(r, {}).get("std") or 0,
        }
        for r in PARSER_RUBRICS
        if parser_agg.get(r, {}).get("mean") is not None
    ]

    if p_rows:
        df_parser = pd.DataFrame(p_rows)
        fig_p = px.bar(
            df_parser, x="Rubric", y="Mean", error_y="Std",
            color="Rubric",
            range_y=[0, 5],
            color_discrete_sequence=["#6366F1", "#EC4899", "#14B8A6"],
            height=350,
        )
        fig_p.update_layout(showlegend=False, margin=dict(t=10, b=40))
        st.plotly_chart(fig_p, width="stretch")

        col1, col2, col3 = st.columns(3)
        for col, r in zip([col1, col2, col3], PARSER_RUBRICS):
            mean = parser_agg.get(r, {}).get("mean")
            col.metric(PARSER_LABELS[r], f"{mean:.2f} / 5" if mean else "—")
    else:
        st.info("No parser evaluation data available.")

    st.divider()

    # Per-query parser scores table
    st.subheader("Per-Query Parser Scores")
    pq_rows = []
    for q in per_query:
        parser_ev = (q.get("eval") or {}).get("parser", {})
        pq_rows.append({
            "ID": q.get("id", ""),
            "Query": q.get("query", "")[:60],
            "Type": ", ".join(q.get("type", [])),
            "Completeness": parser_ev.get("parser_completeness", {}).get("score"),
            "No Hallucination": parser_ev.get("parser_no_hallucination", {}).get("score"),
            "Occasion Detection": parser_ev.get("parser_occasion_detection", {}).get("score"),
        })

    if pq_rows:
        df_pq = pd.DataFrame(pq_rows)
        st.dataframe(df_pq, hide_index=True)

        # Highlight which queries failed occasion detection
        failed = df_pq[df_pq["Occasion Detection"].notna() & (df_pq["Occasion Detection"] < 3)]
        if not failed.empty:
            st.warning(f"**{len(failed)} queries** scored < 3 on occasion detection:")
            for _, row in failed.iterrows():
                st.caption(f"  {row['ID']}: {row['Query']}")

# ── TAB 4: Query Explorer ─────────────────────────────────────────────────────

with tab_explorer:
    st.header("Query Explorer")

    if not per_query:
        st.info("No per-query data available.")
    else:
        query_options = {
            q["id"]: f"{q['id']} — {q['query'][:70]}"
            for q in per_query
        }
        selected_id = st.selectbox(
            "Select a query",
            list(query_options.keys()),
            format_func=lambda k: query_options[k],
        )
        q_data = next((q for q in per_query if q["id"] == selected_id), None)

        if q_data:
            ev = q_data.get("eval", {})
            parsed = q_data.get("parsed", {})
            occ = (parsed.get("occasion") or {})
            constraints = (parsed.get("constraints") or {})

            # Query info
            st.markdown(f"### {q_data['query']}")
            info_cols = st.columns(4)
            info_cols[0].markdown(f"**Type:** {', '.join(q_data.get('type', []))}")
            info_cols[1].markdown(f"**Occasion:** `{occ.get('target') or '—'}` ({occ.get('mode', '—')})")
            info_cols[2].markdown(f"**Colors:** {constraints.get('colors') or '—'}")
            info_cols[3].markdown(f"**Fit:** {constraints.get('fit') or '—'}")

            # Parser scores inline
            parser_ev = ev.get("parser", {})
            p_badges = " · ".join(
                f"**{PARSER_LABELS.get(r, r)}:** {parser_ev.get(r, {}).get('score', '—')}/5"
                for r in PARSER_RUBRICS
            )
            st.caption(f"Parser — {p_badges}")

            # Cross-model result
            cm_result = ev.get("cross_model", {})
            winner = cm_result.get("winner", "—")
            winner_display = {"clip": "CLIP", "fashion_clip": "FashionCLIP", "tie": "Tie"}.get(winner, winner)
            st.info(
                f"**Cross-model preference: {winner_display}**  "
                f"·  CLIP {cm_result.get('clip_score', '—')}/5  "
                f"·  FashionCLIP {cm_result.get('fashionclip_score', '—')}/5\n\n"
                f"_{cm_result.get('reasoning', '')}_"
            )

            st.divider()

            # Product cards: CLIP | FashionCLIP side by side
            for model_key, model_label, color in [
                ("clip", "CLIP", MODEL_COLORS["CLIP"]),
                ("fashion_clip", "FashionCLIP", MODEL_COLORS["FashionCLIP"]),
            ]:
                st.markdown(
                    f"<p style='font-size:1.1rem; font-weight:600; "
                    f"color:{color}; margin-bottom:4px'>{model_label}</p>",
                    unsafe_allow_html=True,
                )

                products = (q_data.get("rows_by_model") or {}).get(model_key, [])
                item_scores = (ev.get(model_key) or {}).get("items", [{}] * 5)
                set_score = (ev.get(model_key) or {}).get("set", {}).get("set_answer_relevance", {})

                if not products:
                    st.caption("No results returned for this model.")
                    continue

                set_s = set_score.get("score")
                st.caption(f"Set Answer Relevance: **{_score_badge(set_s)}**")

                product_cols = st.columns(len(products))
                for i, (prod, item_ev) in enumerate(zip(products, item_scores + [{}] * 5)):
                    with product_cols[i]:
                        img_url = prod.get("image_url", "")
                        if img_url:
                            st.image(img_url, width="stretch")
                        else:
                            st.caption("No image")

                        st.markdown(f"**{prod.get('product_name', '—')}**")
                        st.caption(
                            f"{prod.get('product_category', '')} · {prod.get('color', '')}  \n"
                            f"{prod.get('price', '')}"
                        )

                        score_lines = []
                        for r in ITEM_RUBRICS:
                            s = (item_ev.get(r) or {}).get("score")
                            if s is not None:
                                score_lines.append(f"{ITEM_LABELS.get(r, r)[:18]}: **{s}/5**")
                        if score_lines:
                            st.markdown("\n\n".join(score_lines))

                        expl = prod.get("explanation", "")
                        if expl:
                            with st.expander("Explanation"):
                                st.caption(expl)

                st.markdown("")
