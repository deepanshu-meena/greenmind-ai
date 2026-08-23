"""
orchestrator.py
Coordinates the 4-agent pipeline for GreenMind AI.
Returns intermediate + final results for Streamlit display.
"""

from web_search   import search_web
from knowledge_base import query_knowledge_base
from agents import (
    search_agent, rag_agent,
    analysis_agent, writer_agent,
)


def run_pipeline(query: str,
                 collection,
                 hf_token: str,
                 gemini_key: str,
                 status_callback=None) -> dict:
    """
    Runs the full 4-agent pipeline.

    Args:
        query          : user's sustainability question
        collection     : ChromaDB collection (pre-loaded KB)
        hf_token       : HuggingFace token for IBM Granite
        gemini_key     : Gemini API key (fallback)
        status_callback: optional fn(step: str) for UI updates

    Returns dict with keys:
        web_results, search_summary, rag_context,
        rag_summary, analysis, report
    """

    # ── Step 1: Web Search ───────────────────────────────────
    if status_callback:
        status_callback("🔍 Agent 1 — Searching live web for latest data…")

    web_results    = search_web(f"{query} sustainability SDG 2024 2025")
    search_summary = search_agent(
        query, web_results, hf_token, gemini_key
    )

    # ── Step 2: RAG Knowledge Base ───────────────────────────
    if status_callback:
        status_callback("📚 Agent 2 — Querying SDG knowledge base (RAG)…")

    rag_context = query_knowledge_base(collection, query, gemini_key)
    rag_summary = rag_agent(
        query, rag_context, hf_token, gemini_key
    )

    # ── Step 3: Cross-Analysis ───────────────────────────────
    if status_callback:
        status_callback("🔬 Agent 3 — Analysing & cross-checking sources…")

    analysis = analysis_agent(
        query, search_summary, rag_summary, hf_token, gemini_key
    )

    # ── Step 4: Write Final Report ───────────────────────────
    if status_callback:
        status_callback("✍️  Agent 4 — Writing SDG Intelligence Report…")

    report = writer_agent(
        query, search_summary, rag_summary,
        analysis, hf_token, gemini_key
    )

    return {
        "web_results":    web_results,
        "search_summary": search_summary,
        "rag_context":    rag_context,
        "rag_summary":    rag_summary,
        "analysis":       analysis,
        "report":         report,
    }
