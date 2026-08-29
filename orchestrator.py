"""
orchestrator.py
Coordinates the 4-agent pipeline for GreenMind AI.
Returns intermediate + final results for Streamlit display.
"""

from datetime import datetime, timezone

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
        rag_summary, analysis, report, sources_footer
    """
    current_year = datetime.now(timezone.utc).year

    # ── Step 1: Web Search ───────────────────────────────────
    if status_callback:
        status_callback("🔍 Agent 1 — Searching live web for latest data…")

    # Bias the search toward the current and previous year rather than a
    # hardcoded date range — a fixed "2024 2025" string would silently
    # keep steering results toward older content forever.
    web_results    = search_web(
        f"{query} sustainability SDG {current_year} {current_year - 1}"
    )
    search_summary = search_agent(
        query, web_results, hf_token, gemini_key, current_year=current_year
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

    sources_footer = _build_sources_footer(web_results, rag_context, hf_token)

    return {
        "web_results":    web_results,
        "search_summary": search_summary,
        "rag_context":    rag_context,
        "rag_summary":    rag_summary,
        "analysis":       analysis,
        "report":         report,
        "sources_footer": sources_footer,
    }


def _build_sources_footer(web_results: str, rag_context: str, hf_token: str) -> str:
    """
    Build a self-contained provenance/sources block to append to the
    downloadable report — so the .md file a user shares stands on its
    own without needing the Streamlit UI's expandable agent panels to
    show where any claim came from.
    """
    # Pull "Source: <url>" lines out of the raw DuckDuckGo results.
    web_links = []
    for line in web_results.splitlines():
        if line.startswith("Source: ") and line[8:].strip():
            web_links.append(line[8:].strip())

    # Pull "[SDG N – Name] / Source: Wikipedia: ..." lines out of the RAG context.
    kb_sources = []
    lines = rag_context.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Source: Wikipedia:") and i > 0:
            heading = lines[i - 1].strip("[]")
            kb_sources.append(f"{heading} — {line.replace('Source: ', '')}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    footer = "\n\n---\n\n## 📎 Sources & Report Provenance\n\n"
    footer += f"*Generated {timestamp}. LLM: IBM Granite 3.1 8B"
    footer += " (falls back to Gemini automatically if unavailable)" if hf_token else " — Gemini (no HuggingFace token supplied)"
    footer += "*\n\n"

    if web_links:
        footer += "**Live web sources retrieved for this query:**\n"
        for link in dict.fromkeys(web_links):  # de-dupe, keep order
            footer += f"- {link}\n"
        footer += "\n"
    else:
        footer += "**Live web sources:** none retrieved for this query (search may have been rate-limited).\n\n"

    if kb_sources:
        footer += "**SDG knowledge-base articles referenced:**\n"
        for src in dict.fromkeys(kb_sources):
            footer += f"- {src}\n"

    return footer
