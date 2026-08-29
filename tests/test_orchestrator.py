"""
Tests for orchestrator.py — the 4-agent pipeline coordinator and
the sources-footer builder.
"""
from unittest.mock import MagicMock, patch

import pytest

import orchestrator


def test_run_pipeline_calls_all_four_stages_in_order():
    call_order = []

    def track(name):
        def _inner(*args, **kwargs):
            call_order.append(name)
            return f"{name}_result"
        return _inner

    with patch("orchestrator.search_web", side_effect=track("search_web")), \
         patch("orchestrator.query_knowledge_base", side_effect=track("query_kb")), \
         patch("orchestrator.search_agent", side_effect=track("search_agent")), \
         patch("orchestrator.rag_agent", side_effect=track("rag_agent")), \
         patch("orchestrator.analysis_agent", side_effect=track("analysis_agent")), \
         patch("orchestrator.writer_agent", side_effect=track("writer_agent")):
        result = orchestrator.run_pipeline(
            query="What is SDG 7?",
            collection=MagicMock(),
            hf_token="hf-token",
            gemini_key="gemini-key",
        )

    # Web search + KB query must both complete before analysis/writing,
    # and analysis must run before the writer.
    assert call_order.index("search_web") < call_order.index("search_agent")
    assert call_order.index("query_kb") < call_order.index("rag_agent")
    assert call_order.index("analysis_agent") < call_order.index("writer_agent")
    assert result["report"] == "writer_agent_result"


def test_run_pipeline_invokes_status_callback_for_each_step():
    statuses = []

    with patch("orchestrator.search_web", return_value="web"), \
         patch("orchestrator.query_knowledge_base", return_value="rag"), \
         patch("orchestrator.search_agent", return_value="s"), \
         patch("orchestrator.rag_agent", return_value="r"), \
         patch("orchestrator.analysis_agent", return_value="a"), \
         patch("orchestrator.writer_agent", return_value="w"):
        orchestrator.run_pipeline(
            query="test",
            collection=MagicMock(),
            hf_token="hf",
            gemini_key="g",
            status_callback=statuses.append,
        )

    assert len(statuses) == 4  # one status update per agent stage


def test_run_pipeline_returns_all_expected_keys():
    with patch("orchestrator.search_web", return_value="web"), \
         patch("orchestrator.query_knowledge_base", return_value="rag ctx"), \
         patch("orchestrator.search_agent", return_value="s"), \
         patch("orchestrator.rag_agent", return_value="r"), \
         patch("orchestrator.analysis_agent", return_value="a"), \
         patch("orchestrator.writer_agent", return_value="w"):
        result = orchestrator.run_pipeline(
            query="test", collection=MagicMock(), hf_token="hf", gemini_key="g"
        )

    for key in ("web_results", "search_summary", "rag_context",
                "rag_summary", "analysis", "report", "sources_footer"):
        assert key in result


# ── _build_sources_footer ────────────────────────────────────

def test_sources_footer_extracts_web_links(sample_web_results_text):
    footer = orchestrator._build_sources_footer(sample_web_results_text, "", hf_token="hf")

    assert "https://example.com/sdg13" in footer
    assert "https://example.com/renewables" in footer


def test_sources_footer_extracts_kb_sources(sample_rag_context):
    footer = orchestrator._build_sources_footer("", sample_rag_context, hf_token="hf")

    assert "SDG 13 – Climate Action" in footer
    assert "Wikipedia: Climate_change" in footer


def test_sources_footer_dedupes_repeated_links():
    web_results = (
        "[1] Title A\nBody\nSource: https://example.com/a\n\n"
        "[2] Title A duplicate\nBody\nSource: https://example.com/a\n\n"
    )
    footer = orchestrator._build_sources_footer(web_results, "", hf_token="hf")

    assert footer.count("https://example.com/a") == 1


def test_sources_footer_notes_no_web_links_when_none_found():
    footer = orchestrator._build_sources_footer("No results found.", "", hf_token="hf")

    assert "none retrieved" in footer.lower()


def test_sources_footer_reflects_granite_vs_gemini_in_message():
    footer_with_hf = orchestrator._build_sources_footer("", "", hf_token="hf-token")
    footer_without_hf = orchestrator._build_sources_footer("", "", hf_token="")

    assert "falls back to Gemini" in footer_with_hf
    assert "no HuggingFace token supplied" in footer_without_hf
