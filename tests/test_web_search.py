"""
Tests for web_search.py — the DuckDuckGo wrapper.

These cover the retry/backoff behaviour around RatelimitException,
which is the one piece of real branching logic in this module.
"""
from unittest.mock import MagicMock, patch

import pytest

from ddgs.exceptions import RatelimitException
import web_search


def _mock_ddgs_context(results):
    """Build a MagicMock that behaves like `with DDGS() as ddgs: ddgs.text(...)`."""
    instance = MagicMock()
    instance.text.return_value = results
    ctx = MagicMock()
    ctx.__enter__.return_value = instance
    ctx.__exit__.return_value = False
    return ctx


def test_search_web_formats_results_correctly(sample_ddgs_results):
    with patch("web_search.DDGS", return_value=_mock_ddgs_context(sample_ddgs_results)):
        result = web_search.search_web("climate action", max_results=5)

    assert "[1] SDG 13: Climate Action" in result
    assert "Source: https://example.com/sdg13" in result
    assert "[2] Renewable Energy Trends 2026" in result
    # No trailing whitespace left over from the formatting loop
    assert result == result.strip()


def test_search_web_no_results_returns_message():
    with patch("web_search.DDGS", return_value=_mock_ddgs_context([])):
        result = web_search.search_web("an extremely obscure query")

    assert result == "No results found."


def test_search_web_handles_missing_fields_gracefully():
    partial_results = [{"title": "Untitled"}]  # no body/href
    with patch("web_search.DDGS", return_value=_mock_ddgs_context(partial_results)):
        result = web_search.search_web("test")

    assert "[1] Untitled" in result
    assert "Source:" in result  # falls back to empty string, doesn't crash


def test_search_web_retries_then_succeeds_after_ratelimit(sample_ddgs_results):
    """First call rate-limited, second call succeeds — should NOT surface an error."""
    good_ctx = _mock_ddgs_context(sample_ddgs_results)

    call_count = {"n": 0}

    def ddgs_side_effect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RatelimitException("rate limited")
        return good_ctx

    with patch("web_search.DDGS", side_effect=ddgs_side_effect), \
         patch("web_search.time.sleep") as mock_sleep:  # don't actually wait in tests
        result = web_search.search_web("climate action")

    assert "SDG 13: Climate Action" in result
    assert mock_sleep.called


def test_search_web_gives_up_after_repeated_ratelimit():
    """All 3 attempts rate-limited -> graceful degraded message, not a raised exception."""
    with patch("web_search.DDGS", side_effect=RatelimitException("still limited")), \
         patch("web_search.time.sleep"):
        result = web_search.search_web("climate action")

    assert "rate-limited" in result.lower()
    assert "knowledge-base" in result.lower()


def test_search_web_generic_exception_returns_error_string_not_raise():
    with patch("web_search.DDGS", side_effect=RuntimeError("boom")):
        result = web_search.search_web("climate action")

    assert result.startswith("Web search error:")
