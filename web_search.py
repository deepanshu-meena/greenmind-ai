"""
web_search.py
DuckDuckGo search wrapper — no API key required.

The `duckduckgo_search` PyPI package was renamed to `ddgs`
(the old name now just prints a deprecation warning and re-exports
from the new package, but installing `ddgs` directly is the
supported path going forward). We try the new package first and
fall back to the legacy import so this keeps working either way.
"""

import time

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
except ImportError:  # pragma: no cover - legacy environments only
    from duckduckgo_search import DDGS
    try:
        from duckduckgo_search.exceptions import RatelimitException
    except ImportError:
        RatelimitException = Exception


def search_web(query: str, max_results: int = 5) -> str:
    """Return formatted web search results as a string."""
    last_error = None
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return "No results found."

            formatted = ""
            for i, r in enumerate(results, 1):
                formatted += (
                    f"[{i}] {r.get('title', 'No title')}\n"
                    f"{r.get('body', '')}\n"
                    f"Source: {r.get('href', '')}\n\n"
                )
            return formatted.strip()

        except RatelimitException as e:
            # DuckDuckGo's free/unauthenticated endpoint rate-limits
            # aggressively. Back off briefly and retry a couple of
            # times before giving up so a transient 202 doesn't sink
            # the whole report — the RAG + analysis agents can still
            # produce a useful report even without live web results.
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue
        except Exception as e:
            return f"Web search error: {e}"

    return (
        f"Web search error: DuckDuckGo rate-limited this request after "
        f"{3} attempts ({last_error}). Continuing with knowledge-base "
        f"data only for this query."
    )
