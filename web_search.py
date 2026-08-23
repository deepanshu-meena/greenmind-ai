"""
web_search.py
DuckDuckGo search wrapper — no API key required.
"""

from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """Return formatted web search results as a string."""
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

    except Exception as e:
        return f"Web search error: {e}"
