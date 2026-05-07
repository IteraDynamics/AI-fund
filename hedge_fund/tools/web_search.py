"""
Web search tool using the Anthropic API's built-in web_search capability.
Wraps the tool as a callable Python function agents can use directly.
"""

from __future__ import annotations

from typing import Optional
import anthropic

from hedge_fund.config import ANTHROPIC_API_KEY, CLAUDE_MODEL


_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def web_search(query: str, max_results: int = 5) -> str:
    """
    Execute a web search via the Anthropic claude-sonnet-4-20250514 web_search tool.
    Returns a text summary of the search results.
    """
    client = _get_client()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Search for: {query}\n\n"
                    f"Return a concise summary of the top {max_results} results "
                    "with key facts, dates, and sources."
                ),
            }],
        )
        # Extract text from the response (may include tool use blocks)
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts) if text_parts else "No results returned."
    except Exception as e:
        return f"Web search failed: {str(e)}"


def search_news(topic: str, days_back: int = 7) -> str:
    """Search for recent news on a topic."""
    query = f"{topic} news last {days_back} days financial markets"
    return web_search(query)


def search_earnings(ticker: str) -> str:
    """Search for recent earnings news and analyst reactions."""
    query = f"{ticker} earnings results analyst reaction revenue EPS"
    return web_search(query)


def search_macro_data(indicator: str) -> str:
    """Search for latest macro economic data release."""
    query = f"{indicator} latest data release Federal Reserve economic"
    return web_search(query)


def search_sec_news(ticker: str) -> str:
    """Search for recent SEC filings and regulatory news."""
    query = f"{ticker} SEC filing 8-K 10-Q 10-K regulatory disclosure"
    return web_search(query)
