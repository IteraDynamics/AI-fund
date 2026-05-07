"""
SEC EDGAR filing retrieval using sec-edgar-downloader.
Returns filing text excerpts for analysis by FundamentalAnalysts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from hedge_fund.config import DATA_DIR


FILINGS_DIR = DATA_DIR / "sec_filings"
FILINGS_DIR.mkdir(exist_ok=True)


def _get_downloader():
    try:
        from edgar import Company
        return Company
    except ImportError:
        return None


def download_filing(
    ticker: str,
    form_type: str = "10-K",
    limit: int = 1,
) -> Optional[Path]:
    """
    Download SEC filing for ticker. Returns path to the downloaded directory.
    form_type: '10-K', '10-Q', '8-K', 'DEF 14A', etc.
    """
    try:
        from sec_edgar_downloader import Downloader
        dl = Downloader(
            company_name="AI Hedge Fund",
            email_address="research@aihedgefund.internal",
            save_path=str(FILINGS_DIR),
        )
        dl.get(form_type, ticker, limit=limit, download_details=True)
        filing_dir = FILINGS_DIR / "sec-edgar-filings" / ticker / form_type
        if filing_dir.exists():
            return filing_dir
    except Exception as e:
        print(f"SEC download error for {ticker} {form_type}: {e}")
    return None


def read_filing_text(filing_path: Path, max_chars: int = 8000) -> str:
    """Read and truncate a filing file to max_chars for LLM context."""
    # Try to find the main filing document
    for suffix in [".txt", ".htm", ".html"]:
        files = list(filing_path.rglob(f"*{suffix}"))
        if files:
            try:
                text = files[0].read_text(encoding="utf-8", errors="ignore")
                # Strip HTML tags if present
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:max_chars]
            except Exception:
                continue
    return ""


def get_filing_summary(ticker: str, form_type: str = "10-K") -> str:
    """
    Download the latest filing and return a text excerpt.
    Returns a string suitable for passing to an LLM.
    """
    filing_dir = download_filing(ticker, form_type)
    if not filing_dir:
        return f"Could not retrieve {form_type} for {ticker}."

    # Find most recent subdirectory
    subdirs = sorted(filing_dir.iterdir(), reverse=True)
    if not subdirs:
        return f"No {form_type} filings found for {ticker}."

    text = read_filing_text(subdirs[0])
    if not text:
        return f"Could not read {form_type} text for {ticker}."

    return f"[{form_type} for {ticker} — excerpt]\n{text}"


def get_recent_8k(ticker: str) -> str:
    """Retrieve the most recent 8-K (material event) for a ticker."""
    return get_filing_summary(ticker, "8-K")


def get_annual_report(ticker: str) -> str:
    """Retrieve the most recent 10-K annual report excerpt."""
    return get_filing_summary(ticker, "10-K")


def get_quarterly_report(ticker: str) -> str:
    """Retrieve the most recent 10-Q quarterly report excerpt."""
    return get_filing_summary(ticker, "10-Q")
