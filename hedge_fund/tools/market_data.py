"""
Market data tools: price quotes, fundamentals, historical data.
Uses yfinance as primary source with Alpha Vantage as fallback.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Optional

import requests
import yfinance as yf

from hedge_fund.config import ALPHA_VANTAGE_API_KEY


def get_current_price(ticker: str, retries: int = 3) -> Optional[float]:
    """Return latest close price for a ticker."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="2d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    retries: int = 3,
) -> Optional[list[dict]]:
    """Return list of OHLCV dicts for a ticker."""
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            if hist.empty:
                return None
            records = []
            for dt, row in hist.iterrows():
                records.append({
                    "date": dt.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                })
            return records
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def get_fundamentals(ticker: str) -> dict:
    """Return key fundamental data for an equity ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "ticker": ticker,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "revenue_ttm": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "net_income": info.get("netIncomeToCommon"),
            "free_cash_flow": info.get("freeCashflow"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "gross_margins": info.get("grossMargins"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "profit_margins": info.get("profitMargins"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "short_ratio": info.get("shortRatio"),
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationMean"),
            "description": info.get("longBusinessSummary", "")[:500],
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def get_earnings_calendar(ticker: str) -> dict:
    """Return upcoming earnings date and analyst estimates."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return {"ticker": ticker, "earnings_date": None}
        result = {"ticker": ticker}
        for col in cal.columns:
            result[col] = str(cal[col].iloc[0]) if not cal[col].empty else None
        return result
    except Exception:
        return {"ticker": ticker, "earnings_date": None}


def get_sector_etf_performance() -> dict[str, float]:
    """Return 1-day returns for major sector ETFs."""
    sector_etfs = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication": "XLC",
    }
    perf = {}
    for sector, etf in sector_etfs.items():
        try:
            t = yf.Ticker(etf)
            hist = t.history(period="2d")
            if len(hist) >= 2:
                ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100
                perf[sector] = round(ret, 2)
        except Exception:
            pass
    return perf


def get_macro_indicators() -> dict:
    """Return prices/levels for key macro instruments."""
    tickers = {
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        "VIX": "^VIX",
        "10Y_TREASURY": "^TNX",
        "2Y_TREASURY": "^IRX",
        "GOLD": "GC=F",
        "CRUDE_OIL": "CL=F",
        "USD_INDEX": "DX-Y.NYB",
        "EUR_USD": "EURUSD=X",
        "GBP_USD": "GBPUSD=X",
        "USD_JPY": "JPY=X",
    }
    data = {}
    for name, sym in tickers.items():
        price = get_current_price(sym)
        if price is not None:
            data[name] = price
    return data


def alpha_vantage_news_sentiment(ticker: str) -> Optional[dict]:
    """Return news sentiment from Alpha Vantage (requires API key)."""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}&limit=10"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_options_data(ticker: str) -> Optional[dict]:
    """Return near-term options chain summary (IV, put/call ratio)."""
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None
        exp = expirations[0]  # nearest expiry
        chain = t.option_chain(exp)
        calls = chain.calls
        puts = chain.puts

        total_call_oi = calls["openInterest"].sum() if not calls.empty else 0
        total_put_oi = puts["openInterest"].sum() if not puts.empty else 0
        pc_ratio = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

        atm_iv = None
        if not calls.empty:
            price = get_current_price(ticker)
            if price:
                atm = calls.iloc[(calls["strike"] - price).abs().argsort()[:1]]
                atm_iv = atm["impliedVolatility"].values[0] if not atm.empty else None

        return {
            "nearest_expiry": exp,
            "put_call_ratio": round(pc_ratio, 3) if pc_ratio else None,
            "atm_implied_vol": round(atm_iv * 100, 2) if atm_iv else None,
        }
    except Exception:
        return None
