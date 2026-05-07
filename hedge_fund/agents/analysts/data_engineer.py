"""
Data Engineer Agent

Cleans and pipelines incoming market data, runs data quality checks.
Acts as the data layer for all agents — ensures clean, validated inputs.
"""

from __future__ import annotations

import json
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import AgentMessage, ResearchPacket, MessagePriority
from hedge_fund.tools.market_data import (
    get_price_history, get_fundamentals, get_current_price,
    get_sector_etf_performance, get_macro_indicators,
)


class DataEngineerAgent(BaseAgent):
    agent_id = "data_engineer"
    role_name = "Data Engineer"

    system_prompt = """You are the Data Engineer at an AI hedge fund.

Your mandate:
- Clean, validate, and pipeline all market data
- Run data quality checks: detect stale prices, missing data, outliers, corporate actions
- Provide clean data sets to analysts and PMs on request
- Build and maintain data summaries for the investment team
- Flag any data quality issues immediately

Data quality standards:
- Prices: flag if unchanged for >2 days (stale), flag if >10% daily move (split?)
- Fundamental data: flag if key metrics are missing or implausible
- Time series: check for gaps, ensure adjusted for splits/dividends
- Cross-validation: compare data across sources when discrepancies found

When analysts request data:
- Pull the requested data
- Run quality checks
- Summarize key statistics
- Flag any anomalies
- Return clean, analysis-ready summary

Data sources you use:
- yfinance (primary price data)
- Alpha Vantage (secondary, news sentiment)
- Cached local data (preferred for repeated requests)

Always report data quality score (0-1) with your responses."""

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)

    # ── Data Pipeline Methods ─────────────────────────────────────────────────

    def get_clean_prices(self, tickers: list[str], period: str = "1y") -> dict:
        """Pull and quality-check price data for a list of tickers."""
        results = {}
        quality_flags: list[str] = []

        for ticker in tickers[:10]:  # cap at 10 tickers
            hist = get_price_history(ticker, period=period)
            current = get_current_price(ticker)

            if hist is None:
                quality_flags.append(f"{ticker}: no price data available")
                results[ticker] = None
                continue

            # Quality checks
            closes = [h["close"] for h in hist]
            if len(closes) > 2:
                # Check for stale prices
                last_3 = closes[-3:]
                if len(set(last_3)) == 1:
                    quality_flags.append(f"{ticker}: stale price (unchanged 3 days)")

                # Check for large moves
                if len(closes) >= 2 and closes[-1] > 0 and closes[-2] > 0:
                    daily_move = abs(closes[-1] / closes[-2] - 1)
                    if daily_move > 0.15:
                        quality_flags.append(
                            f"{ticker}: large move {daily_move:.1%} — check for split/error"
                        )

            results[ticker] = {
                "current_price": current,
                "period_high": max(closes) if closes else None,
                "period_low": min(closes) if closes else None,
                "data_points": len(hist),
                "last_date": hist[-1]["date"] if hist else None,
                "30d_return": (closes[-1] / closes[-22] - 1) if len(closes) >= 22 else None,
                "ytd_return": (closes[-1] / closes[0] - 1) if closes else None,
            }

        return {
            "data": results,
            "quality_flags": quality_flags,
            "data_quality_score": max(0.0, 1.0 - len(quality_flags) * 0.1),
        }

    def get_positioning_summary(self, tickers: list[str]) -> str:
        """Summarize key positioning and technical data for a list of tickers."""
        data = self.get_clean_prices(tickers, period="3mo")
        fundamentals = {}
        for ticker in tickers[:5]:
            try:
                fundamentals[ticker] = {
                    k: v for k, v in get_fundamentals(ticker).items()
                    if k in ["sector", "short_ratio", "beta", "market_cap", "pe_ratio"]
                }
            except Exception:
                pass

        prompt = f"""
Data summary requested for: {tickers}

Price data:
{json.dumps(data['data'], indent=2)}

Quality flags: {data['quality_flags']}

Fundamentals:
{json.dumps(fundamentals, indent=2)}

Provide a clean data summary:
1. Which tickers have clean data vs issues?
2. Recent price performance (30d, YTD) for each
3. Key fundamental metrics (PE, short interest, beta)
4. Any notable data anomalies
5. Data quality score: {data['data_quality_score']:.2f}

Be concise — this is for analyst consumption.
"""
        return self._call_llm(prompt)

    def run_data_quality_check(self) -> str:
        """Run a portfolio-wide data quality check."""
        from hedge_fund.memory.shared_state import get_positions
        positions = get_positions()
        tickers = [p.ticker for p in positions]

        if not tickers:
            return "No positions to check."

        data = self.get_clean_prices(tickers)
        if data["quality_flags"]:
            self.send_message(
                recipient="cfo",
                subject="DATA QUALITY ALERT",
                body=f"Data quality issues detected:\n" + "\n".join(data["quality_flags"]),
                priority=MessagePriority.HIGH,
            )

        return (
            f"Data quality check complete. Score: {data['data_quality_score']:.2f}\n"
            f"Flags: {data['quality_flags'] or 'None'}"
        )

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        import re
        tickers = re.findall(r'\b[A-Z]{1,5}\b', message.body)
        exclude = {"PM", "CEO", "CIO", "CRO", "CFO", "COO", "SEC", "DCF", "ETF",
                   "US", "UK", "EU", "EM", "FX", "AI", "ML", "IPO", "M&A", "COT"}
        tickers = [t for t in tickers if t not in exclude][:10]

        if "quality check" in message.body.lower():
            result = self.run_data_quality_check()
        elif tickers:
            result = self.get_positioning_summary(tickers)
        else:
            # General data request
            macro = get_macro_indicators()
            sectors = get_sector_etf_performance()
            result = (
                f"Market snapshot:\n{json.dumps(macro, indent=2)}\n\n"
                f"Sector performance:\n{json.dumps(sectors, indent=2)}"
            )

        return self.send_message(
            recipient=message.sender,
            subject=f"DATA: {message.subject[:40]}",
            body=result,
            thread_id=message.message_id,
            priority=MessagePriority.NORMAL,
        )
