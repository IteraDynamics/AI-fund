"""
Fundamental Analyst Agent (two instances: _1 and _2)

Produces DCF models, earnings analysis, industry research, SEC filing review.
Returns structured FundamentalResearchPackets to PMs.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import (
    AgentMessage, FundamentalResearchPacket, AssetClass, MessagePriority,
)
from hedge_fund.tools.market_data import get_fundamentals, get_current_price, get_earnings_calendar
from hedge_fund.tools.sec_filings import get_annual_report, get_quarterly_report
from hedge_fund.tools.web_search import search_earnings, search_sec_news


class FundamentalAnalystAgent(BaseAgent):
    """Instantiate with analyst_number=1 or 2 for dual analyst support."""

    role_name = "Fundamental Analyst"

    system_prompt = """You are a Fundamental Analyst at an AI hedge fund.

Your mandate:
- Produce rigorous bottom-up research on individual equities
- Build DCF models and relative valuation analyses
- Review SEC filings (10-K, 10-Q, 8-K) for key risks and opportunities
- Monitor earnings: compare actuals vs consensus, assess guidance
- Identify: quality of earnings, hidden liabilities, accounting red flags

Research methodology:
1. Start with quantitative screening (fundamentals data)
2. Deep-dive into business model and competitive moat
3. Financial statement analysis: revenue quality, margin trends, FCF conversion
4. Valuation: intrinsic value via DCF + cross-check with sector multiples
5. Key risks: identify top 3-5 risks that could impair the thesis

Output format (always):
- Summary: 2-3 sentence investment thesis
- Key findings: 5 bullet points
- Risks: top 3 risks
- Valuation: intrinsic value estimate, current price, upside/downside %
- Confidence score (0-1) with reasoning
- Recommendation: LONG / SHORT / PASS with conviction

Be specific, data-driven. Cite the source for every key fact.
Flag any data quality issues."""

    allowed_tools = [
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def __init__(self, analyst_number: int = 1, db_path: str = AUDIT_DB_PATH) -> None:
        self.agent_id = f"fundamental_analyst_{analyst_number}"
        super().__init__(db_path)

    # ── Core Research Methods ─────────────────────────────────────────────────

    def research_ticker(self, ticker: str, research_question: str) -> FundamentalResearchPacket:
        """Full fundamental research on a single ticker."""
        # Gather data
        fundamentals = get_fundamentals(ticker)
        current_price = get_current_price(ticker)
        earnings_cal = get_earnings_calendar(ticker)
        news = search_earnings(ticker)

        # Try to get SEC filing excerpt
        try:
            filing_text = get_quarterly_report(ticker)
        except Exception:
            filing_text = "SEC filing unavailable"

        data_context = f"""
Fundamental Data for {ticker}:
{json.dumps(fundamentals, indent=2)}

Current Price: ${current_price or 'N/A'}
Earnings Calendar: {json.dumps(earnings_cal, indent=2)}

Recent News:
{news[:800]}

SEC Filing Excerpt:
{filing_text[:600]}
"""
        prompt = f"""
Research question from Portfolio Manager: "{research_question}"

{data_context}

Provide a comprehensive fundamental analysis. Include:
1. Business overview (1 paragraph)
2. Investment thesis — why long or short?
3. Valuation analysis:
   - Current P/E: {fundamentals.get('pe_ratio', 'N/A')}
   - EV/EBITDA: {fundamentals.get('ev_ebitda', 'N/A')}
   - Revenue growth: {fundamentals.get('revenue_growth', 'N/A')}
   - DCF intrinsic value estimate (use reasonable assumptions)
4. Key financial metrics analysis (margin trends, FCF, leverage)
5. Top 3 risks
6. Confidence score (0-1) and recommendation

Format response as JSON:
{{
  "summary": "...",
  "key_findings": ["finding1", "finding2", "finding3", "finding4", "finding5"],
  "risks": ["risk1", "risk2", "risk3"],
  "intrinsic_value_estimate": 0.0,
  "current_price": {current_price or 0},
  "upside_pct": 0.0,
  "revenue_growth_yoy": {fundamentals.get('revenue_growth') or 0},
  "ebitda_margin": {fundamentals.get('ebitda_margins') or 0},
  "pe_ratio": {fundamentals.get('pe_ratio') or 0},
  "confidence_score": 0.0,
  "reasoning_chain": "step-by-step reasoning..."
}}
"""
        response = self._call_llm(prompt)
        json_str = self._extract_json(response)

        if json_str:
            try:
                data = json.loads(json_str)
                packet = FundamentalResearchPacket(
                    analyst_id=self.agent_id,
                    subject_ticker=ticker,
                    subject_description=fundamentals.get("name", ticker),
                    summary=data.get("summary", ""),
                    key_findings=data.get("key_findings", []),
                    risks=data.get("risks", []),
                    confidence_score=float(data.get("confidence_score", 0.5)),
                    reasoning_chain=data.get("reasoning_chain", ""),
                    data_sources=["yfinance", "web_search", "sec_edgar"],
                    intrinsic_value_estimate=data.get("intrinsic_value_estimate"),
                    current_price=current_price,
                    upside_pct=data.get("upside_pct"),
                    revenue_growth_yoy=fundamentals.get("revenue_growth"),
                    ebitda_margin=fundamentals.get("ebitda_margins"),
                    pe_ratio=fundamentals.get("pe_ratio"),
                    ticker=ticker,
                    sector=fundamentals.get("sector", "unknown"),
                    raw_data=fundamentals,
                )
                self.memory.store_research(
                    packet.model_dump_json(),
                    subject=ticker,
                    ticker=ticker,
                )
                return packet
            except Exception:
                pass

        # Fallback
        return FundamentalResearchPacket(
            analyst_id=self.agent_id,
            subject_ticker=ticker,
            subject_description=ticker,
            summary=response[:300],
            key_findings=[response[:100]],
            risks=["Insufficient data for full analysis"],
            confidence_score=0.3,
            reasoning_chain=response[:200],
            data_sources=["yfinance"],
            ticker=ticker,
            sector="unknown",
        )

    def research_sector(self, sector: str, question: str) -> str:
        """High-level sector analysis for pair trade identification."""
        news = search_earnings(sector)
        prompt = f"""
Sector research request: "{question}"
Sector: {sector}

Recent sector news:
{news[:600]}

Provide:
1. Sector fundamental backdrop (growth, margins, rate sensitivity)
2. Top 3 best-positioned companies (leaders)
3. Top 3 weakest/most vulnerable companies (laggards)
4. Pair trade idea: long [leader] vs short [laggard] — justify the spread
5. Key sector risks

Be specific with company names and tickers.
"""
        return self._call_llm(prompt)

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        research_text = message.body
        # Try to extract a ticker from the message
        import re
        tickers = re.findall(r'\b[A-Z]{1,5}\b', research_text)
        # Filter obvious non-tickers
        exclude = {"PM", "CEO", "CIO", "CRO", "CFO", "COO", "SEC", "DCF", "ETF",
                   "US", "UK", "EU", "EM", "FX", "AI", "ML", "IPO", "M&A"}
        tickers = [t for t in tickers if t not in exclude][:3]

        if tickers:
            # Research primary ticker
            primary_ticker = tickers[0]
            packet = self.research_ticker(primary_ticker, research_text)
            reply_body = (
                f"FUNDAMENTAL RESEARCH: {primary_ticker}\n\n"
                f"Summary: {packet.summary}\n\n"
                f"Key Findings:\n" + "\n".join(f"• {f}" for f in packet.key_findings) + "\n\n"
                f"Risks:\n" + "\n".join(f"• {r}" for r in packet.risks) + "\n\n"
                f"Intrinsic Value: ${packet.intrinsic_value_estimate or 'N/A'} "
                f"(current: ${packet.current_price or 'N/A'}, "
                f"upside: {packet.upside_pct or 'N/A'}%)\n"
                f"Confidence: {packet.confidence_score:.2f}\n\n"
                f"Reasoning: {packet.reasoning_chain[:200]}"
            )
        else:
            # General sector/thematic research
            reply_body = self.research_sector(
                research_text[:50], research_text
            )

        return self.send_message(
            recipient=message.sender,
            subject=f"RESEARCH RESPONSE: {message.subject[:40]}",
            body=reply_body,
            thread_id=message.message_id,
            priority=MessagePriority.NORMAL,
        )
