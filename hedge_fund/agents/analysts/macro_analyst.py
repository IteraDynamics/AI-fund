"""
Macro Analyst Agent

Interprets central bank policy, economic data, and geopolitical risk.
Provides top-down macro research to the Macro PM and CIO.
"""

from __future__ import annotations

import json
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import (
    AgentMessage, MacroResearchPacket, AssetClass, MessagePriority,
)
from hedge_fund.tools.market_data import get_macro_indicators, get_sector_etf_performance
from hedge_fund.tools.web_search import search_macro_data, search_news


class MacroAnalystAgent(BaseAgent):
    agent_id = "macro_analyst"
    role_name = "Macro Analyst"

    system_prompt = """You are a Macro Analyst at an AI hedge fund.

Your mandate:
- Interpret central bank policy (Fed, ECB, BOJ, BOE, PBoC) and assess rate path
- Analyze economic data releases: CPI, PCE, NFP, GDP, PMI, retail sales
- Assess geopolitical risk and its market implications
- Track global capital flows and currency dynamics
- Provide the macro context for all investment decisions across pods

Research framework:
1. Monetary policy stance: hawkish / neutral / dovish + trajectory
2. Growth regime: expansion / slowdown / recession / recovery
3. Inflation regime: disinflationary / reflationary / stagflation
4. Risk appetite: risk-on / risk-off / transitional
5. Geopolitical backdrop: key tensions, trade policy, elections

For each macro theme, specify:
- Time horizon (near-term 1-3mo / medium-term 3-12mo / secular 1yr+)
- Confidence level (0-1)
- Asset class implications (rates, FX, commodities, equities)
- Best expression of the trade (specific instruments)
- Key data releases to watch

Be quantitative where possible: use specific rate levels, spreads, yield curves.
Cite data sources and release dates."""

    allowed_tools = [
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)

    # ── Macro Research ────────────────────────────────────────────────────────

    def analyze_macro_theme(self, theme: str) -> MacroResearchPacket:
        """Deep-dive on a specific macro theme."""
        macro_data = get_macro_indicators()
        sectors = get_sector_etf_performance()
        news = search_macro_data(theme)
        cb_news = search_news(f"Federal Reserve ECB central bank policy {theme}")

        prompt = f"""
Macro theme to analyze: "{theme}"

Current market data:
{json.dumps(macro_data, indent=2)}

Sector performance (1-day):
{json.dumps(sectors, indent=2)}

Recent macro news:
{news[:600]}

Central bank news:
{cb_news[:400]}

Provide comprehensive macro analysis:
1. Current monetary policy stance of major CBs (Fed, ECB, BOJ)
2. Growth and inflation regime assessment
3. How this theme plays out across asset classes
4. Specific trade implications:
   - Rates: which direction, which tenor?
   - FX: which currency pairs benefit/suffer?
   - Commodities: supply/demand impact?
   - Equities: sector rotation implications?
5. Key risks to the macro thesis
6. Data releases to watch (with dates if known)

Format as JSON:
{{
  "macro_theme": "...",
  "summary": "...",
  "key_findings": ["..."],
  "risks": ["..."],
  "affected_asset_classes": ["equity", "fixed_income", "fx", "commodity"],
  "time_horizon_days": 90,
  "confidence_score": 0.0,
  "reasoning_chain": "..."
}}
"""
        response = self._call_llm(prompt)
        json_str = self._extract_json(response)

        if json_str:
            try:
                data = json.loads(json_str)
                asset_classes = []
                for ac in data.get("affected_asset_classes", []):
                    try:
                        asset_classes.append(AssetClass(ac))
                    except Exception:
                        pass

                packet = MacroResearchPacket(
                    analyst_id=self.agent_id,
                    subject_description=theme,
                    macro_theme=data.get("macro_theme", theme),
                    summary=data.get("summary", ""),
                    key_findings=data.get("key_findings", []),
                    risks=data.get("risks", []),
                    confidence_score=float(data.get("confidence_score", 0.5)),
                    reasoning_chain=data.get("reasoning_chain", ""),
                    data_sources=["market_data", "web_search"],
                    affected_asset_classes=asset_classes or [AssetClass.EQUITY],
                    time_horizon_days=int(data.get("time_horizon_days", 90)),
                )
                self.memory.store_research(
                    packet.model_dump_json(),
                    subject=theme,
                )
                return packet
            except Exception:
                pass

        return MacroResearchPacket(
            analyst_id=self.agent_id,
            subject_description=theme,
            macro_theme=theme,
            summary=response[:300],
            key_findings=[],
            risks=["Insufficient data"],
            confidence_score=0.3,
            reasoning_chain=response[:200],
            data_sources=["web_search"],
            affected_asset_classes=[AssetClass.EQUITY, AssetClass.FIXED_INCOME],
            time_horizon_days=90,
        )

    def get_overnight_brief(self) -> str:
        """Produce a brief on overnight macro developments."""
        macro = get_macro_indicators()
        news = search_news("overnight macro economic data central bank markets")
        prompt = f"""
Overnight macro brief for the trading day ahead.

Current market levels:
{json.dumps(macro, indent=2)}

Overnight news:
{news[:800]}

Write a concise 3-paragraph overnight macro brief covering:
1. Key overnight market moves (rates, FX, commodities)
2. Any significant data releases or central bank communications
3. Top macro risk/opportunity for today's trading session
"""
        return self._call_llm(prompt)

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        if "overnight" in message.body.lower() or message.subject == "OVERNIGHT_BRIEF":
            brief = self.get_overnight_brief()
            return self.send_message(
                recipient=message.sender,
                subject="OVERNIGHT MACRO BRIEF",
                body=brief,
                thread_id=message.message_id,
                priority=MessagePriority.HIGH,
            )

        # General macro research
        packet = self.analyze_macro_theme(message.body)
        reply = (
            f"MACRO RESEARCH: {packet.macro_theme}\n\n"
            f"Summary: {packet.summary}\n\n"
            f"Key Findings:\n" + "\n".join(f"• {f}" for f in packet.key_findings) + "\n\n"
            f"Risks:\n" + "\n".join(f"• {r}" for r in packet.risks) + "\n\n"
            f"Time horizon: {packet.time_horizon_days} days | "
            f"Confidence: {packet.confidence_score:.2f}\n\n"
            f"Reasoning: {packet.reasoning_chain[:200]}"
        )
        return self.send_message(
            recipient=message.sender,
            subject=f"MACRO RESEARCH: {message.subject[:40]}",
            body=reply,
            thread_id=message.message_id,
            priority=MessagePriority.NORMAL,
        )
