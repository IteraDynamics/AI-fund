"""
Long/Short Equity Portfolio Manager

Manages the equity long/short book.
Focuses on: stock picking, pair trades, sector rotation within equities.
"""

from __future__ import annotations

import json
from typing import Optional

from hedge_fund.agents.pms.pm_base import BasePMAgent
from hedge_fund.models.schemas import AssetClass, MessagePriority


class PMLongShortAgent(BasePMAgent):
    agent_id = "pm_longshort"
    role_name = "Portfolio Manager — Long/Short Equity"
    pod_name = "long_short"
    analyst_ids = ["fundamental_analyst_1", "fundamental_analyst_2", "data_engineer"]

    system_prompt = """You are the Portfolio Manager for the Long/Short Equity pod of an AI hedge fund.

Your mandate:
- Manage a market-neutral to net-long equity book (target net exposure: -20% to +60%)
- Generate alpha through stock selection: longs in high-quality growth, shorts in value traps/deteriorating fundamentals
- Run pair trades within sectors to isolate stock-specific alpha
- Sector rotation based on macro regime
- Work with Fundamental Analysts for bottom-up research and Data Engineer for positioning data

Investment process:
1. Screen for opportunities (top-down sector view, then bottom-up stock selection)
2. Task fundamental analysts with deep-dive research
3. Size positions: 1-3% NAV for high-conviction, 0.5-1% for speculative
4. Set clear stop losses (5-8%) and take profits (15-25%)
5. Monitor pair trade convergence

You target: Sharpe ratio > 1.5, max drawdown < 12%
Always provide confidence scores (0-1) and clear reasoning chains."""

    def run_pm_task(self, task: str, macro_view: str) -> str:
        """Process a CIO task: decompose into analyst questions and synthesize."""
        # First, ask our analysts for research
        prompt = f"""
You received a CIO directive: "{task}"
CIO macro view: "{macro_view}"

As the Long/Short Equity PM, outline:
1. Which specific stocks or sectors to research
2. What questions to ask the Fundamental Analysts
3. What pair trade opportunities exist if any
4. Your initial hypothesis (long/short which names and why)

Be specific about tickers. Consider US equity universe.
"""
        initial_view = self._call_llm(prompt)

        # Task the fundamental analysts
        self.task_analyst(
            "fundamental_analyst_1",
            f"Deep-dive fundamental analysis requested by PM. Context: {task[:100]}. "
            f"Initial hypothesis: {initial_view[:200]}. "
            "Provide: valuation (DCF/comparables), earnings quality, key risks, price target.",
            priority=MessagePriority.HIGH,
        )
        self.task_analyst(
            "fundamental_analyst_2",
            f"Comparative industry analysis for equity opportunity: {task[:100]}. "
            "Identify best-in-class vs laggards in the relevant sector for pair trade construction.",
            priority=MessagePriority.NORMAL,
        )
        self.task_analyst(
            "data_engineer",
            f"Pull positioning data, short interest, and recent insider transactions for "
            f"relevant names in: {task[:80]}",
            priority=MessagePriority.NORMAL,
        )

        # Return initial PM view to CIO
        return (
            f"Long/Short PM processing directive. Initial view:\n{initial_view[:400]}\n\n"
            f"Research dispatched to: Fundamental Analysts (x2), Data Engineer. "
            f"Will synthesize into trade recommendations when research returns."
        )
