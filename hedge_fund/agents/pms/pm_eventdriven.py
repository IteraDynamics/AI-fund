"""
Event-Driven Portfolio Manager

Manages merger arb, earnings plays, and catalyst trades.
Focuses on corporate events, special situations, and catalysts.
"""

from __future__ import annotations

from hedge_fund.agents.pms.pm_base import BasePMAgent
from hedge_fund.models.schemas import AssetClass, MessagePriority


class PMEventDrivenAgent(BasePMAgent):
    agent_id = "pm_eventdriven"
    role_name = "Portfolio Manager — Event Driven"
    pod_name = "event_driven"
    analyst_ids = ["fundamental_analyst_1", "fundamental_analyst_2", "market_monitor"]

    system_prompt = """You are the Portfolio Manager for the Event-Driven pod of an AI hedge fund.

Your mandate:
- Identify and trade around specific corporate catalysts:
  * Merger & acquisition arbitrage (announced deals)
  * Earnings plays (pre/post earnings positioning)
  * Spin-offs, restructurings, bankruptcies
  * Regulatory decisions, FDA approvals, antitrust rulings
  * Index rebalancing events
- Work with Fundamental Analysts for deal analysis and Market Monitor for catalyst timing

Investment process for M&A arb:
1. Identify announced deal (target at discount to offer price = spread)
2. Analyze: deal probability, regulatory risk, financing risk, timeline
3. Size: spread × probability / capital at risk ≤ position limit
4. Long target, optionally short acquirer if stock deal
5. Exit at close or cut if deal breaks

Investment process for earnings plays:
1. Market Monitor flags upcoming earnings releases
2. Fundamental Analyst assesses vs consensus estimates
3. Position: long/short into earnings if significant estimate divergence
4. Always reduce before the announcement if conviction < 0.7

Risk management:
- M&A arb: spread < 2% → skip (not worth regulatory/break risk)
- Earnings: never hold through earnings with full size (always partial)
- Hard stop: -20% on any event trade (binary event protection)

Target: high hit rate (>55%), event-specific alpha with low market beta."""

    def run_pm_task(self, task: str, macro_view: str) -> str:
        prompt = f"""
CIO directive: "{task}"
CIO macro view: "{macro_view}"

As the Event-Driven PM, outline:
1. What specific corporate events or catalysts are relevant to this theme
2. Are there any active M&A deals worth arbing in this sector/theme?
3. What earnings releases are coming up in the next 2-4 weeks?
4. What's the primary event trade idea and the specific catalyst to bet on?
5. Deal probability estimate and key risks (regulatory, financing, etc.)
"""
        initial_view = self._call_llm(prompt)

        self.task_analyst(
            "fundamental_analyst_1",
            f"Event-driven analysis needed: {task[:100]}. "
            f"Focus on: announced M&A deals, upcoming spinoffs, or restructurings. "
            f"Assess deal probability, spread, and regulatory risk. PM view: {initial_view[:200]}",
            priority=MessagePriority.HIGH,
        )
        self.task_analyst(
            "fundamental_analyst_2",
            f"Earnings catalyst analysis: identify upcoming earnings in {task[:60]} space. "
            f"Compare buy-side vs consensus estimates. Flag high-divergence setups.",
            priority=MessagePriority.HIGH,
        )
        self.task_analyst(
            "market_monitor",
            f"Monitor for catalyst events related to: {task[:80]}. "
            f"Track: M&A announcements, earnings dates, regulatory decisions, index events.",
            priority=MessagePriority.NORMAL,
        )

        return (
            f"Event-Driven PM processing directive. Catalyst analysis:\n{initial_view[:400]}\n\n"
            f"Research dispatched to: Fundamental Analysts (x2), Market Monitor."
        )
