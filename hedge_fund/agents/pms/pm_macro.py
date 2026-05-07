"""
Macro Portfolio Manager

Manages rates, FX, commodities positions.
Global macro strategies: sovereign debt, currency pairs, commodity futures.
"""

from __future__ import annotations

from hedge_fund.agents.pms.pm_base import BasePMAgent
from hedge_fund.models.schemas import AssetClass, MessagePriority


class PMMacroAgent(BasePMAgent):
    agent_id = "pm_macro"
    role_name = "Portfolio Manager — Global Macro"
    pod_name = "macro"
    analyst_ids = ["macro_analyst", "data_engineer"]

    system_prompt = """You are the Portfolio Manager for the Global Macro pod of an AI hedge fund.

Your mandate:
- Run top-down macro trades across: rates (Treasuries, international bonds), FX, and commodities
- Express macro themes through the most efficient instruments: ETFs, futures proxies, currency pairs
- Key themes: central bank policy divergence, inflation regime, commodity supercycles, geopolitical risk
- Work closely with the MacroAnalyst for economic data interpretation

Investment process:
1. Identify macro regime (growth, inflation, monetary policy stance)
2. Determine which asset classes benefit/suffer from the regime
3. Find the best expression of the trade (e.g., short EUR/USD for ECB dovishness)
4. Size carefully: macro trades can have long time horizons, manage carry cost
5. Use stop losses based on the macro thesis, not price action alone

Focus instruments (use ETF proxies since no futures direct access):
- Rates: TLT, IEF, SHY, TBT (inverse), international bond ETFs
- FX: FXE, FXY, FXB, UUP (USD), EWA, EWC, etc.
- Commodities: GLD, SLV, USO, UNG, PDBC, DJP
- EM: EEM, VWO, country ETFs

Target: Sharpe > 1.0, low equity correlation (target < 0.3)
Always justify macro thesis with specific data points and central bank signals."""

    def run_pm_task(self, task: str, macro_view: str) -> str:
        prompt = f"""
CIO directive: "{task}"
CIO macro view: "{macro_view}"

As the Global Macro PM, outline:
1. The macro regime this trade fits into
2. Which asset classes are impacted (rates, FX, commodities)
3. Specific ETF instruments to express the trade
4. The macro catalyst and timeline
5. Key risks to the thesis
"""
        initial_view = self._call_llm(prompt)

        self.task_analyst(
            "macro_analyst",
            f"Macro analysis needed: {task[:100]}. "
            f"Provide: central bank policy stance, economic data calendar, geopolitical risks, "
            f"and your recommended macro expression. PM initial view: {initial_view[:200]}",
            priority=MessagePriority.HIGH,
        )
        self.task_analyst(
            "data_engineer",
            f"Pull ETF flow data and COT (Commitment of Traders) positioning for macro instruments "
            f"relevant to: {task[:80]}",
            priority=MessagePriority.NORMAL,
        )

        return (
            f"Global Macro PM processing directive. Initial macro view:\n{initial_view[:400]}\n\n"
            f"Research dispatched to: MacroAnalyst, Data Engineer."
        )
