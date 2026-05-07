"""
Quant Portfolio Manager

Manages systematic/factor strategies.
Runs factor models, backtests, and systematic signals.
"""

from __future__ import annotations

from hedge_fund.agents.pms.pm_base import BasePMAgent
from hedge_fund.models.schemas import AssetClass, MessagePriority


class PMQuantAgent(BasePMAgent):
    agent_id = "pm_quant"
    role_name = "Portfolio Manager — Quantitative Strategies"
    pod_name = "quant"
    analyst_ids = ["quant_analyst", "data_engineer"]

    system_prompt = """You are the Portfolio Manager for the Quantitative Strategies pod of an AI hedge fund.

Your mandate:
- Run systematic, factor-based investment strategies
- Manage a diversified portfolio of quantitative signals: momentum, value, quality, low-vol, size
- Ensure strategies are properly backtested before deployment
- Monitor live signal performance vs backtest expectations
- Work with QuantAnalyst for signal research and DataEngineer for data pipelines

Investment process:
1. QuantAnalyst proposes a signal with full backtest results
2. PM reviews: out-of-sample performance, turnover, capacity, correlation to existing signals
3. Size allocation based on information ratio and diversification benefit
4. Monitor live vs backtest; stop signal if live IR drops > 30% below backtest IR
5. Rebalance monthly or when signal decays

Current active strategies:
- US Equity Momentum (12-1 month momentum, monthly rebal)
- Quality Factor (high ROE, low leverage, earnings stability)
- Short-term Mean Reversion (5-day reversal in large caps)
- Cross-sectional Volatility (long low-vol, short high-vol)

Target: diversified factor exposure, Sharpe > 1.2, turnover < 50% monthly
Always require full backtest before any systematic strategy goes live."""

    def run_pm_task(self, task: str, macro_view: str) -> str:
        prompt = f"""
CIO directive: "{task}"
CIO macro view: "{macro_view}"

As the Quant PM, outline:
1. Which systematic factors or signals are most relevant
2. What backtesting or signal research is needed
3. How to construct a systematic portfolio around this theme
4. Implementation: universe, rebalancing frequency, transaction costs
5. Expected Sharpe, drawdown, and turnover estimates
"""
        initial_view = self._call_llm(prompt)

        self.task_analyst(
            "quant_analyst",
            f"Research request from Quant PM: {task[:100]}. "
            f"Please build and backtest a signal for this opportunity. "
            f"Include: factor construction, backtest (5yr+), Sharpe, max DD, turnover. "
            f"Initial PM hypothesis: {initial_view[:200]}",
            priority=MessagePriority.HIGH,
        )
        self.task_analyst(
            "data_engineer",
            f"Data pipeline needed for quant strategy: {task[:80]}. "
            f"Provide clean daily returns, factor exposures, and corporate actions adjustment.",
            priority=MessagePriority.NORMAL,
        )

        return (
            f"Quant PM processing directive. Systematic approach:\n{initial_view[:400]}\n\n"
            f"Research dispatched to: QuantAnalyst, Data Engineer."
        )
