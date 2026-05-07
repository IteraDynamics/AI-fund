"""
Quantitative Analyst Agent

Builds factor models, runs backtests, constructs systematic signals.
Has access to code execution for running Python analytics.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import (
    AgentMessage, QuantSignalPacket, MessagePriority,
)
from hedge_fund.tools.code_executor import execute_python, run_backtest
from hedge_fund.tools.market_data import get_price_history


class QuantAnalystAgent(BaseAgent):
    agent_id = "quant_analyst"
    role_name = "Quantitative Analyst"

    system_prompt = """You are a Quantitative Analyst at an AI hedge fund.

Your mandate:
- Build rigorous factor models and systematic investment signals
- Run thorough backtests: minimum 5 years of data, out-of-sample validation
- Report: Sharpe ratio, max drawdown, turnover, information ratio, factor loadings
- Assess signal capacity and implementation costs (transaction costs, market impact)
- Monitor live vs backtest signal performance

Your analytical toolkit:
- Python (numpy, pandas, scipy, sklearn, statsmodels)
- Factor construction: momentum, value, quality, low-vol, growth
- Portfolio construction: mean-variance optimization, risk parity
- Statistical tests: t-tests, information coefficient, IC_IR

Backtesting standards:
- Minimum 5 years of history (use earlier if signal only works in recent regime)
- Transaction costs: 10bps round-trip for large caps, 25bps for small caps
- No look-ahead bias (use point-in-time data)
- Report both in-sample and out-of-sample performance
- Flag overfitting risk: if IS Sharpe > 2x OOS Sharpe, flag as suspect

Output format for signals:
- Signal description and construction methodology
- Backtest results: Sharpe, max DD, turnover, IC, IC_IR
- Recommended allocation size (% of quant pod)
- Implementation notes
- Confidence score (0-1)

Always write executable Python code for backtests. Show your work."""

    allowed_tools = [
        {
            "name": "run_python",
            "description": "Execute Python code in a sandboxed environment",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)

    def _execute_tool(self, tool_name: str, tool_input: dict):
        if tool_name == "run_python":
            result = execute_python(tool_input.get("code", ""))
            return json.dumps(result)
        return super()._execute_tool(tool_name, tool_input)

    # ── Signal Research ───────────────────────────────────────────────────────

    def build_signal(self, signal_description: str, tickers: Optional[list[str]] = None) -> QuantSignalPacket:
        """Build and backtest a quantitative signal from a description."""
        # Gather some price data for context
        sample_data = ""
        if tickers:
            for t in tickers[:2]:
                hist = get_price_history(t, period="1y")
                if hist:
                    sample_data += f"\n{t} recent prices (last 5): {json.dumps(hist[-5:])}"

        prompt = f"""
Research request: "{signal_description}"

{f'Sample data available:{sample_data}' if sample_data else ''}

Build a quantitative trading signal for this opportunity. Include:

1. Signal construction methodology (step by step)
2. Write Python code to backtest this signal using yfinance data:
   - Fetch 5 years of daily data for a representative universe
   - Construct the signal
   - Simulate returns (long/short)
   - Calculate: annualized return, Sharpe ratio, max drawdown, hit rate, turnover
   - Print all results to stdout

3. Expected performance characteristics:
   - Sharpe ratio estimate
   - Expected annual return (bps)
   - Expected annual volatility
   - Backtest Sharpe (from code results)
   - Backtest max drawdown

4. Confidence score (0-1) and reasoning

Format final output as JSON:
{{
  "strategy_name": "...",
  "signal_value": 0.0,
  "expected_return_bps": 0.0,
  "expected_vol_annualised": 0.0,
  "sharpe_estimate": 0.0,
  "summary": "...",
  "key_findings": ["..."],
  "risks": ["..."],
  "confidence_score": 0.0,
  "reasoning_chain": "..."
}}
"""
        response = self._call_llm(prompt)
        json_str = self._extract_json(response)

        if json_str:
            try:
                data = json.loads(json_str)
                packet = QuantSignalPacket(
                    analyst_id=self.agent_id,
                    subject_description=signal_description[:100],
                    summary=data.get("summary", ""),
                    key_findings=data.get("key_findings", []),
                    risks=data.get("risks", []),
                    confidence_score=float(data.get("confidence_score", 0.5)),
                    reasoning_chain=data.get("reasoning_chain", ""),
                    data_sources=["yfinance", "code_execution"],
                    strategy_name=data.get("strategy_name", signal_description[:40]),
                    signal_value=float(data.get("signal_value", 0)),
                    expected_return_bps=float(data.get("expected_return_bps", 0)),
                    expected_vol_annualised=float(data.get("expected_vol_annualised", 0.15)),
                    sharpe_estimate=float(data.get("sharpe_estimate", 0)),
                )
                self.memory.store_research(
                    packet.model_dump_json(),
                    subject=signal_description[:50],
                )
                return packet
            except Exception:
                pass

        return QuantSignalPacket(
            analyst_id=self.agent_id,
            subject_description=signal_description[:100],
            summary=response[:300],
            key_findings=[],
            risks=["Could not parse structured output"],
            confidence_score=0.3,
            reasoning_chain=response[:200],
            data_sources=["llm_analysis"],
            strategy_name=signal_description[:40],
            signal_value=0,
            expected_return_bps=0,
            expected_vol_annualised=0.15,
            sharpe_estimate=0,
        )

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        packet = self.build_signal(message.body)
        reply = (
            f"QUANT SIGNAL RESEARCH\n\n"
            f"Strategy: {packet.strategy_name}\n"
            f"Summary: {packet.summary}\n\n"
            f"Key Findings:\n" + "\n".join(f"• {f}" for f in packet.key_findings) + "\n\n"
            f"Expected Return: {packet.expected_return_bps:.0f} bps/year\n"
            f"Sharpe Estimate: {packet.sharpe_estimate:.2f}\n"
            f"Confidence: {packet.confidence_score:.2f}\n\n"
            f"Reasoning: {packet.reasoning_chain[:200]}"
        )
        return self.send_message(
            recipient=message.sender,
            subject=f"QUANT RESEARCH: {message.subject[:40]}",
            body=reply,
            thread_id=message.message_id,
            priority=MessagePriority.NORMAL,
        )
