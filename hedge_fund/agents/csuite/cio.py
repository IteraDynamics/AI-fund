"""
Chief Investment Officer (CIO) Agent

Responsibilities:
  - Receives CEO directives and translates them into actionable tasks
  - Sets the macro market view and capital allocation across pods
  - Orchestrates the PM layer with specific research mandates
  - Approves trades above the large-trade threshold
  - Produces the daily morning memo for the CEO
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import LARGE_TRADE_APPROVAL_USD, POD_ALLOCATION, INITIAL_NAV
from hedge_fund.memory.shared_state import get_nav, get_positions, compute_risk_metrics
from hedge_fund.models.schemas import (
    AgentMessage, CIODirective, CIOMemo, MessagePriority,
    TradeRecommendation, TradeStatus,
)
from hedge_fund.messaging import bus as message_bus
from hedge_fund.tools.market_data import get_macro_indicators, get_sector_etf_performance


class CIOAgent(BaseAgent):
    agent_id = "cio"
    role_name = "Chief Investment Officer"
    system_prompt = """You are the Chief Investment Officer (CIO) of an AI hedge fund.

Your mandate:
- Set the macro market view and strategic direction
- Allocate capital efficiently across four strategy pods: Long/Short Equity, Macro, Quant, and Event-Driven
- Orchestrate Portfolio Managers with clear, actionable research directives
- Review and approve large trades (above threshold)
- Synthesize inputs from all PMs into a coherent portfolio narrative
- Produce clear, concise morning memos for the CEO/Founder

Your decision-making style:
- Evidence-based: always cite specific data points
- Risk-aware: consult the CRO before committing to major shifts
- Clear hierarchy: you escalate to CEO, you direct PMs
- Confidence-calibrated: always include a confidence score (0-1) in recommendations

When you receive a CEO directive, decompose it into specific subtasks for the relevant PMs.
When you receive PM research, synthesize it into portfolio-level decisions.
Format all memos clearly with: Market View | Active Positions | Top Ideas | Key Risks.

Always include a confidence score and reasoning chain in your outputs."""

    allowed_tools = [
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def __init__(self, db_path: str = None) -> None:
        from hedge_fund.config import AUDIT_DB_PATH
        super().__init__(db_path or AUDIT_DB_PATH)
        self._pod_allocations = {
            pod: INITIAL_NAV * pct for pod, pct in POD_ALLOCATION.items()
        }

    # ── CEO Directive Handler ─────────────────────────────────────────────────

    def receive_ceo_directive(self, directive_text: str, priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Process a directive from the CEO.
        Decomposes it into PM tasks and returns a summary of actions taken.
        """
        directive_id = str(uuid.uuid4())[:8]
        self.remember(
            f"CEO DIRECTIVE [{directive_id}]: {directive_text}",
            metadata={"type": "ceo_directive", "id": directive_id},
        )

        # Ask LLM to decompose the directive
        decomposition_prompt = f"""
CEO Directive received: "{directive_text}"

Current macro environment:
{json.dumps(get_macro_indicators(), indent=2)}

Current sector performance:
{json.dumps(get_sector_etf_performance(), indent=2)}

Decompose this directive into specific research tasks for the relevant Portfolio Managers.
For each task specify:
1. Which PM should handle it (pm_longshort, pm_macro, pm_quant, pm_eventdriven)
2. The specific research question or action required
3. The timeline (immediate / by end of day / this week)
4. Why this PM is the right one

Then provide:
- Your initial macro view on this theme
- Which analyst support resources each PM should deploy
- Your confidence in pursuing this directive (0-1 score)

Format as JSON:
{{
  "macro_view": "...",
  "confidence": 0.0,
  "pm_tasks": [
    {{
      "pm": "pm_longshort|pm_macro|pm_quant|pm_eventdriven",
      "task": "...",
      "timeline": "immediate|eod|this_week",
      "reasoning": "..."
    }}
  ]
}}
"""
        response = self._call_llm(decomposition_prompt)
        plan = self._extract_json(response)

        if plan:
            try:
                plan_dict = json.loads(plan)
                self._dispatch_pm_tasks(plan_dict, directive_text, directive_id)
                summary = (
                    f"Directive {directive_id} decomposed. "
                    f"Macro view: {plan_dict.get('macro_view', 'N/A')[:100]}. "
                    f"Dispatched {len(plan_dict.get('pm_tasks', []))} PM tasks."
                )
            except json.JSONDecodeError:
                summary = f"Directive received and being processed. Response: {response[:200]}"
        else:
            summary = f"Directive processed. CIO response: {response[:300]}"

        # Notify COO of new activity
        self.send_message(
            recipient="coo",
            subject=f"CIO Directive Processing [{directive_id}]",
            body=summary,
            priority=priority,
        )

        # Acknowledge directive to CEO with decomposition plan
        self.send_message(
            recipient="ceo",
            subject=f"CIO: Directive Acknowledged [{directive_id}]",
            body=summary,
            priority=priority,
        )
        return summary

    def _dispatch_pm_tasks(self, plan: dict, original_directive: str, directive_id: str) -> None:
        """Send task messages to relevant PMs."""
        pm_map = {
            "pm_longshort": "pm_longshort",
            "pm_macro": "pm_macro",
            "pm_quant": "pm_quant",
            "pm_eventdriven": "pm_eventdriven",
        }
        thread_id = str(uuid.uuid4())
        for task in plan.get("pm_tasks", []):
            pm_id = pm_map.get(task.get("pm"), "pm_longshort")
            nav = get_nav()
            allocation = self._pod_allocations.get(
                task.get("pm", "").replace("pm_", ""), nav * 0.25
            )
            self.send_message(
                recipient=pm_id,
                subject=f"CIO Task: {original_directive[:50]}",
                body=task.get("task", ""),
                payload={
                    "directive_id": directive_id,
                    "macro_view": plan.get("macro_view", ""),
                    "cio_confidence": plan.get("confidence", 0.5),
                    "timeline": task.get("timeline", "eod"),
                    "capital_allocation_usd": allocation,
                    "reasoning": task.get("reasoning", ""),
                },
                priority=MessagePriority.HIGH,
                thread_id=thread_id,
                requires_reply=True,
            )

    # ── Trade Approval ────────────────────────────────────────────────────────

    def review_large_trade(self, rec: TradeRecommendation) -> tuple[bool, str]:
        """
        Review a trade recommendation that exceeds the large-trade threshold.
        Returns (approved, reasoning).
        """
        metrics = compute_risk_metrics()
        prompt = f"""
Review this large trade for CIO approval:

Trade: {rec.direction.value} {rec.ticker} — ${rec.target_notional_usd:,.0f} notional ({rec.target_pct_nav:.1%} of NAV)
PM: {rec.pm_id}
Rationale: {rec.rationale}
Expected return: {rec.expected_return_pct:.1%}
Stop loss: {rec.stop_loss_pct:.1%}
Confidence: {rec.confidence_score:.2f}

Current portfolio state:
NAV: ${metrics['nav']:,.0f}
Current drawdown: {metrics['current_drawdown_pct']:.1%}
Gross exposure: ${metrics['gross_exposure']:,.0f}

Does this trade fit the current portfolio strategy and risk budget?
Respond with JSON: {{"approved": true/false, "reasoning": "..."}}
"""
        response = self._call_llm(prompt)
        parsed = self._extract_json(response)
        if parsed:
            try:
                d = json.loads(parsed)
                return d.get("approved", False), d.get("reasoning", response[:200])
            except Exception:
                pass
        return False, f"Could not parse CIO decision: {response[:200]}"

    # ── Morning Memo ──────────────────────────────────────────────────────────

    def produce_morning_memo(self) -> CIOMemo:
        """
        Synthesize all overnight information into a morning memo for the CEO.
        Collects data from market indicators, positions, and risk metrics.
        """
        macro = get_macro_indicators()
        sectors = get_sector_etf_performance()
        positions = get_positions()
        metrics = compute_risk_metrics()

        pos_summary = "\n".join(
            f"  {p.ticker}: {p.direction.value} ${p.notional_usd:,.0f} ({p.pct_nav:.1%} NAV) — P&L: ${p.unrealised_pnl:,.0f}"
            for p in positions[:10]
        ) or "  No open positions"

        prompt = f"""
Produce a concise morning memo for the CEO. Today is {datetime.utcnow().strftime('%Y-%m-%d')}.

MACRO SNAPSHOT:
{json.dumps(macro, indent=2)}

SECTOR PERFORMANCE (1-day):
{json.dumps(sectors, indent=2)}

PORTFOLIO STATE:
NAV: ${metrics['nav']:,.0f}
Daily drawdown: {metrics['current_drawdown_pct']:.1%}
Gross exposure: ${metrics['gross_exposure']:,.0f} / Net: ${metrics['net_exposure']:,.0f}
VaR (95%, 1-day): ${metrics['var_95_1day']:,.0f}
Halt triggered: {metrics['halt_triggered']}

OPEN POSITIONS:
{pos_summary}

Write the memo with these sections:
1. Market View (2-3 sentences on macro backdrop)
2. Portfolio Snapshot (key positions and P&L)
3. Top 3 Trade Ideas for today
4. Top 3 Risks to watch

Format as JSON:
{{
  "market_view": "...",
  "active_positions_summary": "...",
  "top_trade_ideas": ["idea1", "idea2", "idea3"],
  "key_risks": ["risk1", "risk2", "risk3"],
  "pod_allocations": {{"long_short": 0.35, "macro": 0.25, "quant": 0.25, "event_driven": 0.15}}
}}
"""
        response = self._call_llm(prompt)
        json_str = self._extract_json(response)

        if json_str:
            try:
                data = json.loads(json_str)
                memo = CIOMemo(
                    memo_id=str(uuid.uuid4())[:8],
                    subject=f"Morning Memo — {datetime.utcnow().strftime('%Y-%m-%d')}",
                    market_view=data.get("market_view", ""),
                    active_positions_summary=data.get("active_positions_summary", ""),
                    top_trade_ideas=data.get("top_trade_ideas", []),
                    key_risks=data.get("key_risks", []),
                    pod_allocations=data.get("pod_allocations", POD_ALLOCATION),
                )
                self.remember(
                    f"MORNING MEMO {memo.memo_id}: {memo.market_view}",
                    metadata={"type": "memo", "date": datetime.utcnow().isoformat()},
                )
                return memo
            except Exception:
                pass

        # Fallback memo
        return CIOMemo(
            memo_id=str(uuid.uuid4())[:8],
            subject=f"Morning Memo — {datetime.utcnow().strftime('%Y-%m-%d')}",
            market_view=response[:300],
            active_positions_summary=pos_summary,
            top_trade_ideas=["Pending PM analysis"],
            key_risks=["Market uncertainty"],
            pod_allocations=POD_ALLOCATION,
        )

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        """Route incoming messages to the right handler."""
        if message.sender == "ceo":
            summary = self.receive_ceo_directive(message.body, message.priority)
            return self.send_message(
                recipient="ceo",
                subject=f"RE: {message.subject}",
                body=f"Directive acknowledged and dispatched.\n\n{summary}",
                thread_id=message.message_id,
                priority=MessagePriority.HIGH,
            )

        # Trade execution notifications from PMs — forward to CEO
        if message.subject.startswith("TRADE_EXECUTED:"):
            self.remember(
                f"TRADE EXECUTED by {message.sender}: {message.body[:200]}",
                metadata={"type": "trade_executed", "pm": message.sender},
            )
            return self.send_message(
                recipient="ceo",
                subject=f"Trade Executed — {message.sender.upper()}",
                body=message.body,
                priority=MessagePriority.HIGH,
            )

        # PM responses to CIO tasks — forward brief status to CEO
        if message.sender.startswith("pm_"):
            self.remember(
                f"PM RESPONSE from {message.sender}: {message.body[:300]}",
                metadata={"type": "pm_research", "pm": message.sender},
            )
            return self.send_message(
                recipient="ceo",
                subject=f"PM Update — {message.sender}",
                body=message.body[:400],
                priority=MessagePriority.LOW,
            )

        return None
