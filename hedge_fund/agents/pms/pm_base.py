"""
Shared base class for all Portfolio Manager agents.

PMs sit between the CIO (capital allocator) and analysts (research providers).
They:
  1. Receive capital allocation + research mandates from the CIO
  2. Task their specific analysts with targeted questions
  3. Synthesize analyst research into sized trade recommendations
  4. Submit recommendations through the compliance → risk → execution pipeline
  5. Track and manage their pod's open positions
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import (
    MAX_POSITION_SIZE_PCT, LARGE_TRADE_APPROVAL_USD,
    INITIAL_NAV, AUDIT_DB_PATH,
)
from hedge_fund.memory.shared_state import (
    get_nav, get_positions, save_recommendation,
    update_recommendation_status,
)
from hedge_fund.models.schemas import (
    AgentMessage, TradeRecommendation, TradeStatus,
    AssetClass, Direction, OrderType, MessagePriority,
)


class BasePMAgent(BaseAgent):
    """Shared functionality for all PM agents."""

    pod_name: str = "base_pod"
    analyst_ids: list[str] = []

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)
        self._capital_allocation: float = INITIAL_NAV * 0.25
        self._pending_research: dict[str, str] = {}  # message_id → research text

    # ── Task Analysts ─────────────────────────────────────────────────────────

    def task_analyst(
        self,
        analyst_id: str,
        research_question: str,
        context: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> str:
        """Send a targeted research question to an analyst. Returns message_id."""
        body = research_question
        if context:
            body = f"{context}\n\nResearch question: {research_question}"

        msg_id = self.send_message(
            recipient=analyst_id,
            subject=f"PM Research Request: {research_question[:60]}",
            body=body,
            payload={"pod": self.pod_name, "pm": self.agent_id},
            priority=priority,
            requires_reply=True,
        )
        self._pending_research[msg_id] = analyst_id
        return msg_id

    # ── Trade Recommendation Builder ──────────────────────────────────────────

    def build_trade_recommendation(
        self,
        ticker: str,
        asset_class: AssetClass,
        direction: Direction,
        target_pct_nav: float,
        rationale: str,
        expected_return_pct: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        time_horizon_days: int,
        confidence_score: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> TradeRecommendation:
        """Construct and persist a TradeRecommendation."""
        nav = get_nav()
        target_pct_nav = min(target_pct_nav, MAX_POSITION_SIZE_PCT)
        target_notional = nav * target_pct_nav

        rec = TradeRecommendation(
            rec_id=str(uuid.uuid4())[:12],
            pm_id=self.agent_id,
            ticker=ticker,
            asset_class=asset_class,
            direction=direction,
            order_type=order_type,
            limit_price=limit_price,
            target_notional_usd=target_notional,
            target_pct_nav=target_pct_nav,
            rationale=rationale,
            confidence_score=confidence_score,
            expected_return_pct=expected_return_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            time_horizon_days=time_horizon_days,
            status=TradeStatus.PENDING_COMPLIANCE,
        )
        save_recommendation(rec)
        return rec

    # ── Trade Pipeline ────────────────────────────────────────────────────────

    def submit_for_compliance(self, rec: TradeRecommendation) -> None:
        """Send trade recommendation to Compliance for screening."""
        self.send_message(
            recipient="compliance_agent",
            subject=f"COMPLIANCE_CHECK:{rec.rec_id}",
            body=f"{rec.direction.value} {rec.ticker} ${rec.target_notional_usd:,.0f} — {rec.rationale[:100]}",
            payload=rec.model_dump(mode="json"),
            priority=MessagePriority.HIGH,
            requires_reply=True,
        )

    def submit_for_risk(self, rec: TradeRecommendation) -> None:
        """Send trade recommendation to CRO for risk check."""
        self.send_message(
            recipient="cro",
            subject=f"RISK_CHECK:{rec.rec_id}",
            body=f"{rec.direction.value} {rec.ticker} ${rec.target_notional_usd:,.0f}",
            payload=rec.model_dump(mode="json"),
            priority=MessagePriority.HIGH,
            requires_reply=True,
        )

    def submit_for_cio_approval(self, rec: TradeRecommendation) -> None:
        """Escalate large trade to CIO for approval."""
        self.send_message(
            recipient="cio",
            subject=f"LARGE_TRADE_APPROVAL:{rec.rec_id}",
            body=f"Trade requires CIO approval: {rec.direction.value} {rec.ticker} ${rec.target_notional_usd:,.0f}",
            payload=rec.model_dump(mode="json"),
            priority=MessagePriority.HIGH,
            requires_reply=True,
        )

    def execute_trade(self, rec: TradeRecommendation) -> None:
        """
        Simulate trade execution after all approvals.
        Updates blotter and positions in shared state.
        """
        from hedge_fund.tools.market_data import get_current_price
        from hedge_fund.memory.shared_state import (
            record_trade, upsert_position, get_position,
        )
        from hedge_fund.models.schemas import BlotterEntry, Position

        fill_price = get_current_price(rec.ticker)
        if fill_price is None:
            fill_price = 100.0  # fallback for testing

        nav = get_nav()
        quantity = rec.target_notional_usd / fill_price

        # Create blotter entry
        entry = BlotterEntry(
            trade_id=str(uuid.uuid4())[:12],
            rec_id=rec.rec_id,
            ticker=rec.ticker,
            direction=rec.direction,
            quantity=quantity,
            fill_price=fill_price,
            notional_usd=rec.target_notional_usd,
            pod=self.pod_name,
            pm_id=self.agent_id,
        )
        record_trade(entry)

        # Update position
        existing = get_position(rec.ticker)
        if existing and existing.direction == rec.direction:
            # Average in
            total_qty = existing.quantity + quantity
            avg_price = (
                (existing.avg_entry_price * existing.quantity + fill_price * quantity)
                / total_qty
            )
            existing.quantity = total_qty
            existing.avg_entry_price = avg_price
            existing.current_price = fill_price
            existing.notional_usd = fill_price * total_qty
            existing.unrealised_pnl = (fill_price - avg_price) * total_qty if rec.direction == Direction.LONG else (avg_price - fill_price) * total_qty
            existing.pct_nav = existing.notional_usd / nav if nav > 0 else 0
            upsert_position(existing)
        else:
            pos = Position(
                ticker=rec.ticker,
                asset_class=rec.asset_class,
                direction=rec.direction,
                quantity=quantity,
                avg_entry_price=fill_price,
                current_price=fill_price,
                notional_usd=rec.target_notional_usd,
                unrealised_pnl=0.0,
                pct_nav=rec.target_pct_nav,
                pod=self.pod_name,
            )
            upsert_position(pos)

        update_recommendation_status(rec.rec_id, TradeStatus.EXECUTED.value)

        # Notify CIO
        self.send_message(
            recipient="cio",
            subject=f"TRADE_EXECUTED:{rec.rec_id}",
            body=f"Executed: {rec.direction.value} {rec.ticker} x{quantity:.0f} @ ${fill_price:.2f}",
            priority=MessagePriority.NORMAL,
        )

    # ── LLM-driven Synthesis ──────────────────────────────────────────────────

    def synthesize_into_trade(
        self,
        research_texts: list[str],
        mandate: str,
        asset_class: AssetClass,
    ) -> Optional[TradeRecommendation]:
        """
        Use the LLM to synthesize research into a trade recommendation.
        Returns None if no actionable trade is identified.
        """
        combined_research = "\n\n---\n\n".join(research_texts[:5])
        nav = get_nav()
        positions = get_positions()
        current_tickers = [p.ticker for p in positions]

        prompt = f"""
You are {self.role_name} reviewing research for the {self.pod_name} pod.

Pod mandate: {mandate}
Available capital: ${self._capital_allocation:,.0f}
Current portfolio NAV: ${nav:,.0f}
Current pod positions: {current_tickers}

Research received:
{combined_research[:3000]}

Based on this research, determine if there is an actionable trade.
Consider: conviction level, risk/reward, position sizing (max 5% NAV = ${nav*0.05:,.0f}).

If actionable, respond with JSON:
{{
  "actionable": true,
  "ticker": "TICKER",
  "direction": "long|short",
  "target_pct_nav": 0.03,
  "rationale": "...",
  "expected_return_pct": 0.15,
  "stop_loss_pct": 0.05,
  "take_profit_pct": 0.20,
  "time_horizon_days": 30,
  "confidence_score": 0.7
}}

If not actionable:
{{"actionable": false, "reason": "..."}}
"""
        response = self._call_llm(prompt)
        json_str = self._extract_json(response)
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
            if not data.get("actionable", False):
                return None

            return self.build_trade_recommendation(
                ticker=data["ticker"],
                asset_class=asset_class,
                direction=Direction(data["direction"]),
                target_pct_nav=float(data.get("target_pct_nav", 0.02)),
                rationale=data.get("rationale", "LLM synthesis"),
                expected_return_pct=float(data.get("expected_return_pct", 0.10)),
                stop_loss_pct=float(data.get("stop_loss_pct", 0.05)),
                take_profit_pct=float(data.get("take_profit_pct", 0.15)),
                time_horizon_days=int(data.get("time_horizon_days", 30)),
                confidence_score=float(data.get("confidence_score", 0.5)),
            )
        except Exception:
            return None

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        """Handle incoming messages: CIO tasks and analyst research responses."""

        # CIO task assignment
        if message.sender == "cio" and message.payload:
            self._capital_allocation = float(
                message.payload.get("capital_allocation_usd", self._capital_allocation)
            )
            task_text = message.body
            macro_view = message.payload.get("macro_view", "")

            self.remember(
                f"CIO TASK: {task_text[:200]} | MACRO: {macro_view[:100]}",
                metadata={"type": "cio_task"},
            )

            # Have PM run the task
            response = self.run_pm_task(task_text, macro_view)
            return self.send_message(
                recipient="cio",
                subject=f"RE: {message.subject}",
                body=response,
                thread_id=message.message_id,
                priority=MessagePriority.NORMAL,
            )

        # Analyst research response
        if message.sender.endswith("_analyst") or message.sender in [
            "fundamental_analyst_1", "fundamental_analyst_2",
            "quant_analyst", "macro_analyst", "data_engineer",
        ]:
            self._pending_research[message.message_id] = message.body
            self.remember(
                f"ANALYST RESEARCH from {message.sender}: {message.body[:300]}",
                metadata={"type": "analyst_research", "analyst": message.sender},
            )
            return None

        # Compliance / Risk results
        if message.subject.startswith("COMPLIANCE_RESULT:"):
            rec_id = message.subject.split(":")[1]
            if message.payload and message.payload.get("approved"):
                update_recommendation_status(rec_id, TradeStatus.PENDING_RISK.value)
                # Get the rec and send to CRO
                recs = [r for r in __import__(
                    'hedge_fund.memory.shared_state', fromlist=['get_recommendations']
                ).get_recommendations(status=TradeStatus.PENDING_RISK.value)
                        if r['rec_id'] == rec_id]
                if recs:
                    from hedge_fund.models.schemas import TradeRecommendation
                    rec = TradeRecommendation.model_validate(recs[0])
                    self.submit_for_risk(rec)
            return None

        if message.subject.startswith("RISK_RESULT:"):
            rec_id = message.subject.split(":")[1]
            if message.payload and message.payload.get("approved"):
                update_recommendation_status(rec_id, TradeStatus.APPROVED.value)
                from hedge_fund.memory.shared_state import get_recommendations
                from hedge_fund.models.schemas import TradeRecommendation
                recs = [r for r in get_recommendations(status=TradeStatus.APPROVED.value)
                        if r['rec_id'] == rec_id]
                if recs:
                    rec = TradeRecommendation.model_validate(recs[0])
                    # Check if needs CIO approval for size
                    if rec.target_notional_usd > LARGE_TRADE_APPROVAL_USD:
                        self.submit_for_cio_approval(rec)
                    else:
                        self.execute_trade(rec)
            return None

        return None

    def run_pm_task(self, task: str, macro_view: str) -> str:
        """Override in subclasses to implement pod-specific task logic."""
        return self.run_task(task, context=f"Macro view from CIO: {macro_view}")
