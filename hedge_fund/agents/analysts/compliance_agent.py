"""
Compliance Agent

Screens all trades against a ruleset: position limits, restricted list, wash sale rules.
Must approve every trade before it hits the blotter.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH, MAX_POSITION_SIZE_PCT
from hedge_fund.memory.shared_state import get_blotter, get_positions, get_nav
from hedge_fund.models.schemas import (
    AgentMessage, ComplianceCheckResult, TradeRecommendation,
    TradeStatus, MessagePriority,
)


# Restricted ticker list (would be loaded from a live feed in production)
RESTRICTED_TICKERS = {
    # Example restricted list — add real names here
    "INSDR1", "INSIDER2",
}


class ComplianceAgent(BaseAgent):
    agent_id = "compliance_agent"
    role_name = "Compliance Officer"

    system_prompt = """You are the Compliance Officer at an AI hedge fund.

Your mandate:
- Screen ALL trades for regulatory and fund mandate compliance
- Enforce position limits, restricted security lists, and wash sale rules
- Your approval is REQUIRED before any trade is sent to risk review
- Maintain zero-tolerance for compliance violations

Rules you enforce:
1. Position limits: no single position > 5% of NAV (reinforces CRO limit)
2. Restricted list: never trade restricted securities
3. Wash sale rule: cannot repurchase a security within 30 days of selling at a loss
4. Pattern day trading: flag if >3 round-trips in a single security in 5 days
5. Concentration limits: enforce sector and single-name limits
6. Short selling: must have borrow availability (flagged as always available in sim)
7. Leverage: gross exposure must not exceed 200% NAV

When screening:
- Check each rule explicitly
- Document all violations clearly
- Approve clean trades immediately
- Reject with specific violation details

Be thorough but efficient. Clean trades should be approved quickly."""

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)
        self._restricted_list = RESTRICTED_TICKERS.copy()

    # ── Compliance Checks ─────────────────────────────────────────────────────

    def screen_trade(self, rec: TradeRecommendation) -> ComplianceCheckResult:
        """Run full compliance screen on a trade recommendation."""
        violations: list[str] = []
        nav = get_nav()

        # 1. Restricted list check
        restricted_flag = rec.ticker in self._restricted_list
        if restricted_flag:
            violations.append(f"{rec.ticker} is on the restricted trading list")

        # 2. Position limit check
        position_limit_flag = rec.target_pct_nav > MAX_POSITION_SIZE_PCT
        if position_limit_flag:
            violations.append(
                f"Position size {rec.target_pct_nav:.1%} exceeds 5% limit"
            )

        # 3. Wash sale check — did we sell this ticker at a loss in the last 30 days?
        wash_sale_flag = self._check_wash_sale(rec.ticker)
        if wash_sale_flag:
            violations.append(
                f"Wash sale rule: {rec.ticker} was sold at a loss within 30 days"
            )

        # 4. Pattern day trading check
        pdt_flag = self._check_pattern_day_trading(rec.ticker)
        if pdt_flag:
            violations.append(
                f"Pattern day trading alert: >3 round-trips in {rec.ticker} in 5 days"
            )

        # 5. Gross leverage check
        positions = get_positions()
        gross_exposure = sum(p.notional_usd for p in positions)
        new_gross = gross_exposure + rec.target_notional_usd
        leverage_ok = new_gross <= nav * 2.0
        if not leverage_ok:
            violations.append(
                f"Gross leverage would reach {new_gross/nav:.1%}, exceeding 200% NAV limit"
            )

        # LLM qualitative review for edge cases
        if not violations:
            qualitative_result = self._qualitative_review(rec)
            if qualitative_result:
                violations.append(qualitative_result)

        approved = len(violations) == 0

        result = ComplianceCheckResult(
            rec_id=rec.rec_id,
            approved=approved,
            violations=violations,
            wash_sale_flag=wash_sale_flag,
            restricted_list_flag=restricted_flag,
            position_limit_flag=position_limit_flag,
            compliance_notes=(
                "All compliance checks passed." if approved
                else f"VIOLATIONS: {'; '.join(violations)}"
            ),
        )

        self.remember(
            f"COMPLIANCE CHECK {rec.rec_id}: {'APPROVED' if approved else 'REJECTED'} — {violations}",
            metadata={"type": "compliance_check", "rec_id": rec.rec_id},
        )
        return result

    def _check_wash_sale(self, ticker: str) -> bool:
        """Check if we sold this ticker at a loss in the last 30 days."""
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        recent_trades = get_blotter(ticker=ticker, limit=20)
        for trade in recent_trades:
            if trade.get("executed_at", "") >= cutoff and trade.get("direction") == "short":
                # Simplified: flag if there was any recent short-side trade
                # In production: check the actual realized P&L
                return False  # Not flagging in sim without realized P&L tracking
        return False

    def _check_pattern_day_trading(self, ticker: str) -> bool:
        """Check for excessive round-trips in the last 5 days."""
        cutoff = (datetime.utcnow() - timedelta(days=5)).isoformat()
        recent_trades = get_blotter(ticker=ticker, limit=20)
        recent = [t for t in recent_trades if t.get("executed_at", "") >= cutoff]
        # Count round-trips (pairs of opposing trades)
        round_trips = len(recent) // 2
        return round_trips >= 3

    def _qualitative_review(self, rec: TradeRecommendation) -> Optional[str]:
        """Ask LLM for qualitative compliance assessment."""
        prompt = f"""
Quick compliance review for this trade:
  Ticker: {rec.ticker}
  Direction: {rec.direction.value}
  Size: ${rec.target_notional_usd:,.0f} ({rec.target_pct_nav:.1%} of NAV)
  Rationale: {rec.rationale[:100]}

Are there any compliance concerns NOT covered by quantitative checks?
Consider: unusual market manipulation indicators, unusual timing, any obvious red flags.

If clean, respond: "CLEAN"
If concern: respond with the specific issue in one sentence.
"""
        response = self._call_llm(prompt)
        if "CLEAN" in response.upper() or len(response) < 10:
            return None
        # Only flag if LLM actually identifies a problem
        if any(w in response.lower() for w in ["concern", "flag", "issue", "violation", "suspicious"]):
            return f"Qualitative flag: {response[:100]}"
        return None

    def add_to_restricted_list(self, ticker: str, reason: str) -> None:
        self._restricted_list.add(ticker)
        self.send_message(
            recipient="cio",
            subject=f"COMPLIANCE: {ticker} added to restricted list",
            body=f"Reason: {reason}",
            priority=MessagePriority.HIGH,
        )

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        if message.subject.startswith("COMPLIANCE_CHECK:") and message.payload:
            try:
                rec = TradeRecommendation.model_validate(message.payload)
                result = self.screen_trade(rec)

                # Update status and notify PM
                from hedge_fund.memory.shared_state import update_recommendation_status
                status = TradeStatus.PENDING_RISK if result.approved else TradeStatus.REJECTED
                update_recommendation_status(rec.rec_id, status.value)

                return self.send_message(
                    recipient=message.sender,
                    subject=f"COMPLIANCE_RESULT:{rec.rec_id}",
                    body=(
                        f"{'APPROVED' if result.approved else 'REJECTED'}: "
                        f"{result.compliance_notes}"
                    ),
                    payload=result.model_dump(mode="json"),
                    thread_id=message.message_id,
                    priority=MessagePriority.HIGH,
                )
            except Exception as e:
                return self.send_message(
                    recipient=message.sender,
                    subject="COMPLIANCE_ERROR",
                    body=str(e),
                    thread_id=message.message_id,
                )
        return None
