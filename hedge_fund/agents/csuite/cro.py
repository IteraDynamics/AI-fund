"""
Chief Risk Officer (CRO) Agent

Responsibilities:
  - Portfolio-wide VaR, drawdown limit, correlation exposure monitoring
  - Veto power on any individual trade
  - Morning risk report to CEO
  - Triggers trading halt if drawdown breaches the limit
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import (
    MAX_POSITION_SIZE_PCT, MAX_SECTOR_CONCENTRATION_PCT,
    MAX_PORTFOLIO_DRAWDOWN_PCT, AUDIT_DB_PATH,
)
from hedge_fund.memory.shared_state import (
    compute_risk_metrics, get_positions, get_nav, is_halted, set_halt,
    save_risk_check,
)
from hedge_fund.models.schemas import (
    AgentMessage, RiskCheckResult, PortfolioRiskSummary,
    TradeRecommendation, MessagePriority,
)


class CROAgent(BaseAgent):
    agent_id = "cro"
    role_name = "Chief Risk Officer"
    system_prompt = """You are the Chief Risk Officer (CRO) of an AI hedge fund.

Your mandate:
- Monitor and control portfolio-wide risk at all times
- Run rigorous risk checks on every proposed trade before execution
- You have ABSOLUTE VETO POWER on any trade — your decision is final
- Trigger trading halts when drawdown limits are breached
- Produce clear risk reports for the CEO

Your hard limits (non-negotiable):
- Maximum single position size: 5% of NAV
- Maximum sector concentration: 20% of NAV
- Maximum portfolio drawdown: 15% (triggers a full trading halt)
- Flag high correlation trades (adding to crowded positions)

Risk assessment framework:
1. Position size check (absolute limit)
2. Sector concentration check
3. Portfolio drawdown check (if breach → halt)
4. Correlation check (new position vs existing book)
5. Overall VaR impact

Always explain your reasoning. When vetoing, be specific about the limit breached.
Include confidence in your assessments (0-1). Err on the side of caution."""

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)

    # ── Core Risk Check ───────────────────────────────────────────────────────

    def check_trade(self, rec: TradeRecommendation) -> RiskCheckResult:
        """
        Run a full risk check on a trade recommendation.
        This is the CRO's primary function.
        """
        metrics = compute_risk_metrics()
        nav = metrics["nav"]
        positions = get_positions()

        # Hard limit checks
        position_size_ok = rec.target_pct_nav <= MAX_POSITION_SIZE_PCT

        # Sector concentration — check if adding this position breaches sector limit
        sector = next(
            (p.sector for p in positions if p.ticker == rec.ticker), "unknown"
        )
        current_sector_pct = metrics["sector_breakdown"].get(sector or "unknown", 0)
        new_sector_pct = current_sector_pct + rec.target_pct_nav
        sector_concentration_ok = new_sector_pct <= MAX_SECTOR_CONCENTRATION_PCT

        drawdown_within_limit = (
            metrics["current_drawdown_pct"] <= MAX_PORTFOLIO_DRAWDOWN_PCT
        )

        # Check if halt is already triggered
        if metrics["halt_triggered"] or not drawdown_within_limit:
            set_halt(True)
            return RiskCheckResult(
                rec_id=rec.rec_id,
                approved=False,
                veto_reason="TRADING HALT: Portfolio drawdown limit breached.",
                portfolio_var_95=metrics["var_95_1day"],
                position_size_ok=position_size_ok,
                sector_concentration_ok=sector_concentration_ok,
                drawdown_within_limit=False,
                correlation_flag=False,
                cro_notes="All trading suspended due to drawdown limit breach.",
            )

        # Ask LLM for qualitative assessment and correlation check
        existing_tickers = [p.ticker for p in positions]
        prompt = f"""
Risk Assessment for Trade:
  Ticker: {rec.ticker}
  Direction: {rec.direction.value}
  Notional: ${rec.target_notional_usd:,.0f} ({rec.target_pct_nav:.1%} of NAV)
  PM: {rec.pm_id}
  Rationale: {rec.rationale[:200]}
  Expected return: {rec.expected_return_pct:.1%}, Stop loss: {rec.stop_loss_pct:.1%}

Portfolio context:
  NAV: ${nav:,.0f}
  Current drawdown: {metrics['current_drawdown_pct']:.1%} (limit: {MAX_PORTFOLIO_DRAWDOWN_PCT:.0%})
  Current VaR 95%: ${metrics['var_95_1day']:,.0f}
  Largest position: {metrics['largest_position_pct']:.1%} (limit: {MAX_POSITION_SIZE_PCT:.0%})
  Position size check: {'PASS' if position_size_ok else 'FAIL'}
  Sector concentration check: {'PASS' if sector_concentration_ok else 'FAIL'} (current: {current_sector_pct:.1%} → {new_sector_pct:.1%})
  Existing positions: {', '.join(existing_tickers[:10]) or 'none'}

Hard limit checks:
  - Position size: {'✓ PASS' if position_size_ok else '✗ FAIL — exceeds 5% limit'}
  - Sector concentration: {'✓ PASS' if sector_concentration_ok else '✗ FAIL — exceeds 20% limit'}
  - Drawdown: {'✓ PASS' if drawdown_within_limit else '✗ FAIL — breach'}

Assess:
1. Is there high correlation between {rec.ticker} and the existing book? (flag if yes)
2. Does the risk/reward look reasonable given the portfolio context?
3. Your overall recommendation (approve/veto) and why

Respond as JSON:
{{
  "approved": true/false,
  "veto_reason": null or "reason",
  "correlation_flag": true/false,
  "cro_notes": "your assessment"
}}
"""
        response = self._call_llm(prompt)
        parsed = self._extract_json(response)

        approved = position_size_ok and sector_concentration_ok and drawdown_within_limit
        veto_reason = None
        correlation_flag = False
        cro_notes = response[:300]

        if not position_size_ok:
            veto_reason = f"Position size {rec.target_pct_nav:.1%} exceeds 5% limit"
            approved = False
        elif not sector_concentration_ok:
            veto_reason = f"Sector concentration would reach {new_sector_pct:.1%}, exceeding 20% limit"
            approved = False

        if parsed:
            try:
                d = json.loads(parsed)
                # CRO LLM can also veto
                if not d.get("approved", True) and approved:
                    approved = False
                    veto_reason = d.get("veto_reason", "CRO qualitative veto")
                correlation_flag = d.get("correlation_flag", False)
                cro_notes = d.get("cro_notes", cro_notes)
            except Exception as e:
                self._log("error", f"CRO JSON parse failed for {rec.rec_id}: {e} | raw: {parsed[:100]}")
        else:
            self._log("error", f"CRO could not extract JSON for {rec.rec_id} — using hard-limit result only")

        result = RiskCheckResult(
            rec_id=rec.rec_id,
            approved=approved,
            veto_reason=veto_reason,
            portfolio_var_95=metrics["var_95_1day"],
            position_size_ok=position_size_ok,
            sector_concentration_ok=sector_concentration_ok,
            drawdown_within_limit=drawdown_within_limit,
            correlation_flag=correlation_flag,
            cro_notes=cro_notes,
        )

        save_risk_check(result)
        self.remember(
            f"RISK CHECK {rec.rec_id}: {'APPROVED' if approved else 'VETOED'} — {veto_reason or cro_notes[:100]}",
            metadata={"type": "risk_check", "rec_id": rec.rec_id, "approved": str(approved)},
        )
        return result

    # ── Morning Risk Report ───────────────────────────────────────────────────

    def produce_risk_report(self) -> PortfolioRiskSummary:
        """Generate the daily portfolio risk summary."""
        metrics = compute_risk_metrics()
        positions = get_positions()

        prompt = f"""
Produce a concise risk narrative for the morning risk report.

Portfolio metrics:
{json.dumps(metrics, indent=2, default=str)}

Number of open positions: {len(positions)}

In 2-3 sentences summarize:
1. The current risk posture (elevated/normal/low)
2. The biggest risk concentration
3. Any emerging risks to watch

Return just the narrative text (no JSON needed).
"""
        narrative = self._call_llm(prompt)

        report = PortfolioRiskSummary(
            as_of=datetime.utcnow(),
            nav=metrics["nav"],
            gross_exposure=metrics["gross_exposure"],
            net_exposure=metrics["net_exposure"],
            var_95_1day=metrics["var_95_1day"],
            current_drawdown_pct=metrics["current_drawdown_pct"],
            largest_position_pct=metrics["largest_position_pct"],
            top_sector_concentration_pct=metrics["top_sector_concentration_pct"],
            halt_triggered=metrics["halt_triggered"],
            sector_breakdown=metrics["sector_breakdown"],
            pod_pnl=metrics["pod_pnl"],
        )

        # Send report to CEO and CIO
        report_text = (
            f"RISK REPORT — {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"NAV: ${report.nav:,.0f} | Drawdown: {report.current_drawdown_pct:.1%} | "
            f"VaR(95%): ${report.var_95_1day:,.0f}\n"
            f"Halt: {'YES ⚠️' if report.halt_triggered else 'No'}\n\n"
            f"Narrative: {narrative[:400]}"
        )
        for recipient in ["ceo", "cio"]:
            self.send_message(
                recipient=recipient,
                subject=f"CRO Risk Report — {datetime.utcnow().strftime('%Y-%m-%d')}",
                body=report_text,
                priority=MessagePriority.HIGH if report.halt_triggered else MessagePriority.NORMAL,
            )

        self.remember(narrative, metadata={"type": "risk_report"})
        return report

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        """Handle trade risk-check requests from PMs."""
        if message.subject.startswith("RISK_CHECK:") and message.payload:
            try:
                from hedge_fund.models.schemas import TradeRecommendation
                rec = TradeRecommendation.model_validate(message.payload)
                result = self.check_trade(rec)
                return self.send_message(
                    recipient=message.sender,
                    subject=f"RISK_RESULT:{rec.rec_id}",
                    body=f"{'APPROVED' if result.approved else 'VETOED'}: {result.veto_reason or result.cro_notes}",
                    payload=result.model_dump(mode="json"),
                    thread_id=message.message_id,
                    priority=MessagePriority.HIGH,
                )
            except Exception as e:
                return self.send_message(
                    recipient=message.sender,
                    subject="RISK_CHECK_ERROR",
                    body=str(e),
                    thread_id=message.message_id,
                )
        return None
