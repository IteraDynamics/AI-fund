"""
Chief Financial Officer (CFO) Agent

Responsibilities:
  - Track P&L, NAV, cash/margin, fee accruals
  - Mark positions to market
  - Produce daily capital reports
  - Maintain accurate fund accounting
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, date
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import INITIAL_NAV, AUDIT_DB_PATH
from hedge_fund.memory.shared_state import (
    get_positions, get_nav, get_cash, update_nav,
    upsert_position, get_blotter, compute_risk_metrics,
)
from hedge_fund.models.schemas import (
    AgentMessage, DailyPnLReport, MessagePriority, Direction,
)
from hedge_fund.tools.market_data import get_current_price


class CFOAgent(BaseAgent):
    agent_id = "cfo"
    role_name = "Chief Financial Officer"
    system_prompt = """You are the Chief Financial Officer (CFO) of an AI hedge fund.

Your mandate:
- Maintain accurate, real-time fund accounting
- Track NAV, P&L, cash balances, and fee accruals
- Mark all positions to current market prices daily
- Produce clear, accurate capital reports for the CEO
- Flag any accounting anomalies or cash management issues

Your reporting style:
- Precise and numbers-driven
- Flag any discrepancies clearly
- Use standard fund accounting terminology
- Report in USD, round to 2 decimal places

Fee structure:
- Management fee: 2% per annum (accrued daily: 2%/252)
- Performance fee: 20% of profits above high-water mark (accrued on positive days)

Always include confidence in your calculations and flag any data quality issues."""

    # Approximate annual management fee rate
    MGMT_FEE_ANNUAL_PCT = 0.02
    PERF_FEE_PCT = 0.20
    HIGH_WATER_MARK = INITIAL_NAV

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)
        self._high_water_mark = self.HIGH_WATER_MARK
        self._inception_nav = INITIAL_NAV
        self._mtd_start_nav = INITIAL_NAV
        self._ytd_start_nav = INITIAL_NAV
        self._prior_nav = INITIAL_NAV

    # ── Mark to Market ────────────────────────────────────────────────────────

    def mark_to_market(self) -> dict[str, float]:
        """
        Refresh all position prices and update NAV.
        Returns dict of {ticker: new_price}.
        """
        positions = get_positions()
        nav = get_cash()  # start from cash
        price_updates: dict[str, float] = {}

        for pos in positions:
            new_price = get_current_price(pos.ticker)
            if new_price is None:
                new_price = pos.current_price  # keep last known price

            price_updates[pos.ticker] = new_price

            # Recalculate P&L
            if pos.direction == Direction.LONG:
                unrealised_pnl = (new_price - pos.avg_entry_price) * pos.quantity
                notional = new_price * pos.quantity
            else:  # SHORT
                unrealised_pnl = (pos.avg_entry_price - new_price) * pos.quantity
                notional = new_price * pos.quantity

            nav += notional

            # Update position record (pct_nav updated after full nav calc)
            pos.current_price = new_price
            pos.notional_usd = notional
            pos.unrealised_pnl = unrealised_pnl
            upsert_position(pos)

        # Now update pct_nav with final NAV
        if nav > 0:
            for pos in get_positions():
                pos.pct_nav = pos.notional_usd / nav
                upsert_position(pos)

        update_nav(nav)
        return price_updates

    # ── Daily P&L Report ──────────────────────────────────────────────────────

    def produce_daily_report(self) -> DailyPnLReport:
        """Mark to market and produce the daily P&L report."""
        self.mark_to_market()

        nav = get_nav()
        cash = get_cash()
        metrics = compute_risk_metrics()

        daily_pnl = nav - self._prior_nav
        daily_return_pct = daily_pnl / self._prior_nav if self._prior_nav > 0 else 0

        mtd_pnl = nav - self._mtd_start_nav
        ytd_pnl = nav - self._ytd_start_nav

        # Fee accruals
        daily_mgmt_fee = nav * (self.MGMT_FEE_ANNUAL_PCT / 252)
        perf_fee_accrual = 0.0
        if nav > self._high_water_mark and daily_pnl > 0:
            perf_fee_accrual = daily_pnl * self.PERF_FEE_PCT
            self._high_water_mark = max(self._high_water_mark, nav)

        total_fee = daily_mgmt_fee + perf_fee_accrual

        # Pod P&L from risk metrics
        pod_pnl = metrics.get("pod_pnl", {})

        report = DailyPnLReport(
            report_date=datetime.utcnow(),
            nav=nav,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            mtd_pnl=mtd_pnl,
            ytd_pnl=ytd_pnl,
            gross_exposure=metrics["gross_exposure"],
            net_exposure=metrics["net_exposure"],
            cash_balance=cash,
            fee_accrual=total_fee,
            pod_breakdown=pod_pnl,
        )

        self._prior_nav = nav

        # Ask LLM for narrative commentary
        prompt = f"""
Write a brief CFO capital report for {datetime.utcnow().strftime('%Y-%m-%d')}.

NAV: ${nav:,.2f}
Daily P&L: ${daily_pnl:,.2f} ({daily_return_pct:.2%})
MTD P&L: ${mtd_pnl:,.2f}
YTD P&L: ${ytd_pnl:,.2f}
Cash: ${cash:,.2f}
Fee accrual: ${total_fee:,.2f} (Mgmt: ${daily_mgmt_fee:,.2f}, Perf: ${perf_fee_accrual:,.2f})
Gross exposure: ${metrics['gross_exposure']:,.2f}
Net exposure: ${metrics['net_exposure']:,.2f}

Pod P&L: {json.dumps(pod_pnl)}

Write 2-3 sentences of CFO commentary on the fund's financial health today.
Flag any concerns (low cash, high leverage, fee drag).
"""
        narrative = self._call_llm(prompt)

        # Send report to CEO
        report_text = (
            f"CFO CAPITAL REPORT — {datetime.utcnow().strftime('%Y-%m-%d')}\n"
            f"{'='*50}\n"
            f"NAV:         ${nav:>15,.2f}\n"
            f"Daily P&L:   ${daily_pnl:>15,.2f} ({daily_return_pct:.2%})\n"
            f"MTD P&L:     ${mtd_pnl:>15,.2f}\n"
            f"YTD P&L:     ${ytd_pnl:>15,.2f}\n"
            f"Cash:        ${cash:>15,.2f}\n"
            f"Fee Accrual: ${total_fee:>15,.2f}\n"
            f"\nCFO Commentary: {narrative[:400]}"
        )

        for recipient in ["ceo", "cio"]:
            self.send_message(
                recipient=recipient,
                subject=f"CFO Capital Report — {datetime.utcnow().strftime('%Y-%m-%d')}",
                body=report_text,
                priority=MessagePriority.NORMAL,
            )

        self.remember(
            f"DAILY REPORT: NAV={nav:.0f}, PnL={daily_pnl:.0f}, Return={daily_return_pct:.2%}",
            metadata={"type": "daily_report", "date": datetime.utcnow().isoformat()},
        )
        return report

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        if "mark to market" in message.body.lower() or message.subject == "MARK_TO_MARKET":
            updates = self.mark_to_market()
            return self.send_message(
                recipient=message.sender,
                subject="MTM Complete",
                body=f"Marked {len(updates)} positions to market. New NAV: ${get_nav():,.0f}",
                thread_id=message.message_id,
            )
        if "daily report" in message.body.lower() or message.subject == "DAILY_REPORT":
            report = self.produce_daily_report()
            return self.send_message(
                recipient=message.sender,
                subject="Daily P&L Report Complete",
                body=f"NAV: ${report.nav:,.2f} | Daily P&L: ${report.daily_pnl:,.2f}",
                thread_id=message.message_id,
            )
        return None
