"""
Daily Cycle Orchestrator

Manages the daily investment cycle in the correct sequence:
  1. Market Monitor — overnight news + pre-market
  2. MacroAnalyst — overnight data prints
  3. PM morning briefings (each PM tasks their analysts)
  4. CIO synthesizes → daily market view
  5. PMs submit trade ideas
  6. CRO morning risk report
  7. CFO P&L and NAV update
  8. CIO sends morning memo to CEO

Also provides a utility to run the full pipeline on-demand
(outside the scheduled daily cycle) for testing or ad-hoc CEO directives.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

from hedge_fund.agents.csuite.cio import CIOAgent
from hedge_fund.agents.csuite.cro import CROAgent
from hedge_fund.agents.csuite.cfo import CFOAgent
from hedge_fund.agents.csuite.coo import COOAgent
from hedge_fund.agents.pms.pm_longshort import PMLongShortAgent
from hedge_fund.agents.pms.pm_macro import PMMacroAgent
from hedge_fund.agents.pms.pm_quant import PMQuantAgent
from hedge_fund.agents.pms.pm_eventdriven import PMEventDrivenAgent
from hedge_fund.agents.analysts.fundamental_analyst import FundamentalAnalystAgent
from hedge_fund.agents.analysts.quant_analyst import QuantAnalystAgent
from hedge_fund.agents.analysts.macro_analyst import MacroAnalystAgent
from hedge_fund.agents.analysts.data_engineer import DataEngineerAgent
from hedge_fund.agents.analysts.compliance_agent import ComplianceAgent
from hedge_fund.agents.analysts.market_monitor import MarketMonitorAgent
from hedge_fund.messaging.bus import init_bus
from hedge_fund.memory.shared_state import init_portfolio_db
from hedge_fund.models.schemas import CIOMemo, MessagePriority

logger = logging.getLogger(__name__)


class HedgeFundOrchestrator:
    """
    Top-level orchestrator that holds all agent instances and runs cycles.
    The CEO interface (main.py / dashboard) interacts through this class.
    """

    def __init__(self) -> None:
        logger.info("Initializing AI Hedge Fund...")
        # Initialize databases
        init_bus()
        init_portfolio_db()

        # Instantiate all agents
        self.cio = CIOAgent()
        self.cro = CROAgent()
        self.cfo = CFOAgent()
        self.coo = COOAgent()

        self.pm_longshort = PMLongShortAgent()
        self.pm_macro = PMMacroAgent()
        self.pm_quant = PMQuantAgent()
        self.pm_eventdriven = PMEventDrivenAgent()

        self.fundamental_analyst_1 = FundamentalAnalystAgent(analyst_number=1)
        self.fundamental_analyst_2 = FundamentalAnalystAgent(analyst_number=2)
        self.quant_analyst = QuantAnalystAgent()
        self.macro_analyst = MacroAnalystAgent()
        self.data_engineer = DataEngineerAgent()
        self.compliance_agent = ComplianceAgent()
        self.market_monitor = MarketMonitorAgent()

        # Agent registry for routing
        self._agents = {
            "cio": self.cio,
            "cro": self.cro,
            "cfo": self.cfo,
            "coo": self.coo,
            "pm_longshort": self.pm_longshort,
            "pm_macro": self.pm_macro,
            "pm_quant": self.pm_quant,
            "pm_eventdriven": self.pm_eventdriven,
            "fundamental_analyst_1": self.fundamental_analyst_1,
            "fundamental_analyst_2": self.fundamental_analyst_2,
            "quant_analyst": self.quant_analyst,
            "macro_analyst": self.macro_analyst,
            "data_engineer": self.data_engineer,
            "compliance_agent": self.compliance_agent,
            "market_monitor": self.market_monitor,
        }
        logger.info(f"All {len(self._agents)} agents initialized.")

    # ── CEO Interface ─────────────────────────────────────────────────────────

    def ceo_directive(self, directive_text: str, priority: str = "normal") -> str:
        """
        CEO issues a directive. Routes to CIO and returns acknowledgment.
        This is the primary CEO→fund entry point.
        """
        logger.info(f"CEO DIRECTIVE: {directive_text[:80]}")
        prio = MessagePriority(priority)
        return self.cio.receive_ceo_directive(directive_text, priority=prio)

    def get_morning_memo(self) -> CIOMemo:
        """Request the CIO's morning memo."""
        return self.cio.produce_morning_memo()

    def get_risk_report(self):
        """Request the CRO's risk report."""
        return self.cro.produce_risk_report()

    def get_pnl_report(self):
        """Request the CFO's P&L report (marks to market)."""
        return self.cfo.produce_daily_report()

    def get_ops_report(self):
        """Request the COO's operational status."""
        return self.coo.produce_ops_report()

    def generate_trades_now(self, mandate: str = "") -> dict:
        """
        Force all PMs to immediately synthesize trade recommendations from
        memory and LLM knowledge — bypassing the analyst research round-trip.

        Returns a summary dict: {pm_id: trade_ticker or "none" or "error"}.
        """
        summary: dict[str, str] = {}
        pms = [
            ("pm_longshort", self.pm_longshort),
            ("pm_macro", self.pm_macro),
            ("pm_quant", self.pm_quant),
            ("pm_eventdriven", self.pm_eventdriven),
        ]
        for pm_id, pm in pms:
            try:
                rec = pm.synthesize_now(mandate=mandate or None)
                summary[pm_id] = f"{rec.direction.value.upper()} {rec.ticker}" if rec else "none"
                logger.info(f"generate_trades_now: {pm_id} → {summary[pm_id]}")
            except Exception as e:
                summary[pm_id] = f"error: {e}"
                logger.error(f"generate_trades_now: {pm_id} error: {e}")
        return summary

    # ── Message Bus Flush ─────────────────────────────────────────────────────

    def flush_messages(self, rounds: int = 3) -> int:
        """
        Process all pending messages across all agents for `rounds` iterations.
        Each round lets messages propagate one hop through the system.
        Returns total messages processed.
        """
        total = 0
        for round_num in range(rounds):
            round_total = 0
            for agent_id, agent in self._agents.items():
                try:
                    msgs = agent.process_inbox()
                    round_total += len(msgs)
                except Exception as e:
                    logger.error(f"Error processing {agent_id} inbox: {e}")
            total += round_total
            logger.debug(f"Message flush round {round_num+1}: {round_total} messages processed")
            if round_total == 0:
                break  # No more messages to process
        return total

    def drain_messages(
        self,
        max_rounds: int = 10,
        callback=None,
    ) -> dict:
        """
        Flush the message bus until all queues are empty or max_rounds is hit.
        Safer than flush_messages() for interactive use — keeps going until done.

        callback: optional callable(round_num, round_detail) for live progress.
        Returns: {"rounds": N, "total_processed": N, "round_detail": [...]}
        """
        from hedge_fund.messaging.bus import total_pending_messages, all_queue_depths
        rounds_detail = []
        total = 0

        for round_num in range(1, max_rounds + 1):
            pending_before = total_pending_messages(db_path=self.cio.db_path)
            if pending_before == 0:
                break

            round_activity: dict[str, int] = {}
            for agent_id, agent in self._agents.items():
                try:
                    msgs = agent.process_inbox()
                    if msgs:
                        round_activity[agent_id] = len(msgs)
                except Exception as e:
                    logger.error(f"drain: {agent_id} error: {e}")

            round_total = sum(round_activity.values())
            total += round_total
            detail = {
                "round": round_num,
                "responses_sent": round_total,
                "active_agents": round_activity,
                "pending_after": total_pending_messages(db_path=self.cio.db_path),
            }
            rounds_detail.append(detail)
            logger.info(
                f"Drain round {round_num}: {round_total} responses from "
                f"{list(round_activity.keys())}"
            )

            if callback:
                try:
                    callback(round_num, detail)
                except Exception:
                    pass

            if round_total == 0:
                break

        return {
            "rounds": len(rounds_detail),
            "total_processed": total,
            "round_detail": rounds_detail,
            "queues_empty": total_pending_messages(db_path=self.cio.db_path) == 0,
        }

    # ── Daily Cycle ───────────────────────────────────────────────────────────

    def run_daily_cycle(self) -> dict:
        """
        Execute the full daily investment cycle.
        Returns a summary dict with reports from each stage.
        """
        cycle_start = datetime.utcnow()
        results: dict = {"cycle_start": cycle_start.isoformat(), "stages": {}}
        logger.info(f"=== DAILY CYCLE START: {cycle_start.strftime('%Y-%m-%d %H:%M')} ===")

        # ── Stage 1: Market Monitor ───────────────────────────────────────────
        logger.info("Stage 1: Market Monitor scan...")
        try:
            alerts = self.market_monitor.run_monitoring_cycle()
            results["stages"]["market_monitor"] = f"{len(alerts)} alerts generated"
        except Exception as e:
            results["stages"]["market_monitor"] = f"ERROR: {e}"
            logger.error(f"Market monitor error: {e}")

        # ── Stage 2: Macro overnight brief ───────────────────────────────────
        logger.info("Stage 2: MacroAnalyst overnight brief...")
        try:
            macro_brief = self.macro_analyst.get_overnight_brief()
            # Forward to CIO and PMs
            self.macro_analyst.send_message(
                recipient="cio",
                subject="OVERNIGHT MACRO BRIEF",
                body=macro_brief,
                priority=MessagePriority.HIGH,
            )
            self.macro_analyst.send_message(
                recipient="pm_macro",
                subject="OVERNIGHT MACRO BRIEF",
                body=macro_brief,
                priority=MessagePriority.HIGH,
            )
            results["stages"]["macro_brief"] = macro_brief[:200]
        except Exception as e:
            results["stages"]["macro_brief"] = f"ERROR: {e}"
            logger.error(f"Macro brief error: {e}")

        # ── Stage 3: PM Morning Briefings ─────────────────────────────────────
        logger.info("Stage 3: PM morning briefings...")
        for pm_id, pm_agent in [
            ("pm_longshort", self.pm_longshort),
            ("pm_macro", self.pm_macro),
            ("pm_quant", self.pm_quant),
            ("pm_eventdriven", self.pm_eventdriven),
        ]:
            try:
                briefing = pm_agent.run_task(
                    "Morning briefing: review current positions, assess overnight moves, "
                    "identify today's key opportunities and risks for your pod."
                )
                results["stages"][f"{pm_id}_brief"] = briefing[:150]
            except Exception as e:
                results["stages"][f"{pm_id}_brief"] = f"ERROR: {e}"
                logger.error(f"{pm_id} morning brief error: {e}")

        # ── Stage 4: Flush messages (analysts respond to PM tasks) ────────────
        logger.info("Stage 4: Processing analyst responses...")
        flushed = self.flush_messages(rounds=2)
        results["stages"]["message_flush"] = f"{flushed} messages processed"

        # ── Stage 5: CRO Risk Report ──────────────────────────────────────────
        logger.info("Stage 5: CRO risk report...")
        try:
            risk_report = self.cro.produce_risk_report()
            results["stages"]["risk_report"] = {
                "nav": risk_report.nav,
                "drawdown": risk_report.current_drawdown_pct,
                "halt": risk_report.halt_triggered,
                "var_95": risk_report.var_95_1day,
            }
        except Exception as e:
            results["stages"]["risk_report"] = f"ERROR: {e}"
            logger.error(f"Risk report error: {e}")

        # ── Stage 6: CFO P&L Report ───────────────────────────────────────────
        logger.info("Stage 6: CFO P&L report...")
        try:
            pnl_report = self.cfo.produce_daily_report()
            results["stages"]["pnl_report"] = {
                "nav": pnl_report.nav,
                "daily_pnl": pnl_report.daily_pnl,
                "daily_return": pnl_report.daily_return_pct,
                "cash": pnl_report.cash_balance,
            }
        except Exception as e:
            results["stages"]["pnl_report"] = f"ERROR: {e}"
            logger.error(f"P&L report error: {e}")

        # ── Stage 7: CIO Morning Memo ─────────────────────────────────────────
        logger.info("Stage 7: CIO morning memo...")
        try:
            memo = self.cio.produce_morning_memo()
            results["stages"]["cio_memo"] = {
                "memo_id": memo.memo_id,
                "market_view": memo.market_view[:200],
                "top_ideas": memo.top_trade_ideas,
                "key_risks": memo.key_risks,
            }
            results["morning_memo"] = memo
        except Exception as e:
            results["stages"]["cio_memo"] = f"ERROR: {e}"
            logger.error(f"CIO memo error: {e}")

        # ── Stage 8: COO Ops Check ────────────────────────────────────────────
        logger.info("Stage 8: COO ops check...")
        try:
            ops = self.coo.produce_ops_report()
            results["stages"]["ops"] = {
                "health": ops.system_health,
                "bottlenecks": ops.bottlenecks,
            }
        except Exception as e:
            results["stages"]["ops"] = f"ERROR: {e}"

        cycle_end = datetime.utcnow()
        duration = (cycle_end - cycle_start).total_seconds()
        results["cycle_end"] = cycle_end.isoformat()
        results["duration_seconds"] = duration
        logger.info(f"=== DAILY CYCLE COMPLETE in {duration:.1f}s ===")
        return results

    # ── Scheduled Entry Point ─────────────────────────────────────────────────

    def start_scheduler(self) -> None:
        """
        Start the blocking daily scheduler.
        Runs the daily cycle at DAILY_CYCLE_HOUR:DAILY_CYCLE_MINUTE every day.
        Handles SIGTERM/SIGINT for clean shutdown after the current job finishes.
        """
        import signal
        import schedule
        from hedge_fund.config import DAILY_CYCLE_HOUR, DAILY_CYCLE_MINUTE

        _running = True

        def _handle_shutdown(signum, frame):
            nonlocal _running
            logger.info(
                f"Shutdown signal {signum} received — "
                "finishing current job then stopping cleanly."
            )
            _running = False

        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)

        cycle_time = f"{DAILY_CYCLE_HOUR:02d}:{DAILY_CYCLE_MINUTE:02d}"
        logger.info(f"Scheduler started. Daily cycle at {cycle_time} UTC.")

        schedule.every().day.at(cycle_time).do(self.run_daily_cycle)
        # Also run market monitor every 15 minutes
        schedule.every(15).minutes.do(self.market_monitor.run_monitoring_cycle)
        # Process message bus every 5 minutes
        schedule.every(5).minutes.do(lambda: self.flush_messages(rounds=1))

        while _running:
            schedule.run_pending()
            time.sleep(30)

        logger.info("Scheduler stopped cleanly.")
