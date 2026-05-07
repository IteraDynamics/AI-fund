"""
AI Hedge Fund — Main Entry Point

Usage:
  # Run the Streamlit dashboard (primary CEO interface)
  streamlit run hedge_fund/dashboard/app.py

  # Run the daily cycle once (CLI)
  python -m hedge_fund.main --cycle

  # Start the scheduler (runs daily cycle at configured time + continuous monitoring)
  python -m hedge_fund.main --scheduler

  # Issue a directive from CLI
  python -m hedge_fund.main --directive "Analyze semiconductor sector for longs"

  # Interactive CLI mode
  python -m hedge_fund.main --cli
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from hedge_fund.config import LOG_LEVEL
from hedge_fund.messaging.bus import init_bus
from hedge_fund.memory.shared_state import init_portfolio_db

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_orchestrator():
    from hedge_fund.scheduler import HedgeFundOrchestrator
    return HedgeFundOrchestrator()


def cmd_cycle(orch) -> None:
    """Run the full daily cycle and print results."""
    print("\n=== Running Daily Cycle ===\n")
    results = orch.run_daily_cycle()
    print(f"\nCycle complete in {results.get('duration_seconds', 0):.1f}s\n")
    print("Stage results:")
    for stage, result in results.get("stages", {}).items():
        if isinstance(result, dict):
            print(f"  {stage}: {json.dumps(result, indent=4, default=str)}")
        else:
            print(f"  {stage}: {result}")
    memo = results.get("morning_memo")
    if memo:
        print(f"\n=== CIO MORNING MEMO ===")
        print(f"Market View: {memo.market_view}\n")
        print("Top Ideas:")
        for i, idea in enumerate(memo.top_trade_ideas, 1):
            print(f"  {i}. {idea}")
        print("\nKey Risks:")
        for risk in memo.key_risks:
            print(f"  • {risk}")


def cmd_directive(orch, directive: str, priority: str = "normal") -> None:
    """Issue a directive and flush messages."""
    print(f"\n=== Issuing Directive ===")
    print(f"Directive: {directive}\n")
    result = orch.ceo_directive(directive, priority=priority)
    print(f"CIO Response: {result}\n")
    print("Flushing message bus...")
    n = orch.flush_messages(rounds=3)
    print(f"Processed {n} messages through the org.\n")


def cmd_cli(orch) -> None:
    """Interactive CEO CLI."""
    print("\n" + "=" * 60)
    print("  AI HEDGE FUND — CEO TERMINAL")
    print("=" * 60)
    print("Commands:")
    print("  directive <text>  — Issue a directive to the CIO")
    print("  cycle             — Run the daily cycle")
    print("  memo              — Get CIO morning memo")
    print("  risk              — Get CRO risk report")
    print("  pnl               — Get CFO P&L report")
    print("  portfolio         — Show current positions")
    print("  audit             — Show recent audit log")
    print("  flush             — Flush the message bus")
    print("  quit              — Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            cmd = input("CEO> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=1)
        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if action in ("quit", "exit", "q"):
            break

        elif action == "directive" and args:
            cmd_directive(orch, args)

        elif action == "cycle":
            cmd_cycle(orch)

        elif action == "memo":
            memo = orch.get_morning_memo()
            print(f"\n{memo.subject}")
            print(f"Market View: {memo.market_view}")
            print(f"Top Ideas: {memo.top_trade_ideas}")
            print(f"Key Risks: {memo.key_risks}\n")

        elif action == "risk":
            report = orch.get_risk_report()
            print(f"\nRisk Report:")
            print(f"  NAV: ${report.nav:,.2f}")
            print(f"  Drawdown: {report.current_drawdown_pct:.2%}")
            print(f"  VaR 95% (1d): ${report.var_95_1day:,.2f}")
            print(f"  Halt: {'YES' if report.halt_triggered else 'No'}\n")

        elif action == "pnl":
            report = orch.get_pnl_report()
            print(f"\nP&L Report:")
            print(f"  NAV: ${report.nav:,.2f}")
            print(f"  Daily P&L: ${report.daily_pnl:,.2f} ({report.daily_return_pct:.2%})")
            print(f"  Cash: ${report.cash_balance:,.2f}\n")

        elif action == "portfolio":
            from hedge_fund.memory.shared_state import get_positions
            positions = get_positions()
            if positions:
                print(f"\nOpen Positions ({len(positions)}):")
                for p in positions:
                    print(
                        f"  {p.ticker} {p.direction.value.upper()} "
                        f"${p.notional_usd:,.0f} ({p.pct_nav:.1%} NAV) "
                        f"P&L: ${p.unrealised_pnl:,.0f}"
                    )
                print()
            else:
                print("No open positions.\n")

        elif action == "audit":
            from hedge_fund.messaging.bus import get_audit_log
            log = get_audit_log(limit=10)
            print(f"\nRecent audit log ({len(log)} records):")
            for entry in log:
                print(
                    f"  [{entry['timestamp'][:16]}] "
                    f"{entry['sender']} → {entry['recipient']}: "
                    f"{entry['subject'][:50]}"
                )
            print()

        elif action == "flush":
            n = orch.flush_messages(rounds=3)
            print(f"Processed {n} messages.\n")

        else:
            print(f"Unknown command: {action}. Type 'quit' to exit.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hedge Fund")
    parser.add_argument("--cycle", action="store_true", help="Run daily cycle once")
    parser.add_argument("--scheduler", action="store_true", help="Start scheduler")
    parser.add_argument("--directive", type=str, help="Issue a CEO directive")
    parser.add_argument("--priority", type=str, default="normal", help="Directive priority")
    parser.add_argument("--cli", action="store_true", help="Interactive CLI mode")
    args = parser.parse_args()

    orch = get_orchestrator()

    if args.cycle:
        cmd_cycle(orch)
    elif args.directive:
        cmd_directive(orch, args.directive, args.priority)
    elif args.scheduler:
        print("Starting scheduler...")
        orch.start_scheduler()
    elif args.cli:
        cmd_cli(orch)
    else:
        cmd_cli(orch)


if __name__ == "__main__":
    main()
