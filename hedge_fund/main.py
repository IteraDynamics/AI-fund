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


_ACTIVITY_ICONS = {
    "thinking":              "🧠",
    "responded":             "✅",
    "sent":                  "📤",
    "received":              "📥",
    "tool_call":             "🔧",
    "task_start":            "🎯",
    "error":                 "❌",
    "trade_created":         "💡",
    "trade_executed":        "💰",
    "compliance_submitted":  "📋",
    "compliance_approved":   "✅",
    "compliance_rejected":   "🚫",
    "risk_submitted":        "⚖️",
    "risk_approved":         "✅",
    "risk_vetoed":           "🛑",
    "synthesizing":          "🔬",
    "research_received":     "📊",
    "cio_approval_requested":"👔",
    "trade_skipped":         "⏭️",
}


def cmd_status(orch) -> None:
    """Show a snapshot of what every agent is currently up to."""
    from hedge_fund.messaging.bus import all_queue_depths, get_activity_feed, total_pending_messages
    depths = all_queue_depths()
    total = total_pending_messages()
    feed = get_activity_feed(limit=20)

    print(f"\n{'='*60}")
    print("  AGENT STATUS SNAPSHOT")
    print(f"{'='*60}")
    print(f"  Total pending messages: {total}")
    if depths:
        print("\n  Inbox queue depths (agents with pending messages):")
        for agent, depth in sorted(depths.items()):
            bar = "█" * min(depth, 20)
            print(f"    {agent:<30} {depth:>3} {bar}")
    else:
        print("\n  All inboxes empty — no pending messages.")

    print(f"\n  Last 20 activity events:")
    print(f"  {'Time':<10} {'Agent':<28} {'Event':<12} Details")
    print(f"  {'-'*9} {'-'*27} {'-'*11} {'-'*30}")
    for entry in feed:
        icon = _ACTIVITY_ICONS.get(entry.get("event_type", ""), "•")
        ts = entry.get("timestamp", "")[-8:]  # HH:MM:SS
        agent = entry.get("agent_id", "")[:27]
        et = (icon + " " + entry.get("event_type", ""))[:11]
        details = entry.get("details", "")[:60]
        print(f"  {ts:<10} {agent:<28} {et:<12} {details}")
    print()


def cmd_watch(orch, max_rounds: int = 10) -> None:
    """
    Drain the message bus while printing live activity.
    Keeps flushing until all queues are empty (or max_rounds hit).
    """
    from hedge_fund.messaging.bus import total_pending_messages

    pending = total_pending_messages()
    print(f"\n=== WATCH MODE === ({pending} messages pending)")
    print("Processing agents... (Ctrl+C to stop)\n")

    last_seen_activity_id = 0

    def show_new_activity():
        nonlocal last_seen_activity_id
        from hedge_fund.messaging.bus import get_activity_feed
        # Get activities we haven't shown yet
        feed = get_activity_feed(limit=30)
        new_entries = [e for e in reversed(feed) if e["id"] > last_seen_activity_id]
        for entry in new_entries:
            icon = _ACTIVITY_ICONS.get(entry.get("event_type", ""), "•")
            ts = entry.get("timestamp", "")[-8:]
            agent = entry.get("agent_id", "")
            details = entry.get("details", "")[:80]
            print(f"  {ts} [{agent}] {icon} {details}")
        if new_entries:
            last_seen_activity_id = new_entries[-1]["id"]

    def round_callback(round_num, detail):
        show_new_activity()
        pending_after = detail.get("pending_after", 0)
        active = detail.get("active_agents", {})
        if active:
            print(f"\n  ── Round {round_num} complete: "
                  f"{detail['responses_sent']} responses | "
                  f"{pending_after} still pending ──\n")

    try:
        result = orch.drain_messages(max_rounds=max_rounds, callback=round_callback)
        show_new_activity()
    except KeyboardInterrupt:
        print("\nWatch interrupted.")
        return

    print(f"\n{'='*50}")
    if result["queues_empty"]:
        print(f"  Done. {result['total_processed']} total responses, "
              f"{result['rounds']} rounds. All queues empty.")
    else:
        remaining = total_pending_messages()
        print(f"  Stopped after {result['rounds']} rounds. "
              f"{remaining} messages still pending. Run 'watch' again to continue.")
    print()


def cmd_trade(orch, mandate: str = "") -> None:
    """
    Force all PMs to synthesize trade recommendations immediately
    (bypasses analyst research round-trip — uses memory + LLM knowledge).
    Then drains the message bus so compliance/risk checks execute.
    """
    print(f"\n=== GENERATE TRADES NOW ===")
    if mandate:
        print(f"Mandate: {mandate}\n")
    else:
        print("No mandate provided — each PM will use its default strategy.\n")

    summary = orch.generate_trades_now(mandate=mandate)
    print("PM synthesis results:")
    for pm_id, result in summary.items():
        print(f"  {pm_id:<25} → {result}")

    print("\nDraining message bus (compliance + risk checks)...")
    result = orch.drain_messages(max_rounds=8)
    print(f"Done: {result['total_processed']} messages processed in {result['rounds']} rounds.")
    if result["queues_empty"]:
        print("All queues empty — trades have been submitted through the pipeline.\n")
    else:
        print("Some messages still pending. Run 'watch' to continue draining.\n")

    # Print any new trade recommendations
    from hedge_fund.memory.shared_state import get_recommendations
    recs = get_recommendations(status_filter=None)
    if recs:
        print(f"Trade recommendations in system ({len(recs)} total):")
        for r in recs[-10:]:  # last 10
            print(
                f"  [{r.status.value:<22}] {r.direction.value.upper()} "
                f"{r.ticker} ${r.notional_usd:,.0f}  (id={r.recommendation_id[:8]})"
            )
        print()
    else:
        print("No trade recommendations in system yet.\n")


def cmd_activity(orch, agent_id: str = "", limit: int = 30) -> None:
    """Print recent activity log entries, optionally filtered by agent."""
    from hedge_fund.messaging.bus import get_activity_feed
    feed = get_activity_feed(
        limit=limit,
        agent_id=agent_id or None,
    )
    label = f"for [{agent_id}]" if agent_id else "(all agents)"
    print(f"\nActivity log {label} — {len(feed)} events (newest first):")
    print(f"{'Time':<10} {'Agent':<28} {'Event':<12} Details")
    print(f"{'-'*9} {'-'*27} {'-'*11} {'-'*30}")
    for entry in feed:
        icon = _ACTIVITY_ICONS.get(entry.get("event_type", ""), "•")
        ts = entry.get("timestamp", "")[-8:]
        agent = entry.get("agent_id", "")[:27]
        et = (icon + " " + entry.get("event_type", ""))[:11]
        details = entry.get("details", "")[:70]
        print(f"{ts:<10} {agent:<28} {et:<12} {details}")
    print()


def cmd_cli(orch) -> None:
    """Interactive CEO CLI."""
    print("\n" + "=" * 60)
    print("  AI HEDGE FUND — CEO TERMINAL")
    print("=" * 60)
    print("Commands:")
    print("  directive <text>      — Issue a directive to the CIO")
    print("  watch                 — Drain message bus + show live activity")
    print("  status                — Show agent queue depths + recent activity")
    print("  activity [agent_id]   — Print activity log (all or one agent)")
    print("  cycle                 — Run the full daily cycle")
    print("  memo                  — Get CIO morning memo")
    print("  risk                  — Get CRO risk report")
    print("  pnl                   — Get CFO P&L report")
    print("  portfolio             — Show current positions")
    print("  audit                 — Show recent message audit log")
    print("  trade [mandate]       — Force all PMs to generate trades NOW (bypass analysts)")
    print("  flush                 — Quick flush (3 rounds, no output)")
    print("  quit                  — Exit")
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

        elif action == "directive":
            if args:
                cmd_directive(orch, args)
                print("Tip: run 'watch' to see agents process the directive.\n")
            else:
                print("Usage: directive <your directive text>\n")

        elif action == "watch":
            cmd_watch(orch)

        elif action == "status":
            cmd_status(orch)

        elif action == "activity":
            cmd_activity(orch, agent_id=args.strip())

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
            print(f"\nRecent message audit log ({len(log)} records):")
            for entry in log:
                print(
                    f"  [{entry['timestamp'][:16]}] "
                    f"{entry['sender']} → {entry['recipient']}: "
                    f"{entry['subject'][:50]}"
                )
            print()

        elif action == "trade":
            cmd_trade(orch, mandate=args.strip())

        elif action == "flush":
            n = orch.flush_messages(rounds=3)
            print(f"Flushed {n} messages. Run 'status' to see queue depths.\n")

        else:
            print(f"Unknown command: '{action}'. Type 'quit' to exit.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hedge Fund")
    parser.add_argument("--cycle", action="store_true", help="Run daily cycle once")
    parser.add_argument("--scheduler", action="store_true", help="Start scheduler")
    parser.add_argument("--directive", type=str, help="Issue a CEO directive")
    parser.add_argument("--priority", type=str, default="normal", help="Directive priority")
    parser.add_argument("--trade", type=str, nargs="?", const="", metavar="MANDATE",
                        help="Force PMs to generate trades now (optional mandate text)")
    parser.add_argument("--cli", action="store_true", help="Interactive CLI mode")
    args = parser.parse_args()

    orch = get_orchestrator()

    if args.cycle:
        cmd_cycle(orch)
    elif args.directive:
        cmd_directive(orch, args.directive, args.priority)
    elif args.trade is not None:
        cmd_trade(orch, mandate=args.trade)
    elif args.scheduler:
        print("Starting scheduler...")
        orch.start_scheduler()
    elif args.cli:
        cmd_cli(orch)
    else:
        cmd_cli(orch)


if __name__ == "__main__":
    main()
