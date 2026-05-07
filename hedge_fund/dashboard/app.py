"""
CEO Dashboard — Streamlit Interface

Pages:
  Activity Monitor  — live feed of what every agent is doing right now
  Morning Memo      — CIO's daily market view and trade ideas
  Issue Directive   — send a directive to the CIO and watch it propagate
  Portfolio         — open positions, blotter, sector breakdown
  Risk              — CRO risk metrics and hard-limit status
  Trade Approvals   — approve or veto pending large trades
  Audit Log         — full immutable message history
  Run Cycle         — manual cycle control and individual reports

Run with: streamlit run hedge_fund/dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure hedge_fund is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from hedge_fund.messaging.bus import (
    init_bus, get_audit_log, get_activity_feed,
    all_queue_depths, total_pending_messages,
)
from hedge_fund.memory.shared_state import (
    init_portfolio_db, get_positions, get_nav, get_cash,
    compute_risk_metrics, get_blotter, get_recommendations,
    is_halted,
)
from hedge_fund.models.schemas import MessagePriority, TradeStatus


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Hedge Fund — CEO Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Event type styling ────────────────────────────────────────────────────────
EVENT_ICON = {
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
EVENT_COLOR = {
    "thinking":              "#d0e8ff",
    "responded":             "#d4edda",
    "sent":                  "#fff3cd",
    "received":              "#e2e3e5",
    "tool_call":             "#f0d9ff",
    "task_start":            "#cfe2ff",
    "error":                 "#f8d7da",
    "trade_created":         "#d1f0d1",
    "trade_executed":        "#a8e6a8",
    "compliance_submitted":  "#fff3cd",
    "compliance_approved":   "#d4edda",
    "compliance_rejected":   "#f8d7da",
    "risk_submitted":        "#fff3cd",
    "risk_approved":         "#d4edda",
    "risk_vetoed":           "#f8d7da",
    "synthesizing":          "#e8d5f0",
    "research_received":     "#d0e8ff",
    "cio_approval_requested":"#fce8b2",
    "trade_skipped":         "#e2e3e5",
}


# ── Init Databases (once per session) ─────────────────────────────────────────
@st.cache_resource
def get_orchestrator():
    init_bus()
    init_portfolio_db()
    from hedge_fund.scheduler import HedgeFundOrchestrator
    return HedgeFundOrchestrator()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_usd(v: float) -> str:
    return f"${v:,.2f}"

def fmt_pct(v: float) -> str:
    return f"{v:.2%}"

def ts_short(ts_str: str) -> str:
    """Format ISO timestamp to HH:MM:SS."""
    try:
        return datetime.fromisoformat(ts_str).strftime("%H:%M:%S")
    except Exception:
        return ts_str[-8:] if len(ts_str) >= 8 else ts_str


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Hedge Fund")
    st.caption("CEO Dashboard")
    st.divider()

    try:
        nav = get_nav()
        cash = get_cash()
        metrics = compute_risk_metrics()
        halted = is_halted()
        pending = total_pending_messages()

        st.metric("NAV", fmt_usd(nav))
        st.metric("Cash", fmt_usd(cash))
        st.metric("Drawdown", fmt_pct(metrics["current_drawdown_pct"]))
        st.metric("Pending messages", pending,
                  help="Unread messages waiting in agent inboxes. Run 'Flush & Drain' to process.")
        if halted:
            st.error("⚠️ TRADING HALT ACTIVE")
    except Exception as e:
        st.warning(f"DB not ready: {e}")

    st.divider()
    page = st.radio(
        "Navigate",
        [
            "Activity Monitor",
            "Morning Memo",
            "Issue Directive",
            "Portfolio",
            "Risk",
            "Trade Approvals",
            "Audit Log",
            "Run Cycle",
        ],
    )


# ── Main Content ──────────────────────────────────────────────────────────────

orch = get_orchestrator()


# ════════════════════════════════════════════════════════════════════════════════
if page == "Activity Monitor":
    st.header("Activity Monitor")
    st.caption(
        "Live feed of what every agent is doing. Each row is one logged event: "
        "thinking (LLM call started), responded (LLM call finished), "
        "sent/received (messages), tool_call (external tool use), error."
    )

    # ── Controls row ─────────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 2])
    with col_a:
        agent_filter = st.selectbox(
            "Agent",
            ["All agents"] + sorted(orch._agents.keys()),
        )
    with col_b:
        event_filter = st.multiselect(
            "Event types",
            options=["thinking", "responded", "sent", "received", "tool_call", "task_start", "error"],
            default=[],
            placeholder="All types",
        )
    with col_c:
        limit = st.number_input("Max rows", min_value=20, max_value=500, value=80)
    with col_d:
        auto_refresh = st.toggle("Auto-refresh every 5s", value=False)

    st.divider()

    # ── Queue depth snapshot ──────────────────────────────────────────────────
    st.markdown("**Current inbox queue depths** (messages waiting to be processed)")
    try:
        depths = all_queue_depths()
        if depths:
            depth_df = pd.DataFrame(
                [{"Agent": k, "Pending messages": v} for k, v in sorted(depths.items())]
            )
            st.dataframe(depth_df, use_container_width=True, hide_index=True)
        else:
            st.success("All agent inboxes are empty — no pending messages.")
    except Exception as e:
        st.error(f"Could not load queue depths: {e}")

    # ── Flush control ─────────────────────────────────────────────────────────
    st.markdown("**Process pending messages**")
    f1, f2 = st.columns(2)
    with f1:
        if st.button("⚡ Flush & Drain (process until empty)", type="primary"):
            progress_placeholder = st.empty()
            log_lines = []

            def show_progress(round_num, detail):
                active = detail.get("active_agents", {})
                if active:
                    agents_str = ", ".join(
                        f"{a}({n})" for a, n in active.items()
                    )
                    log_lines.append(
                        f"Round {round_num}: {detail['responses_sent']} responses "
                        f"from [{agents_str}] — {detail['pending_after']} still pending"
                    )
                    progress_placeholder.text("\n".join(log_lines[-8:]))

            with st.spinner("Draining message bus..."):
                result = orch.drain_messages(max_rounds=10, callback=show_progress)

            if result["queues_empty"]:
                st.success(
                    f"Done — {result['total_processed']} responses across "
                    f"{result['rounds']} rounds. All queues empty."
                )
            else:
                st.warning(
                    f"Stopped after {result['rounds']} rounds. "
                    f"{total_pending_messages()} messages still pending — run again."
                )
            st.rerun()

    with f2:
        if st.button("Single flush (1 round)"):
            n = orch.flush_messages(rounds=1)
            st.info(f"1-round flush: {n} responses sent.")
            st.rerun()

    st.divider()

    # ── Activity feed ─────────────────────────────────────────────────────────
    st.markdown("**Agent activity log** (most recent first)")
    try:
        feed = get_activity_feed(
            limit=int(limit),
            agent_id=None if agent_filter == "All agents" else agent_filter,
            event_types=event_filter if event_filter else None,
        )
        if feed:
            rows = []
            for entry in feed:
                et = entry.get("event_type", "")
                icon = EVENT_ICON.get(et, "•")
                rows.append({
                    "Time": ts_short(entry.get("timestamp", "")),
                    "Agent": entry.get("agent_id", ""),
                    "Event": f"{icon} {et}",
                    "Details": entry.get("details", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time": st.column_config.TextColumn(width="small"),
                    "Agent": st.column_config.TextColumn(width="medium"),
                    "Event": st.column_config.TextColumn(width="small"),
                    "Details": st.column_config.TextColumn(width="large"),
                },
            )
            st.caption(f"Showing {len(feed)} events.")
        else:
            st.info(
                "No activity yet. Issue a directive on the 'Issue Directive' page, "
                "then click 'Flush & Drain' above to watch agents process it."
            )
    except Exception as e:
        st.error(f"Could not load activity feed: {e}")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(5)
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Morning Memo":
    st.header("CIO Morning Memo")

    if st.button("Refresh Memo", type="primary"):
        with st.spinner("CIO generating morning memo..."):
            try:
                memo = orch.get_morning_memo()
                st.session_state["memo"] = memo
            except Exception as e:
                st.error(f"Error: {e}")

    memo = st.session_state.get("memo")
    if memo:
        st.subheader(memo.subject)
        st.caption(f"Prepared: {memo.prepared_at.strftime('%Y-%m-%d %H:%M UTC')}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Market View")
            st.write(memo.market_view)

            st.markdown("#### Top Trade Ideas")
            for i, idea in enumerate(memo.top_trade_ideas, 1):
                st.markdown(f"{i}. {idea}")

        with col2:
            st.markdown("#### Key Risks")
            for risk in memo.key_risks:
                st.warning(risk)

            st.markdown("#### Pod Allocations")
            alloc_df = pd.DataFrame(
                list(memo.pod_allocations.items()),
                columns=["Pod", "Allocation"],
            )
            alloc_df["Allocation"] = alloc_df["Allocation"].apply(fmt_pct)
            st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        st.markdown("#### Active Positions Summary")
        st.text(memo.active_positions_summary)
    else:
        st.info("Click 'Refresh Memo' to generate today's morning memo from the CIO.")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Issue Directive":
    st.header("Issue Directive to CIO")
    st.caption(
        "Type a directive in natural language. The CIO will decompose it into "
        "PM tasks and dispatch research. Use **Flush & Drain** after sending to "
        "propagate the work through the full org hierarchy."
    )

    directive = st.text_area(
        "Directive",
        placeholder=(
            "e.g. 'Analyze the semiconductor sector for long opportunities given "
            "the AI infrastructure buildout theme. Focus on equipment and EDA software names.'"
        ),
        height=120,
    )
    priority = st.selectbox("Priority", ["normal", "high", "urgent", "low"])

    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("Send to CIO", type="primary", disabled=not directive.strip())

    if submit and directive.strip():
        with st.spinner("CIO processing directive..."):
            try:
                result = orch.ceo_directive(directive.strip(), priority=priority)
                st.success("Directive dispatched to CIO.")
                st.markdown(f"**CIO decomposition:**\n\n{result}")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # Drain the bus — show live progress
        st.markdown("---")
        st.markdown("**Propagating through org hierarchy...**")
        st.caption(
            "Messages are being passed: CIO → PMs → Analysts → PMs → Compliance → CRO. "
            "Each round is one hop. This makes real Claude API calls so it takes a moment per agent."
        )
        progress_box = st.empty()
        log_lines: list[str] = []

        def update_progress(round_num, detail):
            active = detail.get("active_agents", {})
            if active:
                agents_str = ", ".join(
                    f"{a} ({n} msg)" for a, n in active.items()
                )
                log_lines.append(f"  Round {round_num}: {agents_str}")
            else:
                log_lines.append(f"  Round {round_num}: (no new responses)")
            pending = detail.get("pending_after", 0)
            display = "\n".join(log_lines[-6:])
            display += f"\n  → {pending} messages still pending"
            progress_box.code(display)

        with st.spinner("Draining..."):
            drain_result = orch.drain_messages(max_rounds=10, callback=update_progress)

        if drain_result["queues_empty"]:
            st.success(
                f"All done. {drain_result['total_processed']} agent responses across "
                f"{drain_result['rounds']} rounds. All queues are empty."
            )
        else:
            remaining = total_pending_messages()
            st.warning(
                f"Completed {drain_result['rounds']} rounds "
                f"({drain_result['total_processed']} responses). "
                f"{remaining} messages still pending — click 'Flush & Drain' on "
                f"the Activity Monitor page to continue."
            )

        st.info("Check the **Activity Monitor** page to see what every agent did.")

    st.divider()
    st.markdown("#### CEO Inbox (messages addressed to you)")
    try:
        from hedge_fund.messaging.bus import receive
        ceo_msgs = receive("ceo", limit=20, mark_read=False)
        if ceo_msgs:
            for msg in ceo_msgs:
                with st.expander(
                    f"[{msg.priority.value.upper()}] From: {msg.sender} — {msg.subject}",
                    expanded=False,
                ):
                    st.caption(f"Received: {msg.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
                    st.write(msg.body)
        else:
            st.info("No messages for the CEO yet.")
    except Exception as e:
        st.warning(f"Could not load inbox: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio":
    st.header("Portfolio State")

    try:
        metrics = compute_risk_metrics()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("NAV", fmt_usd(metrics["nav"]))
        col2.metric("Gross Exposure", fmt_usd(metrics["gross_exposure"]))
        col3.metric("Net Exposure", fmt_usd(metrics["net_exposure"]))
        col4.metric("VaR 95% (1d)", fmt_usd(metrics["var_95_1day"]))
        col5.metric("Drawdown", fmt_pct(metrics["current_drawdown_pct"]))
    except Exception as e:
        st.error(f"Could not load metrics: {e}")

    st.divider()
    st.subheader("Open Positions")
    try:
        positions = get_positions()
        if positions:
            pos_data = []
            for p in positions:
                pos_data.append({
                    "Ticker": p.ticker,
                    "Direction": p.direction.value.upper(),
                    "Qty": f"{p.quantity:,.0f}",
                    "Avg Entry": fmt_usd(p.avg_entry_price),
                    "Current": fmt_usd(p.current_price),
                    "Notional": fmt_usd(p.notional_usd),
                    "Unreal P&L": fmt_usd(p.unrealised_pnl),
                    "% NAV": fmt_pct(p.pct_nav),
                    "Pod": p.pod or "—",
                    "Sector": p.sector or "—",
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
        else:
            st.info(
                "No open positions. Issue a directive and drain the message bus — "
                "trade recommendations will appear in 'Trade Approvals' once the pipeline runs."
            )
    except Exception as e:
        st.error(f"Could not load positions: {e}")

    st.divider()
    st.subheader("Recent Trades (Blotter)")
    try:
        blotter = get_blotter(limit=20)
        if blotter:
            st.dataframe(pd.DataFrame(blotter), use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed yet.")
    except Exception as e:
        st.error(f"Could not load blotter: {e}")

    st.divider()
    st.subheader("Sector Breakdown")
    try:
        metrics = compute_risk_metrics()
        sector_df = pd.DataFrame(
            [{"Sector": k, "% NAV": fmt_pct(v)}
             for k, v in metrics["sector_breakdown"].items()]
        )
        if not sector_df.empty:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        else:
            st.info("No sector data — no open positions.")
    except Exception as e:
        st.error(f"Sector breakdown error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Risk":
    st.header("Risk Dashboard")

    if st.button("Refresh Risk Report", type="primary"):
        with st.spinner("CRO generating risk report..."):
            try:
                report = orch.get_risk_report()
                st.session_state["risk_report"] = report
            except Exception as e:
                st.error(f"Error: {e}")

    report = st.session_state.get("risk_report")
    if report:
        if report.halt_triggered:
            st.error("⚠️ TRADING HALT ACTIVE — Drawdown limit breached")

        col1, col2, col3 = st.columns(3)
        col1.metric("NAV", fmt_usd(report.nav))
        col2.metric("Current Drawdown", fmt_pct(report.current_drawdown_pct))
        col3.metric("VaR 95% (1d)", fmt_usd(report.var_95_1day))

        col4, col5, col6 = st.columns(3)
        col4.metric("Gross Exposure", fmt_usd(report.gross_exposure))
        col5.metric("Net Exposure", fmt_usd(report.net_exposure))
        col6.metric("Largest Position", fmt_pct(report.largest_position_pct))

        st.divider()
        st.subheader("Pod P&L")
        if report.pod_pnl:
            pod_df = pd.DataFrame(
                [{"Pod": k, "Unrealised P&L": fmt_usd(v)}
                 for k, v in report.pod_pnl.items()]
            )
            st.dataframe(pod_df, use_container_width=True, hide_index=True)

        st.subheader("Sector Exposure")
        if report.sector_breakdown:
            sec_df = pd.DataFrame(
                [{"Sector": k, "% NAV": fmt_pct(v)}
                 for k, v in report.sector_breakdown.items()]
            )
            st.dataframe(sec_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Refresh Risk Report' to generate the CRO's risk assessment.")

    st.divider()
    st.subheader("Hard Limits")
    limits = {
        "Max Position Size": "5% of NAV",
        "Max Sector Concentration": "20% of NAV",
        "Max Portfolio Drawdown": "15% — triggers trading halt",
        "Max Gross Leverage": "200% of NAV",
        "Large Trade Approval": "$50,000 notional → CIO approval required",
    }
    for k, v in limits.items():
        st.write(f"**{k}:** {v}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Trade Approvals":
    st.header("Trade Pipeline")
    st.caption(
        "Full trade pipeline visibility: every recommendation from PM synthesis through "
        "Compliance → CRO → CIO approval → execution. "
        "Large trades (> $50k notional) stop at **PENDING_CIO** for your approval."
    )

    # ── Generate Trades Now ───────────────────────────────────────────────────
    with st.expander("⚡ Force PMs to Generate Trades Now", expanded=False):
        st.caption(
            "Bypass the analyst research round-trip. Each PM immediately synthesizes "
            "a trade from memory and LLM knowledge, then submits through compliance → risk."
        )
        mandate_input = st.text_input(
            "Optional mandate (leave blank for each PM's default strategy)",
            placeholder="e.g. 'Focus on energy sector longs given supply squeeze'",
            key="force_trade_mandate",
        )
        if st.button("Generate Trades Now", type="primary", key="btn_generate_trades"):
            with st.spinner("PMs synthesizing trades..."):
                try:
                    summary = orch.generate_trades_now(mandate=mandate_input.strip())
                    for pm_id, result in summary.items():
                        icon = "✅" if "none" not in result and "error" not in result else (
                            "⚠️" if "none" in result else "❌"
                        )
                        st.write(f"{icon} **{pm_id}**: {result}")
                except Exception as e:
                    st.error(f"Error: {e}")
            with st.spinner("Running compliance + risk checks..."):
                drain_result = orch.drain_messages(max_rounds=8)
            st.success(
                f"Done. {drain_result['total_processed']} messages processed. "
                f"Refresh the page to see updated pipeline status."
            )
            st.rerun()

    st.divider()

    # ── Pipeline Status Bar ───────────────────────────────────────────────────
    _PIPELINE_STAGES = [
        ("pending_compliance", "Compliance", "🔍"),
        ("pending_risk",       "Risk Check", "⚖️"),
        ("pending_cio",        "CIO Approval", "👔"),
        ("approved",           "Approved",   "✅"),
        ("rejected",           "Rejected",   "❌"),
        ("executed",           "Executed",   "💰"),
    ]
    try:
        all_recs = get_recommendations()
        counts = {s: 0 for s, _, _ in _PIPELINE_STAGES}
        for r in all_recs:
            s = r.get("status", "")
            if s in counts:
                counts[s] += 1

        cols = st.columns(len(_PIPELINE_STAGES))
        for col, (status_key, label, icon) in zip(cols, _PIPELINE_STAGES):
            with col:
                st.metric(f"{icon} {label}", counts[status_key])

        st.divider()

        # ── Per-Stage Details ─────────────────────────────────────────────────
        for status_key, label, icon in _PIPELINE_STAGES:
            recs = [r for r in all_recs if r.get("status") == status_key]
            if not recs:
                continue
            st.subheader(f"{icon} {label} ({len(recs)})")
            for rec in recs[:10]:
                expand = (status_key == "pending_cio")
                with st.expander(
                    f"{rec['direction'].upper()} **{rec['ticker']}** — "
                    f"${rec['target_notional_usd']:,.0f} notional | "
                    f"PM: {rec['pm_id']} | "
                    f"Confidence: {rec['confidence_score']:.0%}",
                    expanded=expand,
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Size:** {rec['target_pct_nav']:.1%} of NAV")
                        st.write(f"**Expected return:** {rec['expected_return_pct']:.1%}")
                        st.write(f"**Stop loss:** {rec['stop_loss_pct']:.1%}")
                        st.write(f"**Take profit:** {rec['take_profit_pct']:.1%}")
                        st.write(f"**Horizon:** {rec['time_horizon_days']} days")
                        st.write(f"**Asset class:** {rec.get('asset_class', 'N/A')}")
                    with col2:
                        st.write(f"**Rationale:**")
                        st.write(rec['rationale'][:400])
                        st.caption(f"Submitted: {rec['timestamp']} | ID: {rec['rec_id'][:12]}")

                    if status_key == "pending_cio":
                        st.markdown("---")
                        st.markdown("**Your action required** — this trade exceeded the $50k threshold.")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ Approve", key=f"approve_{rec['rec_id']}"):
                                from hedge_fund.memory.shared_state import update_recommendation_status
                                update_recommendation_status(rec['rec_id'], "approved")
                                st.success("Trade approved — PM will execute on next flush.")
                                st.rerun()
                        with c2:
                            if st.button("❌ Veto", key=f"veto_{rec['rec_id']}"):
                                from hedge_fund.memory.shared_state import update_recommendation_status
                                update_recommendation_status(rec['rec_id'], "rejected")
                                st.error("Trade vetoed.")
                                st.rerun()

        if not all_recs:
            st.info(
                "No trade recommendations in the pipeline yet.\n\n"
                "**To generate trades:**\n"
                "1. Use the **⚡ Force PMs to Generate Trades Now** section above, or\n"
                "2. Issue a directive on the **Issue Directive** page and drain the message bus.\n\n"
                "Trades will appear here as they move through Compliance → Risk → CIO → Execution."
            )

    except Exception as e:
        st.error(f"Error loading trade pipeline: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.header("Agent Communication Audit Log")
    st.caption("Every message sent between agents — immutable, timestamped.")

    col1, col2, col3 = st.columns(3)
    with col1:
        sender_filter = st.text_input("Filter by sender", "")
    with col2:
        recipient_filter = st.text_input("Filter by recipient", "")
    with col3:
        limit = st.number_input("Max rows", min_value=10, max_value=500, value=50)

    try:
        log = get_audit_log(
            limit=int(limit),
            sender=sender_filter.strip() or None,
            recipient=recipient_filter.strip() or None,
        )
        if log:
            df = pd.DataFrame(log)
            if "body" in df.columns:
                df["body"] = df["body"].str[:150]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(log)} records.")
        else:
            st.info("No audit records found.")
    except Exception as e:
        st.error(f"Could not load audit log: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Run Cycle":
    st.header("Run Daily Cycle")
    st.caption(
        "Manually trigger the full 8-stage daily investment cycle, or run individual reports. "
        "The cycle makes many Claude API calls — allow several minutes to complete."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Full Daily Cycle", type="primary"):
            progress = st.empty()
            with st.spinner("Running daily cycle..."):
                try:
                    results = orch.run_daily_cycle()
                    st.success(f"Cycle complete in {results.get('duration_seconds', 0):.1f}s")
                    st.json(results.get("stages", {}))
                    if "morning_memo" in results:
                        memo = results["morning_memo"]
                        st.subheader("Morning Memo")
                        st.write(memo.market_view)
                except Exception as e:
                    st.error(f"Cycle error: {e}")

    with col2:
        if st.button("Flush & Drain"):
            log_lines: list[str] = []
            progress_box = st.empty()

            def cb(round_num, detail):
                active = detail.get("active_agents", {})
                line = f"Round {round_num}: " + (
                    ", ".join(f"{a}({n})" for a, n in active.items())
                    if active else "(idle)"
                )
                log_lines.append(line)
                progress_box.code("\n".join(log_lines[-8:]))

            with st.spinner("Draining..."):
                result = orch.drain_messages(max_rounds=10, callback=cb)
            if result["queues_empty"]:
                st.success(
                    f"{result['total_processed']} responses in {result['rounds']} rounds. Done."
                )
            else:
                st.warning(f"Stopped at {result['rounds']} rounds — run again.")

    st.divider()
    st.subheader("Individual Reports")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("CFO P&L Report"):
            with st.spinner("CFO generating report..."):
                try:
                    report = orch.get_pnl_report()
                    st.metric("NAV", fmt_usd(report.nav))
                    st.metric("Daily P&L", fmt_usd(report.daily_pnl))
                    st.metric("Daily Return", fmt_pct(report.daily_return_pct))
                    st.metric("Cash", fmt_usd(report.cash_balance))
                    st.metric("Fee Accrual", fmt_usd(report.fee_accrual))
                except Exception as e:
                    st.error(str(e))

    with c2:
        if st.button("CRO Risk Report"):
            with st.spinner("CRO generating report..."):
                try:
                    report = orch.get_risk_report()
                    st.metric("Drawdown", fmt_pct(report.current_drawdown_pct))
                    st.metric("VaR 95%", fmt_usd(report.var_95_1day))
                    st.metric("Halt", "YES ⚠️" if report.halt_triggered else "No")
                except Exception as e:
                    st.error(str(e))

    with c3:
        if st.button("COO Ops Report"):
            with st.spinner("COO generating report..."):
                try:
                    report = orch.get_ops_report()
                    st.metric("System Health", report.system_health.upper())
                    st.write("Bottlenecks:", report.bottlenecks or "None")
                except Exception as e:
                    st.error(str(e))
