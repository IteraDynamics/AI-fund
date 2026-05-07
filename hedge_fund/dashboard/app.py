"""
CEO Dashboard — Streamlit Interface

Provides the Founder/CEO with:
  - Directive input to the CIO
  - Morning memo display
  - Live portfolio state and P&L
  - Risk metrics and alerts
  - Trade approval/veto workflow
  - Full audit log viewer

Run with: streamlit run hedge_fund/dashboard/app.py
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Ensure hedge_fund is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd

from hedge_fund.messaging.bus import init_bus, get_audit_log
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


# ── Init Databases ────────────────────────────────────────────────────────────
@st.cache_resource
def get_orchestrator():
    """Load the orchestrator once per session."""
    init_bus()
    init_portfolio_db()
    from hedge_fund.scheduler import HedgeFundOrchestrator
    return HedgeFundOrchestrator()


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_usd(v: float) -> str:
    return f"${v:,.2f}"

def fmt_pct(v: float) -> str:
    return f"{v:.2%}"

def color_pnl(v: float) -> str:
    return "green" if v >= 0 else "red"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AI Hedge Fund")
    st.caption("CEO Dashboard")
    st.divider()

    # Nav metrics in sidebar
    try:
        nav = get_nav()
        cash = get_cash()
        metrics = compute_risk_metrics()
        halted = is_halted()

        st.metric("NAV", fmt_usd(nav))
        st.metric("Cash", fmt_usd(cash))
        st.metric("Drawdown", fmt_pct(metrics["current_drawdown_pct"]))
        if halted:
            st.error("TRADING HALT ACTIVE")
    except Exception as e:
        st.warning(f"DB not ready: {e}")

    st.divider()
    page = st.radio(
        "Navigate",
        [
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
if page == "Morning Memo":
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
    st.caption("Your directive will be decomposed and dispatched to the relevant Portfolio Managers.")

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
                st.success("Directive dispatched.")
                st.markdown(f"**CIO Response:**\n\n{result}")

                # Auto-flush messages
                with st.spinner("Propagating messages through org..."):
                    flushed = orch.flush_messages(rounds=2)
                st.info(f"{flushed} messages processed through the org.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("#### CEO Inbox (messages addressed to you)")
    try:
        from hedge_fund.messaging.bus import receive
        ceo_msgs = receive("ceo", limit=20)
        if ceo_msgs:
            for msg in ceo_msgs:
                with st.expander(
                    f"[{msg.priority.value.upper()}] From: {msg.sender} — {msg.subject}",
                    expanded=False,
                ):
                    st.caption(f"Received: {msg.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
                    st.write(msg.body)
        else:
            st.info("No new messages.")
    except Exception as e:
        st.warning(f"Could not load inbox: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio":
    st.header("Portfolio State")

    # Metrics row
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
            df = pd.DataFrame(pos_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Could not load positions: {e}")

    st.divider()
    st.subheader("Recent Trades (Blotter)")
    try:
        blotter = get_blotter(limit=20)
        if blotter:
            st.dataframe(pd.DataFrame(blotter), use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")
    except Exception as e:
        st.error(f"Could not load blotter: {e}")

    st.divider()
    st.subheader("Sector Breakdown")
    try:
        metrics = compute_risk_metrics()
        sector_df = pd.DataFrame(
            [{"Sector": k, "% NAV": fmt_pct(v)} for k, v in metrics["sector_breakdown"].items()]
        )
        if not sector_df.empty:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)
        else:
            st.info("No sector data.")
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
        halted = report.halt_triggered
        if halted:
            st.error("⚠️ TRADING HALT ACTIVE — Drawdown limit breached")

        col1, col2, col3 = st.columns(3)
        col1.metric("NAV", fmt_usd(report.nav))
        col2.metric("Current Drawdown", fmt_pct(report.current_drawdown_pct),
                    delta_color="inverse")
        col3.metric("VaR 95% (1d)", fmt_usd(report.var_95_1day))

        col4, col5, col6 = st.columns(3)
        col4.metric("Gross Exposure", fmt_usd(report.gross_exposure))
        col5.metric("Net Exposure", fmt_usd(report.net_exposure))
        col6.metric("Largest Position", fmt_pct(report.largest_position_pct))

        st.divider()
        st.subheader("Pod P&L")
        if report.pod_pnl:
            pod_df = pd.DataFrame(
                [{"Pod": k, "Unrealised P&L": fmt_usd(v)} for k, v in report.pod_pnl.items()]
            )
            st.dataframe(pod_df, use_container_width=True, hide_index=True)

        st.subheader("Sector Exposure")
        if report.sector_breakdown:
            sec_df = pd.DataFrame(
                [{"Sector": k, "% NAV": fmt_pct(v)} for k, v in report.sector_breakdown.items()]
            )
            st.dataframe(sec_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Refresh Risk Report' to generate the CRO's risk assessment.")

    st.divider()
    st.subheader("Hard Limits")
    limits = {
        "Max Position Size": "5% of NAV",
        "Max Sector Concentration": "20% of NAV",
        "Max Portfolio Drawdown": "15% (triggers halt)",
        "Large Trade Approval": "$50,000 notional → CIO approval required",
    }
    for k, v in limits.items():
        st.write(f"**{k}:** {v}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Trade Approvals":
    st.header("Trade Approval Queue")
    st.caption("Review and approve/veto trade recommendations pending CEO action.")

    try:
        # Show all recent recommendations grouped by status
        for status in ["pending_cio", "approved", "rejected", "executed"]:
            recs = get_recommendations(status=status)
            if recs:
                st.subheader(f"{status.replace('_', ' ').upper()} ({len(recs)})")
                for rec in recs[:10]:
                    with st.expander(
                        f"{rec['direction'].upper()} {rec['ticker']} — "
                        f"${rec['target_notional_usd']:,.0f} | "
                        f"Confidence: {rec['confidence_score']:.2f}",
                        expanded=(status == "pending_cio"),
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**PM:** {rec['pm_id']}")
                            st.write(f"**Size:** {rec['target_pct_nav']:.1%} of NAV")
                            st.write(f"**Expected Return:** {rec['expected_return_pct']:.1%}")
                            st.write(f"**Stop Loss:** {rec['stop_loss_pct']:.1%}")
                            st.write(f"**Take Profit:** {rec['take_profit_pct']:.1%}")
                            st.write(f"**Horizon:** {rec['time_horizon_days']} days")
                        with col2:
                            st.write(f"**Rationale:** {rec['rationale'][:300]}")
                            st.write(f"**Status:** {rec['status']}")
                            st.write(f"**Submitted:** {rec['timestamp']}")

                        if status == "pending_cio":
                            c1, c2 = st.columns(2)
                            with c1:
                                if st.button(f"✅ Approve", key=f"approve_{rec['rec_id']}"):
                                    from hedge_fund.memory.shared_state import update_recommendation_status
                                    update_recommendation_status(rec['rec_id'], "approved")
                                    st.success("Trade approved.")
                                    st.rerun()
                            with c2:
                                if st.button(f"❌ Veto", key=f"veto_{rec['rec_id']}"):
                                    from hedge_fund.memory.shared_state import update_recommendation_status
                                    update_recommendation_status(rec['rec_id'], "rejected")
                                    st.error("Trade vetoed.")
                                    st.rerun()

        if not any(
            get_recommendations(status=s)
            for s in ["pending_cio", "approved", "rejected", "executed"]
        ):
            st.info("No trade recommendations in the system yet.")

    except Exception as e:
        st.error(f"Error loading recommendations: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.header("Agent Communication Audit Log")

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
            # Truncate long body for display
            if "body" in df.columns:
                df["body"] = df["body"].str[:150]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(log)} audit records.")
        else:
            st.info("No audit records found.")
    except Exception as e:
        st.error(f"Could not load audit log: {e}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Run Cycle":
    st.header("Run Daily Cycle")
    st.caption("Manually trigger the full daily investment cycle.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Full Daily Cycle", type="primary"):
            with st.spinner("Running daily cycle... (this may take a few minutes)"):
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
        if st.button("Flush Message Bus"):
            with st.spinner("Flushing messages..."):
                try:
                    n = orch.flush_messages(rounds=3)
                    st.success(f"Flushed {n} messages.")
                except Exception as e:
                    st.error(f"Flush error: {e}")

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
                except Exception as e:
                    st.error(str(e))

    with c2:
        if st.button("CRO Risk Report"):
            with st.spinner("CRO generating report..."):
                try:
                    report = orch.get_risk_report()
                    st.metric("Drawdown", fmt_pct(report.current_drawdown_pct))
                    st.metric("VaR 95%", fmt_usd(report.var_95_1day))
                    st.metric("Halt", "YES" if report.halt_triggered else "No")
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
