"""
Market Monitor Agent

Runs on a loop — watches for price alerts, news catalysts,
earnings releases, macro data prints. Acts as the real-time
alerting system for the fund.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.models.schemas import AgentMessage, MessagePriority
from hedge_fund.tools.market_data import (
    get_current_price, get_macro_indicators,
    get_sector_etf_performance, get_earnings_calendar,
)
from hedge_fund.tools.web_search import search_news


# Watchlist of tickers/instruments to monitor
DEFAULT_WATCHLIST = [
    "^GSPC", "^VIX", "^TNX", "GLD", "CL=F",   # macro
    "AAPL", "MSFT", "NVDA", "AMZN", "META",    # mega caps
    "XLK", "XLF", "XLE", "XLY",               # sector ETFs
]

# Alert thresholds
PRICE_ALERT_THRESHOLD = 0.03      # 3% daily move
VIX_ALERT_LEVEL = 25.0            # VIX above 25 → risk-off alert
YIELD_ALERT_BPS = 10              # 10bps move in 10Y → alert


class MarketMonitorAgent(BaseAgent):
    agent_id = "market_monitor"
    role_name = "Market Monitor"

    system_prompt = """You are the Market Monitor at an AI hedge fund.

Your mandate:
- Monitor markets continuously for significant price moves, news catalysts, and macro events
- Alert the CIO and relevant PMs when significant events occur
- Track earnings releases, macro data prints, and geopolitical events
- Maintain a watchlist and alert when thresholds are breached
- Provide real-time situational awareness to the investment team

Alert priorities:
- URGENT: Trading halt signals, VIX > 35, >5% moves in key indices, crisis news
- HIGH: VIX > 25, >3% moves in watched positions, earnings beats/misses
- NORMAL: Earnings calendar updates, sector rotation signals, macro data in line
- LOW: General market color, sector performance summaries

When issuing alerts:
- Be specific: ticker, % move, context
- Suggest which PM/analyst should be notified
- Assess if it requires immediate action or is informational
- Include confidence in your assessment

You run continuously — process your inbox frequently."""

    allowed_tools = [
        {"type": "web_search_20250305", "name": "web_search"},
    ]

    def __init__(
        self,
        watchlist: Optional[list[str]] = None,
        db_path: str = AUDIT_DB_PATH,
    ) -> None:
        super().__init__(db_path)
        self.watchlist = watchlist or DEFAULT_WATCHLIST
        self._prior_prices: dict[str, float] = {}
        self._alerts_sent: set[str] = set()  # deduplicate alerts

    # ── Monitoring Cycle ──────────────────────────────────────────────────────

    def run_monitoring_cycle(self) -> list[str]:
        """
        One monitoring cycle: check prices, news, earnings.
        Returns list of alert message_ids sent.
        """
        alerts = []

        # 1. Price alerts
        price_alerts = self._check_price_alerts()
        for alert in price_alerts:
            msg_id = self._send_alert(alert["text"], alert["priority"], alert["recipients"])
            if msg_id:
                alerts.append(msg_id)

        # 2. Macro alerts (VIX, yields)
        macro_alerts = self._check_macro_alerts()
        for alert in macro_alerts:
            msg_id = self._send_alert(alert["text"], alert["priority"], alert["recipients"])
            if msg_id:
                alerts.append(msg_id)

        # 3. News scan
        news_alert = self._scan_news()
        if news_alert:
            msg_id = self._send_alert(news_alert, MessagePriority.NORMAL, ["cio"])
            if msg_id:
                alerts.append(msg_id)

        return alerts

    def _check_price_alerts(self) -> list[dict]:
        """Check for significant price moves in watchlist."""
        alerts = []
        for ticker in self.watchlist:
            current = get_current_price(ticker)
            if current is None:
                continue

            prior = self._prior_prices.get(ticker)
            if prior and prior > 0:
                move = (current - prior) / prior
                if abs(move) >= PRICE_ALERT_THRESHOLD:
                    direction = "up" if move > 0 else "down"
                    key = f"{ticker}_{datetime.utcnow().date()}"
                    if key not in self._alerts_sent:
                        self._alerts_sent.add(key)
                        alerts.append({
                            "text": (
                                f"PRICE ALERT: {ticker} {direction} {abs(move):.1%} "
                                f"to ${current:.2f}"
                            ),
                            "priority": (
                                MessagePriority.URGENT if abs(move) >= 0.05
                                else MessagePriority.HIGH
                            ),
                            "recipients": ["cio", "cro"],
                        })
            self._prior_prices[ticker] = current

        return alerts

    def _check_macro_alerts(self) -> list[dict]:
        """Check VIX and yield levels for macro alerts."""
        alerts = []
        try:
            macro = get_macro_indicators()

            vix = macro.get("VIX")
            if vix and vix > VIX_ALERT_LEVEL:
                key = f"VIX_{datetime.utcnow().date()}"
                if key not in self._alerts_sent:
                    self._alerts_sent.add(key)
                    priority = MessagePriority.URGENT if vix > 35 else MessagePriority.HIGH
                    alerts.append({
                        "text": f"MACRO ALERT: VIX at {vix:.1f} — elevated fear reading",
                        "priority": priority,
                        "recipients": ["cio", "cro", "pm_macro"],
                    })

            tnx = macro.get("10Y_TREASURY")
            prior_tnx = self._prior_prices.get("10Y_TREASURY")
            if tnx and prior_tnx:
                bps_move = abs(tnx - prior_tnx) * 100
                if bps_move >= YIELD_ALERT_BPS:
                    key = f"TNX_{datetime.utcnow().date()}"
                    if key not in self._alerts_sent:
                        self._alerts_sent.add(key)
                        alerts.append({
                            "text": f"RATES ALERT: 10Y Treasury moved {bps_move:.0f}bps to {tnx:.2f}%",
                            "priority": MessagePriority.HIGH,
                            "recipients": ["cio", "pm_macro"],
                        })
            if tnx:
                self._prior_prices["10Y_TREASURY"] = tnx

        except Exception:
            pass
        return alerts

    def _scan_news(self) -> Optional[str]:
        """Quick news scan for market-moving headlines."""
        news = search_news("market breaking news catalyst today stocks rates")
        if not news or len(news) < 50:
            return None

        prompt = f"""
Review these market headlines and identify if there is anything requiring
immediate PM/CIO attention:

{news[:600]}

If there is a significant market-moving event, summarize it in 1-2 sentences.
If nothing notable, respond with "NOTHING_NOTABLE".
"""
        result = self._call_llm(prompt)
        if "NOTHING_NOTABLE" in result.upper():
            return None
        return f"NEWS SCAN: {result[:300]}"

    def check_earnings_calendar(self, tickers: list[str]) -> str:
        """Check upcoming earnings for a list of tickers."""
        results = []
        for ticker in tickers[:10]:
            try:
                cal = get_earnings_calendar(ticker)
                if cal.get("Earnings Date"):
                    results.append(f"{ticker}: earnings {cal['Earnings Date']}")
            except Exception:
                pass
        return "\n".join(results) if results else "No upcoming earnings data found."

    def _send_alert(
        self,
        text: str,
        priority: MessagePriority,
        recipients: list[str],
    ) -> Optional[str]:
        """Send an alert to multiple recipients. Returns first message_id."""
        if not text:
            return None
        first_id = None
        for recipient in recipients:
            msg_id = self.send_message(
                recipient=recipient,
                subject=f"MARKET ALERT: {text[:50]}",
                body=text,
                priority=priority,
            )
            if first_id is None:
                first_id = msg_id
        self.remember(text, metadata={"type": "alert", "priority": priority.value})
        return first_id

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        if "monitor" in message.subject.lower() or message.subject == "RUN_MONITOR":
            alerts = self.run_monitoring_cycle()
            return self.send_message(
                recipient=message.sender,
                subject="MONITOR CYCLE COMPLETE",
                body=f"Monitoring cycle complete. {len(alerts)} alerts sent.",
                thread_id=message.message_id,
            )

        if "earnings" in message.body.lower():
            import re
            tickers = re.findall(r'\b[A-Z]{1,5}\b', message.body)
            exclude = {"PM", "CEO", "CIO", "CRO", "CFO", "COO", "SEC"}
            tickers = [t for t in tickers if t not in exclude][:10]
            result = self.check_earnings_calendar(tickers or self.watchlist[:5])
            return self.send_message(
                recipient=message.sender,
                subject="EARNINGS CALENDAR",
                body=result,
                thread_id=message.message_id,
            )

        # Default: run a monitoring cycle
        alerts = self.run_monitoring_cycle()
        return self.send_message(
            recipient=message.sender,
            subject="MONITOR UPDATE",
            body=f"Monitor cycle: {len(alerts)} alerts generated.",
            thread_id=message.message_id,
        )
