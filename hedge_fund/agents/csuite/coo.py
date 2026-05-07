"""
Chief Operating Officer (COO) Agent

Responsibilities:
  - Monitor agent task queues and detect bottlenecks
  - Re-route work if an agent is overloaded or producing low-quality output
  - Track agent health metrics
  - Produce daily ops report
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from hedge_fund.agents.base_agent import BaseAgent
from hedge_fund.config import AUDIT_DB_PATH
from hedge_fund.messaging import bus as message_bus
from hedge_fund.models.schemas import (
    AgentMessage, AgentHealthStatus, OpsReport, MessagePriority,
)


# All agent IDs in the system — COO monitors all of them
ALL_AGENT_IDS = [
    "cio", "cro", "cfo",
    "pm_longshort", "pm_macro", "pm_quant", "pm_eventdriven",
    "fundamental_analyst_1", "fundamental_analyst_2",
    "quant_analyst", "macro_analyst",
    "data_engineer", "compliance_agent", "market_monitor",
]


class COOAgent(BaseAgent):
    agent_id = "coo"
    role_name = "Chief Operating Officer"
    system_prompt = """You are the Chief Operating Officer (COO) of an AI hedge fund.

Your mandate:
- Monitor the health and performance of every agent in the fund
- Detect bottlenecks: agents with large task queues or slow response times
- Re-route work when agents are overloaded
- Ensure operational continuity: no task gets stuck or lost
- Report operational status to the CEO daily

Your key metrics:
- Task queue depth per agent (flag if > 10 unread messages)
- Error rate per agent (flag if > 20%)
- Response latency (flag if > 60s average)
- Overall system health: green (<5% error rate) / yellow (5-20%) / red (>20%)

When you detect a bottleneck:
1. Identify the cause
2. Consider: re-assigning tasks to a backup agent, reducing task volume, escalating to CEO
3. Send the affected agent a priority message to clear its queue
4. Log the intervention

Be concise in your reports. Use a traffic-light system for status."""

    def __init__(self, db_path: str = AUDIT_DB_PATH) -> None:
        super().__init__(db_path)
        # Track agent health stats in-memory (would be persisted to DB in production)
        self._agent_stats: dict[str, dict] = {
            aid: {
                "task_count": 0,
                "error_count": 0,
                "total_response_time": 0.0,
                "last_heartbeat": datetime.utcnow().isoformat(),
            }
            for aid in ALL_AGENT_IDS
        }

    # ── Health Monitoring ─────────────────────────────────────────────────────

    def check_all_agents(self) -> list[AgentHealthStatus]:
        """Poll queue depths and compile health statuses for all agents."""
        statuses = []
        for agent_id in ALL_AGENT_IDS:
            q_depth = message_bus.queue_depth(agent_id, db_path=self.db_path)
            stats = self._agent_stats.get(agent_id, {})
            task_count = max(stats.get("task_count", 1), 1)
            error_count = stats.get("error_count", 0)
            avg_rt = stats.get("total_response_time", 0) / task_count

            is_healthy = (
                q_depth < 15
                and error_count / task_count < 0.20
            )
            statuses.append(
                AgentHealthStatus(
                    agent_id=agent_id,
                    is_healthy=is_healthy,
                    task_queue_depth=q_depth,
                    avg_response_time_s=round(avg_rt, 2),
                    error_rate_pct=round(error_count / task_count * 100, 1),
                    last_heartbeat=datetime.fromisoformat(
                        stats.get("last_heartbeat", datetime.utcnow().isoformat())
                    ),
                    notes="Queue backlog" if q_depth >= 15 else None,
                )
            )
        return statuses

    def produce_ops_report(self) -> OpsReport:
        """Generate the daily operations report."""
        statuses = self.check_all_agents()
        bottlenecks = [
            s.agent_id for s in statuses
            if not s.is_healthy or s.task_queue_depth >= 10
        ]
        unhealthy_count = sum(1 for s in statuses if not s.is_healthy)

        if unhealthy_count == 0:
            system_health = "green"
        elif unhealthy_count <= 3:
            system_health = "yellow"
        else:
            system_health = "red"

        # LLM narrative
        prompt = f"""
Ops check — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}

Agent health summary:
{json.dumps([s.model_dump(mode='json') for s in statuses[:6]], indent=2, default=str)}

Bottlenecks detected: {bottlenecks}
System health: {system_health}

Write a 2-sentence operational status update for the CEO.
Mention any bottlenecks and recommended actions.
"""
        narrative = self._call_llm(prompt)

        report = OpsReport(
            generated_at=datetime.utcnow(),
            agent_statuses=statuses,
            bottlenecks=bottlenecks,
            rerouted_tasks=[],
            system_health=system_health,
        )

        # Send to CEO if system is not green
        if system_health != "green":
            self.send_message(
                recipient="ceo",
                subject=f"COO Ops Alert [{system_health.upper()}] — {datetime.utcnow().strftime('%Y-%m-%d')}",
                body=narrative + f"\n\nBottlenecks: {bottlenecks}",
                priority=MessagePriority.HIGH if system_health == "red" else MessagePriority.NORMAL,
            )

        self.remember(narrative, metadata={"type": "ops_report"})
        return report

    def nudge_agent(self, agent_id: str, reason: str) -> str:
        """Send a priority nudge to an agent that's falling behind."""
        return self.send_message(
            recipient=agent_id,
            subject="COO: Process your queue",
            body=f"You have a backlog. Priority: clear your inbox now. Reason: {reason}",
            priority=MessagePriority.URGENT,
        )

    def update_agent_stat(
        self, agent_id: str, response_time_s: float, error: bool = False
    ) -> None:
        """Record a task completion for an agent (called by the orchestrator)."""
        if agent_id not in self._agent_stats:
            self._agent_stats[agent_id] = {
                "task_count": 0, "error_count": 0,
                "total_response_time": 0.0, "last_heartbeat": datetime.utcnow().isoformat(),
            }
        s = self._agent_stats[agent_id]
        s["task_count"] += 1
        s["total_response_time"] += response_time_s
        if error:
            s["error_count"] += 1
        s["last_heartbeat"] = datetime.utcnow().isoformat()

    # ── Message Handler ───────────────────────────────────────────────────────

    def handle_message(self, message: AgentMessage) -> Optional[str]:
        if "ops report" in message.body.lower() or message.subject == "OPS_REPORT":
            report = self.produce_ops_report()
            return self.send_message(
                recipient=message.sender,
                subject="COO Ops Report",
                body=f"System: {report.system_health.upper()} | Bottlenecks: {report.bottlenecks}",
                thread_id=message.message_id,
            )
        # Accept heartbeat / status messages
        if message.subject.startswith("HEARTBEAT:"):
            parts = message.subject.split(":")
            if len(parts) >= 2:
                self.update_agent_stat(parts[1], float(parts[2]) if len(parts) > 2 else 0.0)
        return None
