"""
Pydantic models for all structured outputs exchanged between agents.
Every inter-agent payload must validate against one of these schemas.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class TradeStatus(str, Enum):
    PENDING_COMPLIANCE = "pending_compliance"
    PENDING_RISK = "pending_risk"
    PENDING_CIO = "pending_cio"        # trades above large-trade threshold
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"

class MessagePriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    FX = "fx"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"


# ── Core Message Envelope ─────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    message_id: str
    sender: str
    recipient: str
    subject: str
    body: str
    payload: Optional[dict[str, Any]] = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    thread_id: Optional[str] = None     # links replies to originating message
    requires_reply: bool = False


# ── Research Outputs ──────────────────────────────────────────────────────────

class ResearchPacket(BaseModel):
    analyst_id: str
    subject_ticker: Optional[str] = None
    subject_description: str
    summary: str
    key_findings: list[str]
    risks: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning_chain: str
    data_sources: list[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_data: Optional[dict[str, Any]] = None


class MacroResearchPacket(ResearchPacket):
    macro_theme: str
    affected_asset_classes: list[AssetClass]
    time_horizon_days: int


class FundamentalResearchPacket(ResearchPacket):
    ticker: str
    sector: str
    intrinsic_value_estimate: Optional[float] = None
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    ebitda_margin: Optional[float] = None
    pe_ratio: Optional[float] = None


class QuantSignalPacket(ResearchPacket):
    strategy_name: str
    signal_value: float          # normalised z-score or probability
    expected_return_bps: float   # basis points
    expected_vol_annualised: float
    sharpe_estimate: float
    backtest_sharpe: Optional[float] = None
    backtest_max_dd: Optional[float] = None


# ── Trade Recommendations ─────────────────────────────────────────────────────

class TradeRecommendation(BaseModel):
    rec_id: str
    pm_id: str
    ticker: str
    asset_class: AssetClass
    direction: Direction
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    target_notional_usd: float
    target_pct_nav: float
    rationale: str
    research_refs: list[str] = Field(default_factory=list)  # ResearchPacket IDs
    confidence_score: float = Field(ge=0.0, le=1.0)
    expected_return_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    time_horizon_days: int
    status: TradeStatus = TradeStatus.PENDING_COMPLIANCE
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Risk Reports ──────────────────────────────────────────────────────────────

class RiskCheckResult(BaseModel):
    rec_id: str
    approved: bool
    veto_reason: Optional[str] = None
    portfolio_var_95: float           # $-VaR at 95% confidence
    position_size_ok: bool
    sector_concentration_ok: bool
    drawdown_within_limit: bool
    correlation_flag: bool            # True = high correlation to existing book
    cro_notes: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PortfolioRiskSummary(BaseModel):
    as_of: datetime
    nav: float
    gross_exposure: float
    net_exposure: float
    var_95_1day: float
    current_drawdown_pct: float
    largest_position_pct: float
    top_sector_concentration_pct: float
    halt_triggered: bool
    sector_breakdown: dict[str, float]
    pod_pnl: dict[str, float]


# ── P&L and Portfolio State ───────────────────────────────────────────────────

class Position(BaseModel):
    ticker: str
    asset_class: AssetClass
    direction: Direction
    quantity: float
    avg_entry_price: float
    current_price: float
    notional_usd: float
    unrealised_pnl: float
    pct_nav: float
    sector: Optional[str] = None
    pod: Optional[str] = None
    open_date: datetime = Field(default_factory=datetime.utcnow)


class BlotterEntry(BaseModel):
    trade_id: str
    rec_id: str
    ticker: str
    direction: Direction
    quantity: float
    fill_price: float
    notional_usd: float
    pod: str
    pm_id: str
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    commission: float = 0.0
    status: str = "simulated_fill"


class DailyPnLReport(BaseModel):
    report_date: datetime
    nav: float
    daily_pnl: float
    daily_return_pct: float
    mtd_pnl: float
    ytd_pnl: float
    gross_exposure: float
    net_exposure: float
    cash_balance: float
    fee_accrual: float
    pod_breakdown: dict[str, float]


# ── CIO Directives and Memos ──────────────────────────────────────────────────

class CIODirective(BaseModel):
    directive_id: str
    from_ceo: bool = True
    text: str
    priority: MessagePriority = MessagePriority.NORMAL
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    requires_response_by: Optional[datetime] = None


class CIOMemo(BaseModel):
    memo_id: str
    subject: str
    market_view: str
    active_positions_summary: str
    top_trade_ideas: list[str]
    key_risks: list[str]
    pod_allocations: dict[str, float]
    prepared_at: datetime = Field(default_factory=datetime.utcnow)


# ── Compliance ────────────────────────────────────────────────────────────────

class ComplianceCheckResult(BaseModel):
    rec_id: str
    approved: bool
    violations: list[str] = Field(default_factory=list)
    wash_sale_flag: bool = False
    restricted_list_flag: bool = False
    position_limit_flag: bool = False
    compliance_notes: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── COO Ops ───────────────────────────────────────────────────────────────────

class AgentHealthStatus(BaseModel):
    agent_id: str
    is_healthy: bool
    task_queue_depth: int
    avg_response_time_s: float
    error_rate_pct: float
    last_heartbeat: datetime
    notes: Optional[str] = None


class OpsReport(BaseModel):
    generated_at: datetime
    agent_statuses: list[AgentHealthStatus]
    bottlenecks: list[str]
    rerouted_tasks: list[str]
    system_health: str   # "green" | "yellow" | "red"
