"""
Central SQLite database for portfolio state: positions, blotter, P&L, risk metrics.
This is the single source of truth for all financial state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, date
from typing import Iterator, Optional

from hedge_fund.config import PORTFOLIO_DB_PATH, INITIAL_NAV
from hedge_fund.models.schemas import (
    Position, BlotterEntry, DailyPnLReport,
    Direction, AssetClass,
)


_lock = threading.Lock()


@contextmanager
def _conn(db_path: str = PORTFOLIO_DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_portfolio_db(db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _conn(db_path) as c:
        c.executescript(f"""
            CREATE TABLE IF NOT EXISTS fund_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO fund_state (key, value)
            VALUES ('nav', '{INITIAL_NAV}'),
                   ('cash', '{INITIAL_NAV}'),
                   ('halt', 'false'),
                   ('peak_nav', '{INITIAL_NAV}');

            CREATE TABLE IF NOT EXISTS positions (
                ticker          TEXT PRIMARY KEY,
                asset_class     TEXT NOT NULL,
                direction       TEXT NOT NULL,
                quantity        REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                current_price   REAL NOT NULL,
                notional_usd    REAL NOT NULL,
                unrealised_pnl  REAL NOT NULL DEFAULT 0,
                pct_nav         REAL NOT NULL DEFAULT 0,
                sector          TEXT,
                pod             TEXT,
                open_date       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS blotter (
                trade_id        TEXT PRIMARY KEY,
                rec_id          TEXT NOT NULL,
                ticker          TEXT NOT NULL,
                direction       TEXT NOT NULL,
                quantity        REAL NOT NULL,
                fill_price      REAL NOT NULL,
                notional_usd    REAL NOT NULL,
                pod             TEXT NOT NULL,
                pm_id           TEXT NOT NULL,
                executed_at     TEXT NOT NULL,
                commission      REAL DEFAULT 0,
                status          TEXT DEFAULT 'simulated_fill'
            );

            CREATE INDEX IF NOT EXISTS idx_blotter_ticker ON blotter (ticker);
            CREATE INDEX IF NOT EXISTS idx_blotter_pod    ON blotter (pod);

            CREATE TABLE IF NOT EXISTS daily_pnl (
                report_date     TEXT PRIMARY KEY,
                nav             REAL NOT NULL,
                daily_pnl       REAL NOT NULL,
                daily_return_pct REAL NOT NULL,
                mtd_pnl         REAL NOT NULL,
                ytd_pnl         REAL NOT NULL,
                gross_exposure  REAL NOT NULL,
                net_exposure    REAL NOT NULL,
                cash_balance    REAL NOT NULL,
                fee_accrual     REAL NOT NULL,
                pod_breakdown   TEXT NOT NULL  -- JSON
            );

            CREATE TABLE IF NOT EXISTS risk_checks (
                rec_id              TEXT PRIMARY KEY,
                approved            INTEGER NOT NULL,
                veto_reason         TEXT,
                portfolio_var_95    REAL,
                position_size_ok    INTEGER,
                sector_conc_ok      INTEGER,
                drawdown_ok         INTEGER,
                correlation_flag    INTEGER,
                cro_notes           TEXT,
                timestamp           TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_recommendations (
                rec_id              TEXT PRIMARY KEY,
                pm_id               TEXT NOT NULL,
                ticker              TEXT NOT NULL,
                asset_class         TEXT NOT NULL,
                direction           TEXT NOT NULL,
                order_type          TEXT NOT NULL,
                limit_price         REAL,
                target_notional_usd REAL NOT NULL,
                target_pct_nav      REAL NOT NULL,
                rationale           TEXT NOT NULL,
                confidence_score    REAL NOT NULL,
                expected_return_pct REAL NOT NULL,
                stop_loss_pct       REAL NOT NULL,
                take_profit_pct     REAL NOT NULL,
                time_horizon_days   INTEGER NOT NULL,
                status              TEXT NOT NULL,
                timestamp           TEXT NOT NULL
            );
        """)


# ── Fund State ────────────────────────────────────────────────────────────────

def get_nav(db_path: str = PORTFOLIO_DB_PATH) -> float:
    with _conn(db_path) as c:
        row = c.execute("SELECT value FROM fund_state WHERE key='nav'").fetchone()
    return float(row["value"])


def get_cash(db_path: str = PORTFOLIO_DB_PATH) -> float:
    with _conn(db_path) as c:
        row = c.execute("SELECT value FROM fund_state WHERE key='cash'").fetchone()
    return float(row["value"])


def is_halted(db_path: str = PORTFOLIO_DB_PATH) -> bool:
    with _conn(db_path) as c:
        row = c.execute("SELECT value FROM fund_state WHERE key='halt'").fetchone()
    return row["value"] == "true"


def set_halt(halted: bool, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("UPDATE fund_state SET value=? WHERE key='halt'",
                      ("true" if halted else "false",))


def update_cash(new_cash: float, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("UPDATE fund_state SET value=? WHERE key='cash'", (str(new_cash),))


def update_nav(new_nav: float, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("UPDATE fund_state SET value=? WHERE key='nav'", (str(new_nav),))
            # track peak for drawdown calc
            peak = float(c.execute(
                "SELECT value FROM fund_state WHERE key='peak_nav'"
            ).fetchone()["value"])
            if new_nav > peak:
                c.execute("UPDATE fund_state SET value=? WHERE key='peak_nav'",
                          (str(new_nav),))


def get_peak_nav(db_path: str = PORTFOLIO_DB_PATH) -> float:
    with _conn(db_path) as c:
        row = c.execute("SELECT value FROM fund_state WHERE key='peak_nav'").fetchone()
    return float(row["value"])


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions(db_path: str = PORTFOLIO_DB_PATH) -> list[Position]:
    with _conn(db_path) as c:
        rows = c.execute("SELECT * FROM positions").fetchall()
    return [_row_to_position(r) for r in rows]


def get_position(ticker: str, db_path: str = PORTFOLIO_DB_PATH) -> Optional[Position]:
    with _conn(db_path) as c:
        row = c.execute("SELECT * FROM positions WHERE ticker=?", (ticker,)).fetchone()
    return _row_to_position(row) if row else None


def upsert_position(pos: Position, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("""
                INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET
                    direction=excluded.direction,
                    quantity=excluded.quantity,
                    avg_entry_price=excluded.avg_entry_price,
                    current_price=excluded.current_price,
                    notional_usd=excluded.notional_usd,
                    unrealised_pnl=excluded.unrealised_pnl,
                    pct_nav=excluded.pct_nav,
                    sector=excluded.sector,
                    pod=excluded.pod
            """, (
                pos.ticker, pos.asset_class.value, pos.direction.value,
                pos.quantity, pos.avg_entry_price, pos.current_price,
                pos.notional_usd, pos.unrealised_pnl, pos.pct_nav,
                pos.sector, pos.pod, pos.open_date.isoformat(),
            ))


def close_position(ticker: str, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("DELETE FROM positions WHERE ticker=?", (ticker,))


def _row_to_position(row: sqlite3.Row) -> Position:
    return Position(
        ticker=row["ticker"],
        asset_class=AssetClass(row["asset_class"]),
        direction=Direction(row["direction"]),
        quantity=row["quantity"],
        avg_entry_price=row["avg_entry_price"],
        current_price=row["current_price"],
        notional_usd=row["notional_usd"],
        unrealised_pnl=row["unrealised_pnl"],
        pct_nav=row["pct_nav"],
        sector=row["sector"],
        pod=row["pod"],
        open_date=datetime.fromisoformat(row["open_date"]),
    )


# ── Blotter ───────────────────────────────────────────────────────────────────

def record_trade(entry: BlotterEntry, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("""
                INSERT OR REPLACE INTO blotter VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                entry.trade_id, entry.rec_id, entry.ticker,
                entry.direction.value, entry.quantity, entry.fill_price,
                entry.notional_usd, entry.pod, entry.pm_id,
                entry.executed_at.isoformat(), entry.commission, entry.status,
            ))
            # deduct cash
            cash = float(c.execute(
                "SELECT value FROM fund_state WHERE key='cash'"
            ).fetchone()["value"])
            if entry.direction == Direction.LONG:
                cash -= (entry.notional_usd + entry.commission)
            else:
                cash += (entry.notional_usd - entry.commission)
            c.execute("UPDATE fund_state SET value=? WHERE key='cash'", (str(cash),))


def get_blotter(
    ticker: Optional[str] = None,
    pod: Optional[str] = None,
    limit: int = 100,
    db_path: str = PORTFOLIO_DB_PATH,
) -> list[dict]:
    clauses, params = [], []
    if ticker:
        clauses.append("ticker=?"); params.append(ticker)
    if pod:
        clauses.append("pod=?"); params.append(pod)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _conn(db_path) as c:
        rows = c.execute(
            f"SELECT * FROM blotter {where} ORDER BY executed_at DESC LIMIT ?", params
        ).fetchall()
    return [dict(r) for r in rows]


# ── Trade Recommendations ─────────────────────────────────────────────────────

def save_recommendation(rec, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("""
                INSERT OR REPLACE INTO trade_recommendations VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rec.rec_id, rec.pm_id, rec.ticker, rec.asset_class.value,
                rec.direction.value, rec.order_type.value, rec.limit_price,
                rec.target_notional_usd, rec.target_pct_nav, rec.rationale,
                rec.confidence_score, rec.expected_return_pct,
                rec.stop_loss_pct, rec.take_profit_pct, rec.time_horizon_days,
                rec.status.value, rec.timestamp.isoformat(),
            ))


def update_recommendation_status(rec_id: str, status: str, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("UPDATE trade_recommendations SET status=? WHERE rec_id=?",
                      (status, rec_id))


def get_recommendations(status: Optional[str] = None, db_path: str = PORTFOLIO_DB_PATH) -> list[dict]:
    if status:
        with _conn(db_path) as c:
            rows = c.execute(
                "SELECT * FROM trade_recommendations WHERE status=? ORDER BY timestamp DESC",
                (status,)
            ).fetchall()
    else:
        with _conn(db_path) as c:
            rows = c.execute(
                "SELECT * FROM trade_recommendations ORDER BY timestamp DESC LIMIT 50"
            ).fetchall()
    return [dict(r) for r in rows]


# ── Risk Records ──────────────────────────────────────────────────────────────

def save_risk_check(result, db_path: str = PORTFOLIO_DB_PATH) -> None:
    with _lock:
        with _conn(db_path) as c:
            c.execute("""
                INSERT OR REPLACE INTO risk_checks VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                result.rec_id, int(result.approved), result.veto_reason,
                result.portfolio_var_95, int(result.position_size_ok),
                int(result.sector_concentration_ok), int(result.drawdown_within_limit),
                int(result.correlation_flag), result.cro_notes,
                result.timestamp.isoformat(),
            ))


# ── Portfolio Analytics ───────────────────────────────────────────────────────

def compute_risk_metrics(db_path: str = PORTFOLIO_DB_PATH) -> dict:
    """Return key risk metrics from current portfolio state."""
    positions = get_positions(db_path)
    nav = get_nav(db_path)
    peak = get_peak_nav(db_path)

    if not positions or nav == 0:
        return {
            "nav": nav, "gross_exposure": 0, "net_exposure": 0,
            "var_95_1day": 0, "current_drawdown_pct": (peak - nav) / peak if peak else 0,
            "largest_position_pct": 0, "top_sector_concentration_pct": 0,
            "halt_triggered": is_halted(db_path),
            "sector_breakdown": {}, "pod_pnl": {},
        }

    gross = sum(p.notional_usd for p in positions)
    net = sum(
        p.notional_usd if p.direction == Direction.LONG else -p.notional_usd
        for p in positions
    )

    sector_exp: dict[str, float] = {}
    pod_pnl: dict[str, float] = {}
    for p in positions:
        s = p.sector or "unknown"
        sector_exp[s] = sector_exp.get(s, 0) + p.notional_usd
        pod = p.pod or "unknown"
        pod_pnl[pod] = pod_pnl.get(pod, 0) + p.unrealised_pnl

    largest_pct = max((p.pct_nav for p in positions), default=0)
    top_sector_pct = max((v / nav for v in sector_exp.values()), default=0)
    drawdown_pct = (peak - nav) / peak if peak > 0 else 0

    # Simplified VaR: 1.65 * 1% daily vol * gross (placeholder without real vol)
    var_95 = gross * 0.01 * 1.65

    return {
        "nav": nav,
        "gross_exposure": gross,
        "net_exposure": net,
        "var_95_1day": var_95,
        "current_drawdown_pct": drawdown_pct,
        "largest_position_pct": largest_pct,
        "top_sector_concentration_pct": top_sector_pct,
        "halt_triggered": is_halted(db_path),
        "sector_breakdown": {k: v / nav for k, v in sector_exp.items()},
        "pod_pnl": pod_pnl,
    }
