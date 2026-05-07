"""
Central configuration for the AI Hedge Fund system.
All API keys and operational parameters live here.
Set environment variables or populate .env before running.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Claude Model ──────────────────────────────────────────────────────────────
CLAUDE_MODEL: str = "claude-sonnet-4-5"
CLAUDE_MAX_TOKENS: int = 4096

# ── Database Paths ────────────────────────────────────────────────────────────
PORTFOLIO_DB_PATH: str = str(DATA_DIR / "portfolio.db")
AUDIT_DB_PATH: str = str(DATA_DIR / "audit.db")
CHROMA_PERSIST_DIR: str = str(DATA_DIR / "chroma")

# ── Risk Limits ───────────────────────────────────────────────────────────────
MAX_POSITION_SIZE_PCT: float = 0.05       # 5% of NAV per position
MAX_SECTOR_CONCENTRATION_PCT: float = 0.20 # 20% of NAV per sector
MAX_PORTFOLIO_DRAWDOWN_PCT: float = 0.15   # 15% drawdown triggers halt
LARGE_TRADE_APPROVAL_USD: float = 50_000   # CIO must approve trades > $50k notional

# ── Capital ───────────────────────────────────────────────────────────────────
INITIAL_NAV: float = 10_000_000.00        # $10M paper capital

# ── Capital Allocation by Pod (% of NAV) ─────────────────────────────────────
POD_ALLOCATION: dict[str, float] = {
    "long_short": 0.35,
    "macro": 0.25,
    "quant": 0.25,
    "event_driven": 0.15,
}

# ── Scheduler ─────────────────────────────────────────────────────────────────
DAILY_CYCLE_HOUR: int = 7    # 7:00 AM ET
DAILY_CYCLE_MINUTE: int = 0

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
