# AI Hedge Fund

A fully autonomous, hierarchical AI hedge fund system. Every role below the Founder/CEO is staffed by an AI agent powered by Claude. Agents communicate through a message bus, maintain long-term memory, enforce hard risk limits, and run a structured daily investment cycle — all in paper-trading simulation.

---

## Table of Contents

1. [Overview](#overview)
2. [Fund Structure & Org Chart](#fund-structure--org-chart)
   - [Your Role: CEO/Founder](#your-role-ceofounder)
   - [Layer 1 — C-Suite](#layer-1--c-suite)
   - [Layer 2 — Portfolio Managers](#layer-2--portfolio-managers)
   - [Layer 3 — Analysts & Specialists](#layer-3--analysts--specialists)
3. [How the System Works](#how-the-system-works)
   - [The Message Bus](#the-message-bus)
   - [Agent Memory](#agent-memory)
   - [The Trade Pipeline](#the-trade-pipeline)
   - [The Daily Cycle](#the-daily-cycle)
   - [Risk Controls & Hard Limits](#risk-controls--hard-limits)
   - [Fund Accounting](#fund-accounting)
4. [Setup & Installation](#setup--installation)
5. [Using the System as CEO](#using-the-system-as-ceo)
   - [The Streamlit Dashboard](#the-streamlit-dashboard)
   - [The CLI](#the-cli)
6. [Configuration Reference](#configuration-reference)
7. [File Structure](#file-structure)
8. [Important Notes](#important-notes)

---

## Overview

This system simulates a real hedge fund org structure where every role is executed by a Claude AI agent. You, the Founder/CEO, issue high-level directives. The CIO decomposes them into subtasks, dispatches Portfolio Managers, who task their Analysts, who return research packets. The CRO and Compliance Agent gate every trade. The CFO marks books to market. The COO monitors system health. All decisions are logged to an immutable audit trail.

**Everything is paper trading.** No real money moves. All positions are simulated fills at live market prices fetched from yfinance.

**Starting capital:** $10,000,000 (configurable in `config.py`).

---

## Fund Structure & Org Chart

```
                          ┌─────────────────┐
                          │   CEO / Founder  │  ← You
                          └────────┬────────┘
                                   │ directives / approvals
                    ┌──────────────▼──────────────┐
                    │         C-SUITE              │
          ┌─────────┤                              ├─────────┐
          │         │  CIO   CRO   CFO   COO       │         │
          │         └──────────────────────────────┘         │
          │                       │                           │
          │            ┌──────────▼──────────┐               │
          │            │    PM LAYER          │               │
          │   ┌────────┼──────────────────────┼────────┐     │
          │   │        │                      │        │     │
          │  PM       PM        PM           PM        │     │
          │  L/S     Macro     Quant       Event       │     │
          │   │        │        │          Driven       │     │
          └───▼────────▼────────▼────────────▼─────────┘     │
                          ANALYST LAYER                        │
          ┌───────────────────────────────────────────────┐   │
          │  FA1  FA2  QuantA  MacroA  DataEng  Comp  MM  │   │
          └───────────────────────────────────────────────┘
```

### Your Role: CEO/Founder

You sit above the org chart. Your interaction points are:

- **Issue directives** to the CIO (e.g. "Analyze semiconductors for AI infrastructure longs")
- **Read memos and reports** from the C-Suite (morning memo, risk report, P&L summary)
- **Approve or veto trades** that exceed the $50,000 large-trade threshold
- **Monitor the portfolio** — live positions, P&L, sector exposure
- **Review the audit log** — every agent-to-agent message, timestamped

You never talk directly to PMs or Analysts. Everything flows through the CIO.

---

### Layer 1 — C-Suite

#### CIO — Chief Investment Officer (`agents/csuite/cio.py`)

The CIO is your primary counterpart. Every CEO directive lands here first.

**What it does:**
- Receives your directives and runs an LLM-driven decomposition: which PMs are relevant, what specific research questions should each one pursue, and what the macro backdrop means for this theme
- Sets capital allocation across the four strategy pods
- Dispatches specific tasks to PMs with context (capital available, macro view, timeline)
- Reviews and approves trades above the $50,000 large-trade threshold
- Synthesizes overnight market data, PM research, and risk metrics into the **morning memo** — your primary daily read

**Pod capital allocations (default):**

| Pod | % of NAV | $ (on $10M) |
|---|---|---|
| Long/Short Equity | 35% | $3,500,000 |
| Global Macro | 25% | $2,500,000 |
| Quant | 25% | $2,500,000 |
| Event Driven | 15% | $1,500,000 |

---

#### CRO — Chief Risk Officer (`agents/csuite/cro.py`)

The CRO is the last line of defense before any trade executes. It has **absolute veto power**.

**What it does:**
- Runs a full risk check on every trade recommendation after Compliance clears it
- Enforces hard position, sector, and drawdown limits (see [Risk Controls](#risk-controls--hard-limits))
- Produces a daily portfolio risk summary with VaR, drawdown, sector concentration
- Triggers a **full trading halt** if the portfolio drawdown exceeds 15%
- Sends risk reports to the CEO and CIO automatically

**Risk check sequence for each trade:**
1. Position size check (≤ 5% NAV)
2. Sector concentration check (sector total ≤ 20% NAV after adding position)
3. Portfolio drawdown check (current drawdown < 15%; if breached → halt immediately)
4. LLM qualitative assessment: correlation to existing book, risk/reward judgment

---

#### CFO — Chief Financial Officer (`agents/csuite/cfo.py`)

Maintains accurate fund accounting. Runs mark-to-market on demand or during the daily cycle.

**What it does:**
- Marks all positions to current live market prices (yfinance)
- Tracks NAV, unrealized P&L, cash balance, and fee accruals
- Accrues fees daily: **2% annual management fee** (NAV × 2% ÷ 252 per day) and **20% performance fee** on profits above the high-water mark
- Produces the daily capital report sent to the CEO and CIO
- Flags cash management concerns (low cash, high leverage)

---

#### COO — Chief Operating Officer (`agents/csuite/coo.py`)

Monitors system health. Thinks of it as the ops layer watching all agents.

**What it does:**
- Polls every agent's message queue depth — flags queues > 10 unread messages as a bottleneck
- Tracks error rates and response times per agent
- Produces a traffic-light ops report: **green** (<5% error rate) / **yellow** (5–20%) / **red** (>20%)
- Sends URGENT alerts to the CEO if system health is red
- Nudges overloaded agents to clear their queues

---

### Layer 2 — Portfolio Managers

Each PM runs one strategy pod. PMs are instantiated from a shared `BasePMAgent` class (`agents/pms/pm_base.py`) and override `run_pm_task()` with pod-specific logic.

**Every PM:**
- Receives capital allocation and a research mandate from the CIO
- Tasks their specific analyst roster with targeted questions
- Synthesizes analyst research into `TradeRecommendation` objects (Pydantic-validated)
- Sends recommendations through the pipeline: Compliance → CRO → (CIO if large) → execute
- Tracks their pod's open positions and unrealized P&L

#### PM Long/Short (`agents/pms/pm_longshort.py`)

- **Focus:** US equity stock picking, pair trades, sector rotation
- **Target:** Net exposure -20% to +60%, Sharpe > 1.5
- **Analysts it tasks:** Fundamental Analyst 1, Fundamental Analyst 2, Data Engineer
- **Sizing:** 1–3% NAV for high-conviction, 0.5–1% for speculative
- **Stop losses:** 5–8% | **Take profits:** 15–25%

#### PM Macro (`agents/pms/pm_macro.py`)

- **Focus:** Rates (TLT, IEF, TBT), FX (FXE, FXY, UUP), Commodities (GLD, USO, UNG), EM (EEM, VWO)
- **Target:** Sharpe > 1.0, equity correlation < 0.3
- **Analysts it tasks:** MacroAnalyst, Data Engineer
- **Instruments:** ETF proxies for futures (no direct futures access in simulation)

#### PM Quant (`agents/pms/pm_quant.py`)

- **Focus:** Systematic factor strategies — momentum, value, quality, low-vol, mean reversion
- **Target:** Sharpe > 1.2, monthly turnover < 50%
- **Analysts it tasks:** QuantAnalyst, Data Engineer
- **Gate:** Full backtest required before any strategy goes live; live IR drop > 30% below backtest IR triggers stop

#### PM Event Driven (`agents/pms/pm_eventdriven.py`)

- **Focus:** M&A arbitrage (long target, optional short acquirer), earnings plays, spinoffs, regulatory catalysts
- **Target:** Hit rate > 55%, low market beta
- **Analysts it tasks:** Fundamental Analyst 1, Fundamental Analyst 2, Market Monitor
- **M&A arb rule:** Skip if spread < 2% (not worth regulatory/break risk)
- **Earnings rule:** Never hold full size through an earnings print if conviction < 0.7

---

### Layer 3 — Analysts & Specialists

#### Fundamental Analyst 1 & 2 (`agents/analysts/fundamental_analyst.py`)

Two instances running identical code, differentiated by `agent_id` (`fundamental_analyst_1`, `fundamental_analyst_2`). Running two allows the Long/Short and Event-Driven PMs to task them simultaneously without queue contention.

**What they produce:**
- DCF valuation (intrinsic value estimate, upside/downside %)
- Key financial metrics: revenue growth, EBITDA margin, FCF conversion, leverage
- SEC filing review: pulls 10-K/10-Q excerpts via `sec-edgar-downloader`
- Earnings analysis vs consensus
- Structured `FundamentalResearchPacket` with confidence score and reasoning chain
- Sector pair-trade analysis (best-in-class vs laggards)

**Data sources:** yfinance (prices + fundamentals), Anthropic web_search, SEC EDGAR

#### Quant Analyst (`agents/analysts/quant_analyst.py`)

- Builds quantitative signals from scratch using sandboxed Python execution
- Has access to the `run_python` tool — writes and executes backtest code in a subprocess (30-second timeout, no network access, no filesystem writes)
- Produces `QuantSignalPacket` with: strategy name, signal z-score, expected return (bps), annualized vol, Sharpe estimate, backtest Sharpe, backtest max drawdown
- Available libraries in sandbox: numpy, pandas, scipy, sklearn, statsmodels, yfinance, matplotlib

#### Macro Analyst (`agents/analysts/macro_analyst.py`)

- Interprets central bank policy (Fed, ECB, BOJ, BOE, PBoC)
- Analyzes economic data releases: CPI, PCE, NFP, GDP, PMI
- Produces `MacroResearchPacket` with: macro theme, affected asset classes, time horizon, confidence
- Generates the **overnight brief** at the start of every daily cycle
- Uses web_search for latest CB statements and macro data

#### Data Engineer (`agents/analysts/data_engineer.py`)

- Pulls and quality-checks price data for any list of tickers
- Flags: stale prices (unchanged >2 days), large moves >15% (possible split), missing data
- Provides clean data summaries including: 30d/YTD returns, 52W high/low, short interest, beta
- Runs portfolio-wide data quality checks on request; alerts CFO on anomalies
- Returns a `data_quality_score` (0–1) with every response

#### Compliance Agent (`agents/analysts/compliance_agent.py`)

Every single trade passes through here before touching the CRO. There are no exceptions.

**Rules enforced:**
1. **Restricted list** — hardcoded set of blocked tickers (editable in `compliance_agent.py`)
2. **Position limit** — ≤ 5% of NAV (redundant with CRO; belt-and-suspenders)
3. **Wash sale** — no repurchase within 30 days of a loss sale
4. **Pattern day trading** — flags ≥ 3 round-trips in any ticker within 5 days
5. **Leverage** — gross exposure must stay ≤ 200% NAV
6. **Qualitative LLM review** — catches edge cases: manipulation patterns, unusual timing, obvious red flags

Returns a `ComplianceCheckResult` (approved/rejected, violations list).

#### Market Monitor (`agents/analysts/market_monitor.py`)

Runs a monitoring cycle during the daily scheduler and on demand. Acts as the real-time alert system.

**Watchlist (default):**
```
Macro:   ^GSPC, ^VIX, ^TNX, GLD, CL=F
Equities: AAPL, MSFT, NVDA, AMZN, META
Sectors: XLK, XLF, XLE, XLY
```

**Alert thresholds:**
- Price move ≥ 3% → HIGH alert to CIO + CRO
- Price move ≥ 5% → URGENT alert
- VIX > 25 → HIGH alert to CIO, CRO, PM Macro
- VIX > 35 → URGENT alert
- 10Y Treasury move ≥ 10bps → HIGH alert to CIO, PM Macro
- News scan: deduplicates; only flags genuinely market-moving headlines

---

## How the System Works

### The Message Bus

Every agent-to-agent communication runs through a **SQLite-backed message bus** (`messaging/bus.py`). Nothing communicates out-of-band.

**Key properties:**
- Messages have a sender, recipient, subject, body, and optional JSON payload
- Priority levels: `urgent > high > normal > low` — agents process higher-priority messages first
- Every message is also written to an **immutable audit log table** — sent, received, and replied events are all recorded with timestamps
- `thread_id` links replies to their originating message for full conversation reconstruction
- `requires_reply=True` flags a message as needing a response

**How agents communicate:**

```
CIO.send_message(recipient="pm_longshort", subject="CIO Task: ...", body="Research semiconductors...")
→ written to messages table (recipient=pm_longshort, read_at=NULL)

PM.receive_messages()
→ reads rows WHERE recipient=pm_longshort AND read_at IS NULL
→ marks read_at, writes received event to audit_log

PM.send_message(recipient="fundamental_analyst_1", ...)
→ written as new message
```

The `HedgeFundOrchestrator.flush_messages(rounds=N)` method calls `process_inbox()` on every agent N times in sequence, propagating messages through the hierarchy. One round = one hop. Three rounds handles CEO→CIO→PM→Analyst→PM→CIO chains.

---

### Agent Memory

Each agent has two memory layers:

**Short-term (in-process):** A rolling list of `{"role": "user"/"assistant", "content": "..."}` messages passed to Claude on every API call. Capped at `max_context_messages * 2` (default 40 messages). This is the conversational context window.

**Long-term (persistent):** A **ChromaDB collection** per agent stored in `hedge_fund/data/chroma/`. Every agent stores:
- Research packets it produced (with ticker metadata)
- Past decisions and their outcomes
- Task logs (task description + response summary)

On every LLM call, the base agent automatically performs a semantic search against the agent's own ChromaDB collection using the current query as the search key, and prepends the top 3 relevant past memories as context. This means agents learn over time — an analyst who researched NVDA last week will surface that research when asked about semiconductors again.

---

### The Trade Pipeline

A trade recommendation travels through four gates before it's recorded as executed:

```
PM synthesizes research
        │
        ▼
 TradeRecommendation created
 (status: pending_compliance)
        │
        ▼
 ┌── Compliance Agent ──┐
 │  Rules check          │
 │  • Restricted list    │
 │  • Position limit     │
 │  • Wash sale          │
 │  • PDT check          │
 │  • Leverage cap       │
 │  • LLM qualitative    │
 └──────────────────────┘
    │ APPROVED                │ REJECTED → status=rejected, PM notified
    ▼
 status: pending_risk
    │
    ▼
 ┌── CRO Risk Check ────┐
 │  Hard limits          │
 │  • Position size ≤5%  │
 │  • Sector conc ≤20%  │
 │  • Drawdown < 15%    │
 │  • LLM correlation   │
 └──────────────────────┘
    │ APPROVED                │ VETOED → status=rejected, reason logged
    ▼
 status: approved
    │
    ├── notional > $50k? ──► CIO large-trade approval
    │                             │ APPROVED
    │◄────────────────────────────┘
    ▼
 Execute trade (simulated fill at live market price)
    │
    ├── BlotterEntry written to portfolio.db
    ├── Position upserted (new or averaged-in)
    ├── Cash balance updated
    └── CIO notified: TRADE_EXECUTED
```

**Simulated fill:** The system calls `yfinance.Ticker(ticker).history(period="2d")` to get the latest close as the fill price. Quantity = notional / fill_price. No slippage model is applied (paper trading).

---

### The Daily Cycle

The daily cycle runs automatically via `scheduler.py` at **7:00 AM UTC** (configurable). You can also trigger it manually. The sequence:

```
07:00 UTC — Stage 1: Market Monitor
              └─ Checks watchlist for overnight moves
              └─ Scans for news headlines
              └─ Alerts CIO/CRO if thresholds breached

              Stage 2: MacroAnalyst overnight brief
              └─ Pulls live macro indicators (S&P, VIX, 10Y, Gold, Oil, FX)
              └─ Web-searches for overnight CB statements and economic data
              └─ Sends overnight brief to CIO and PM Macro

              Stage 3: PM morning briefings (all 4 PMs in parallel)
              └─ Each PM reviews its positions and overnight moves
              └─ Tasks analysts with morning research questions
              └─ Returns initial view to CIO

              Stage 4: Message flush (2 rounds)
              └─ Analysts process PM tasks and respond
              └─ PMs receive analyst research
              └─ PMs may generate trade recommendations

              Stage 5: CRO risk report
              └─ Marks positions to market
              └─ Computes VaR, drawdown, sector concentrations
              └─ Sends report to CEO and CIO

              Stage 6: CFO P&L report
              └─ Full mark-to-market
              └─ Computes daily/MTD/YTD P&L
              └─ Accrues management + performance fees
              └─ Sends capital report to CEO and CIO

              Stage 7: CIO morning memo
              └─ Synthesizes all of the above
              └─ Produces market view, top 3 trade ideas, top 3 risks
              └─ Available in your dashboard under "Morning Memo"

              Stage 8: COO ops check
              └─ Polls all 15 agent queue depths
              └─ Reports system health (green/yellow/red)
              └─ Alerts CEO if any bottlenecks
```

---

### Risk Controls & Hard Limits

These limits are enforced by code — not just LLM judgment — and cannot be argued around:

| Control | Limit | Enforced by |
|---|---|---|
| Max single position size | 5% of NAV | CRO + Compliance (dual check) |
| Max sector concentration | 20% of NAV | CRO |
| Max portfolio drawdown | 15% → trading halt | CRO |
| Max gross leverage | 200% of NAV | Compliance |
| Large trade threshold | $50,000 notional | CIO must approve |
| Pattern day trading | ≥ 3 round-trips/5 days per ticker | Compliance (flagged) |
| Restricted securities | Configurable list | Compliance (hard block) |
| Wash sale | 30-day lookback | Compliance (flagged) |

**Trading halt:** When portfolio drawdown hits 15%, `shared_state.set_halt(True)` is called. The CRO immediately rejects all incoming trade recommendations with "TRADING HALT: Portfolio drawdown limit breached." The halt persists until manually cleared (set `halt=false` in `hedge_fund/data/portfolio.db` fund_state table, or add a `clear_halt()` call in the CLI).

---

### Fund Accounting

The CFO marks all positions to market using live yfinance prices. NAV is computed as:

```
NAV = cash_balance + Σ(position_notional)
    where position_notional = current_price × quantity
```

For short positions, unrealized P&L = (avg_entry_price − current_price) × quantity.

**Fees (accrued daily):**
- Management fee: NAV × 2% ÷ 252 per trading day
- Performance fee: 20% × daily_profit (only on days when NAV > high-water mark)

The high-water mark starts at $10M and only moves up — performance fees are only charged on new profit records.

---

## Setup & Installation

### 1. Prerequisites

- Python 3.11+
- An Anthropic API key with access to `claude-sonnet-4-5`
- (Optional) An Alpha Vantage API key for news sentiment

### 2. Clone and install

```bash
git clone <repo-url>
cd AI-fund
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-api03-...
ALPHA_VANTAGE_API_KEY=          # optional
LOG_LEVEL=INFO                   # DEBUG for verbose agent output
```

The system reads these via `os.getenv()`. You can also export them as shell environment variables instead of using a `.env` file — the system doesn't auto-load `.env`; you need to source it first or use a tool like `python-dotenv`.

**To auto-load the `.env` file**, add this to the top of `hedge_fund/config.py`:

```python
from dotenv import load_dotenv
load_dotenv()
```

And install: `pip install python-dotenv`

### 4. Verify setup

```bash
python -c "
import sys; sys.path.insert(0, '.')
from hedge_fund.messaging.bus import init_bus
from hedge_fund.memory.shared_state import init_portfolio_db, get_nav
init_bus(); init_portfolio_db()
print(f'NAV: \${get_nav():,.0f}')
"
```

Expected output: `NAV: $10,000,000`

### 5. Data directory

On first run the system creates `hedge_fund/data/` with:
- `audit.db` — the message bus and audit log
- `portfolio.db` — positions, blotter, P&L, risk records
- `chroma/` — ChromaDB vector stores for all agents

These are excluded from git (`.gitignore`). They persist between runs.

---

## Using the System as CEO

### The Streamlit Dashboard

The dashboard is your primary interface. Launch it with:

```bash
streamlit run hedge_fund/dashboard/app.py
```

It opens in your browser at `http://localhost:8501`.

The left sidebar shows live NAV, cash, current drawdown, and a trading halt warning if active. Navigation is via the radio buttons in the sidebar.

---

#### Page: Morning Memo

Your daily read. Click **"Refresh Memo"** to have the CIO generate a fresh morning memo from current market data, open positions, and overnight intelligence.

The memo contains:
- **Market View** — 2–3 sentences on the macro backdrop, written by the CIO
- **Active Positions Summary** — what's in the book and how it's performing
- **Top 3 Trade Ideas** — the CIO's best current opportunities across all pods
- **Key Risks** — the top 3 things that could hurt the portfolio today
- **Pod Allocations** — current capital split across the four strategy pods

Use this page first thing every morning (or click "Run Cycle" to generate it automatically).

---

#### Page: Issue Directive

This is how you talk to the fund. Type a directive in natural language and send it to the CIO.

**Good directive examples:**

```
"Analyze the semiconductor sector for long opportunities. Focus on 
AI infrastructure names — equipment, EDA, and leading-edge fabs."

"Build a macro trade around Fed pivot expectations. Look at rate 
sensitivity across equities and bonds."

"The energy sector looks interesting given geopolitical tensions. 
Assess long opportunities in oil majors vs. renewables short."

"Run a quant screen for momentum names in large-cap tech. 
I want a systematic signal with 6-month backtest."

"There's an M&A rumor in biotech — find any announced deals 
with arb spreads worth trading."
```

The directive is sent to the CIO, which decomposes it into specific PM tasks using an LLM call, dispatches those tasks with capital allocation context, and returns a summary of what was dispatched to whom.

After sending a directive, click the auto-flush button that appears (or manually run "Flush Message Bus" on the Run Cycle page) to propagate messages through the org: CIO → PMs → Analysts → PMs → CIO.

**CEO Inbox** below the directive input shows messages addressed directly to you — the CIO's acknowledgments, CRO halt alerts, COO system warnings.

---

#### Page: Portfolio

Live portfolio state. Refreshes from the database on every page load.

**Top metrics row:**
- NAV, Gross Exposure, Net Exposure, VaR 95% (1-day), Current Drawdown

**Open Positions table:** every live position with ticker, direction (LONG/SHORT), quantity, average entry, current price, notional value, unrealized P&L, % of NAV, pod, and sector.

**Recent Trades (Blotter):** the last 20 simulated fills — what executed, at what price, from which PM and pod.

**Sector Breakdown:** current exposure by sector as % of NAV.

---

#### Page: Risk

The CRO's risk dashboard. Click **"Refresh Risk Report"** to generate a fresh assessment.

Shows:
- NAV, drawdown, VaR
- Gross and net exposure
- Largest single position as % of NAV
- Pod-level unrealized P&L
- Sector concentration table
- Hard limit reference (what the limits are and where you currently stand)

A red banner appears if a trading halt is active.

---

#### Page: Trade Approvals

Trades requiring your attention are surfaced here. Currently this shows all recommendations grouped by status: `PENDING_CIO`, `APPROVED`, `REJECTED`, `EXECUTED`.

Trades flagged `PENDING_CIO` have already passed Compliance and CRO risk checks — they're large trades (> $50,000 notional) waiting for your explicit sign-off.

For each pending trade you see:
- Ticker, direction, notional, % NAV
- Which PM submitted it
- Expected return, stop loss, take profit, time horizon
- Full rationale text
- Confidence score

You have two buttons: **Approve** (sets status to `approved`, PM executes the trade) or **Veto** (sets status to `rejected`, logged to audit trail).

---

#### Page: Audit Log

Every agent-to-agent message ever sent, with timestamps. This is the full audit trail of the fund's decision-making.

Filter by **sender** (e.g. `cio`, `pm_longshort`) or **recipient** (e.g. `cro`, `ceo`) to trace any decision thread. Set max rows up to 500.

The body column is truncated to 150 chars for readability in the table — to read full message bodies, query `hedge_fund/data/audit.db` directly:

```sql
SELECT sender, recipient, subject, body, timestamp
FROM audit_log
WHERE sender = 'cio'
ORDER BY timestamp DESC
LIMIT 20;
```

---

#### Page: Run Cycle

Manual controls for operating the fund outside the automated scheduler.

**Run Full Daily Cycle** — triggers all 8 stages of the daily cycle synchronously. This takes several minutes because it makes multiple Claude API calls. A spinner shows while it runs; results appear stage by stage in JSON.

**Flush Message Bus** — processes all pending inbox messages across all 15 agents for 3 rounds. Run this after issuing a directive to let research propagate through the hierarchy.

**Individual Reports** — trigger any single C-suite report in isolation: CFO P&L, CRO Risk, or COO Ops. Faster than running the full cycle if you just need one report.

---

### The CLI

For a terminal-first workflow, use the interactive CLI:

```bash
python -m hedge_fund.main
```

Or with flags:

```bash
# Run the full daily cycle and print all results
python -m hedge_fund.main --cycle

# Issue a one-shot directive and flush the bus
python -m hedge_fund.main --directive "Analyze semiconductor sector for AI longs"

# Issue a directive with high priority
python -m hedge_fund.main --directive "Emergency: assess exposure to regional bank contagion" --priority urgent

# Start the automated scheduler (blocking — runs forever)
python -m hedge_fund.main --scheduler
```

**Interactive CLI commands:**

```
CEO> directive <text>     Issue a directive to the CIO
CEO> cycle                Run the full daily cycle
CEO> memo                 Get CIO morning memo (prints to terminal)
CEO> risk                 Get CRO risk report
CEO> pnl                  Get CFO P&L report
CEO> portfolio            List all open positions
CEO> audit                Show the 10 most recent audit log entries
CEO> flush                Flush the message bus (3 rounds)
CEO> quit                 Exit
```

**Example session:**

```
CEO> directive Analyze the energy sector. Oil is moving on geopolitical risk.
      I want long exposure to integrated majors, and a macro hedge via crude ETF.

CIO Response: Directive 3f8a1c decomposed. Macro view: Elevated geopolitical
risk in MENA region creating supply disruption premium...
Dispatched 3 PM tasks.
Flushing message bus...
Processed 7 messages through the org.

CEO> audit
  [2026-05-07 07:23] cio → pm_macro: CIO Task: Analyze energy sector...
  [2026-05-07 07:23] cio → pm_longshort: CIO Task: Analyze energy sector...
  [2026-05-07 07:23] cio → coo: CIO Directive Processing [3f8a1c]
  [2026-05-07 07:23] macro_analyst → pm_macro: MACRO RESEARCH: Energy...
  [2026-05-07 07:23] fundamental_analyst_1 → pm_longshort: RESEARCH RESPONSE...

CEO> portfolio
Open Positions (2):
  XOM LONG $320,000 (3.2% NAV) P&L: $4,200
  USO LONG $180,000 (1.8% NAV) P&L: $1,100

CEO> risk
  NAV: $10,000,000
  Drawdown: 0.00%
  VaR 95% (1d): $8,250
  Halt: No
```

---

### Starting the Automated Scheduler

For fully autonomous operation, run the scheduler as a background process:

```bash
python -m hedge_fund.main --scheduler &
```

The scheduler runs:
- The full daily cycle at **7:00 AM UTC** every day
- The Market Monitor every **15 minutes**
- A message bus flush every **5 minutes**

Alerts from the Market Monitor are routed to the CEO inbox (readable on the "Issue Directive" page of the dashboard or via `CEO> audit` in the CLI).

---

## Configuration Reference

All configuration lives in `hedge_fund/config.py`. Edit these values directly or set them via environment variables.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required. Your Anthropic API key |
| `ALPHA_VANTAGE_API_KEY` | — | Optional. For news sentiment |
| `CLAUDE_MODEL` | `claude-sonnet-4-5` | Claude model for all agents |
| `CLAUDE_MAX_TOKENS` | `4096` | Max tokens per agent response |
| `INITIAL_NAV` | `10_000_000` | Starting paper capital ($) |
| `MAX_POSITION_SIZE_PCT` | `0.05` | Hard max per position (5% NAV) |
| `MAX_SECTOR_CONCENTRATION_PCT` | `0.20` | Hard max per sector (20% NAV) |
| `MAX_PORTFOLIO_DRAWDOWN_PCT` | `0.15` | Drawdown halt trigger (15%) |
| `LARGE_TRADE_APPROVAL_USD` | `50_000` | Trades above this need CIO approval |
| `DAILY_CYCLE_HOUR` | `7` | Hour (UTC) for daily cycle |
| `DAILY_CYCLE_MINUTE` | `0` | Minute for daily cycle |
| `LOG_LEVEL` | `INFO` | `DEBUG` for full agent output |

**Pod allocations** (in `POD_ALLOCATION` dict):
```python
"long_short": 0.35,   # 35%
"macro": 0.25,        # 25%
"quant": 0.25,        # 25%
"event_driven": 0.15, # 15%
```

Change these to reallocate capital across pods. They take effect on the next CIO directive dispatch.

---

## File Structure

```
AI-fund/
├── requirements.txt              # Python dependencies
├── setup.py
├── .env.example                  # Copy to .env and fill in keys
└── hedge_fund/
    ├── config.py                 # All configuration & constants
    ├── main.py                   # CLI entry point
    ├── scheduler.py              # HedgeFundOrchestrator + daily cycle
    │
    ├── agents/
    │   ├── base_agent.py         # BaseAgent: LLM calls, memory, messaging, tools
    │   ├── csuite/
    │   │   ├── cio.py            # Chief Investment Officer
    │   │   ├── cro.py            # Chief Risk Officer
    │   │   ├── cfo.py            # Chief Financial Officer
    │   │   └── coo.py            # Chief Operating Officer
    │   ├── pms/
    │   │   ├── pm_base.py        # Shared PM base class
    │   │   ├── pm_longshort.py   # Long/Short Equity PM
    │   │   ├── pm_macro.py       # Global Macro PM
    │   │   ├── pm_quant.py       # Quantitative Strategies PM
    │   │   └── pm_eventdriven.py # Event Driven PM
    │   └── analysts/
    │       ├── fundamental_analyst.py  # FA1 + FA2 (instantiated with analyst_number=)
    │       ├── quant_analyst.py        # Factor models + backtest code execution
    │       ├── macro_analyst.py        # CB policy + economic data
    │       ├── data_engineer.py        # Data quality + price pipelines
    │       ├── compliance_agent.py     # Trade compliance screening
    │       └── market_monitor.py       # Real-time price/news alerts
    │
    ├── tools/
    │   ├── market_data.py        # yfinance + Alpha Vantage wrappers
    │   ├── web_search.py         # Anthropic web_search tool
    │   ├── sec_filings.py        # SEC EDGAR downloader
    │   └── code_executor.py      # Sandboxed Python subprocess
    │
    ├── memory/
    │   ├── vector_store.py       # AgentMemory: ChromaDB per-agent collection
    │   └── shared_state.py       # Portfolio DB: positions, blotter, P&L, risk
    │
    ├── messaging/
    │   └── bus.py                # SQLite message bus + audit log
    │
    ├── models/
    │   └── schemas.py            # All Pydantic models for structured outputs
    │
    └── dashboard/
        └── app.py                # Streamlit CEO interface

hedge_fund/data/                  # Created at runtime (gitignored)
    ├── audit.db                  # Message bus + audit log
    ├── portfolio.db              # Positions, blotter, P&L, risk checks
    └── chroma/                   # ChromaDB vector stores (one per agent)
```

---

## Important Notes

**This is paper trading only.** No real orders are routed anywhere. All fills are simulated at the last yfinance close price. The system makes no connections to brokers, exchanges, or real trading APIs.

**API costs.** Every agent response requires a Claude API call. A full daily cycle makes roughly 20–30 API calls (one per agent stage). Running the full cycle continuously will accumulate API usage. Use `LOG_LEVEL=DEBUG` to monitor exactly which calls are being made.

**Rate limits.** If you hit Anthropic rate limits, the `BaseAgent._call_llm()` method will automatically wait 30 seconds and retry. For heavy use, run the daily cycle during off-peak hours or add your own rate-limit logic in `base_agent.py`.

**First run.** ChromaDB collections are empty on first run — agents have no memories yet. The system works fine; agents simply won't surface past research in context. Memory accumulates naturally as the fund operates.

**Resetting the fund.** Delete the `hedge_fund/data/` directory to wipe all state (positions, history, agent memories, audit log) and start fresh from $10M NAV.

**Adding tickers to the restricted list.** In `agents/analysts/compliance_agent.py`, edit the `RESTRICTED_TICKERS` set at the top of the file. Or call `compliance_agent.add_to_restricted_list("TICKER", "reason")` at runtime — this also notifies the CIO.

**Extending the system.** To add a new agent: subclass `BaseAgent`, set `agent_id`, `role_name`, and `system_prompt`, implement `handle_message()`, and register it in `HedgeFundOrchestrator._agents` in `scheduler.py`. The message bus will automatically route to it.
