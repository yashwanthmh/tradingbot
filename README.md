# tradingbot

An autonomous long-only equity trading bot for **Trading 212**: it generates its own
strategies, gates them against out-of-sample backtests, allocates its own capital, executes
and protects its own positions, and records every step in a hash-chained audit ledger that
can explain any trade months later.

> **This trades real money and can lose it.** Nothing here is financial advice, and no
> backtest result is a prediction. Read [Risk and honest limitations](#risk-and-honest-limitations)
> before arming live mode — particularly the part about why the fee schedule, not the
> strategy, is the binding constraint on this venue.

## Why it is built this way

Three properties of Trading 212's API shape almost every design decision:

**1. There is no market data in the API.** No quotes, no candles, no websocket. The
execution venue is therefore not the data venue, which creates a class of bug that
single-venue systems never see: a signal computed on one venue's prices and filled at
another's. Handled by a symbol map plus a cross-venue price reconciliation gate — when the
data feed and the broker's own quote disagree beyond a band, that symbol stops accepting
new entries.

**2. Cost, not alpha, is the binding constraint.** Trading is commission-free, which is not
the same as free: 0.15% per FX conversion and 0.5% stamp duty on UK purchases put a round
trip at roughly 30–55bps. Gross edge on minute-bar signals in liquid names is 5–20bps. So
the system targets **minute-resolution features with hour-to-day position changes**, and
every order must clear a pre-trade gate on `expected_cost_bps / expected_edge_bps`. A
strategy that will not declare its expected edge cannot trade.

**3. There are no idempotency tokens and no bracket orders, and writes outrun reads.**
Market orders POST at 50/60s while the order list reads at 1/5s and fill history at 6/60s —
the system can create state five times faster than it can observe it. So exactly-once
submission is synthesised client-side through a write-ahead intent log, and an intent whose
state is unknown after a crash is treated as *unknown*, never as failed. Treating unknown
as failed is how double-fills happen.

And one consequence of the account type: the API covers Invest and Stocks ISA only, so the
system is **long-only and unlevered**. Cash is the sole defensive position. That bounds
what the strategy search is even permitted to invent, and it means every live strategy is a
long-equity beta expression — so portfolio risk is managed by a regime gate above the
allocator, not by diversifying across strategies that all correlate to one in a drawdown.

## Architecture

```
 CONTROL LAYER (config/hard_limits.yaml — read-only mount, hash-pinned, re-verified each cycle)
      │  bounds everything below; raising a ceiling is a human action
      ▼
 1 RESEARCH ──────► 2 GATE ──────► 3 ALLOCATE ──────► 4 DECIDE ──────► 5 EXECUTE
 genetic search     cost-aware     capital by         minute bars       intent WAL
 over a strategy    OOS backtest   prior, shrunk      → features        → risk token
 DSL (LLM is an     + sealed       toward realised    → signal          → T212 order
 optional plug-in)  holdout, DSR   + correlation      + edge estimate   → broker stop
 + ML models       + null-gen         caps                 │                 │
      ▲             calibration                            ▼                 ▼
      │                                              6 RECORD ◄────── 7 RECONCILE
      └──────────── 8 REVIEW ◄───────────────────────  hash-chained     3 axes: intents
        KEEP / KILL / ITERATE / SCALE                  event log        vs open orders
        kill on little evidence, scale on a lot        + full lineage    vs position/cash
```

## Guardrails

Ten controls, none of them optional:

| Control | What it stops |
|---|---|
| Hash-pinned immutable limits | Caps being loosened by the process they constrain |
| Absolute currency ceiling | The bot enlarging its own blast radius |
| Per-strategy / per-lineage loss budgets | A losing idea returning under a new name |
| Asymmetric size ratchet | A lucky streak being mistaken for an edge |
| Daily / rolling / drawdown breakers | A bad day becoming a bad month |
| **Anomaly circuit breaker** | The runaway loop — the failure that actually bankrupts bots |
| Cost-aware pre-trade rejection | Strategies whose edge is smaller than their fees |
| Cross-venue disagreement block | Trading on a price the broker does not recognise |
| Bidirectional dead-man switch | A stalled process holding unprotected positions |
| `RiskToken` on the order path | A refactor quietly bypassing the risk engine |

That last one is structural rather than conventional: `Broker.place_order` requires a token
only `RiskEngine` can construct, and a test walks the AST to assert no other construction
site exists anywhere in the codebase. "Every order goes through the risk engine" as a code
review rule survives about three months.

The kill switch is **fail-closed**. A missing switch file, an unreadable one, or a
permission error all mean *killed*. Undeterminable is not permission to trade.

## Every move, recorded

`event_log` is the single source of truth: append-only, enforced by SQLite triggers that
reject `UPDATE` and `DELETE`, with each row chained to its predecessor by
`sha256(prev_hash ‖ seq ‖ ts ‖ payload_hash)`. Every other table is a projection rebuildable
from it — which is also the strongest test in the suite.

A hash chain only proves anything against someone who cannot rewrite the file, and the
process writing it can. So the chain head is periodically **anchored across a trust
boundary** (a git commit, an object store with versioning). Without that, the chain is
integrity theatre.

The lineage `decisions → risk_verdicts → order_intents → fills` is complete, so for any fill
the ledger alone answers: which strategy version, from which spec, proposed by which search
run, on which feature snapshot, cleared by which gate evaluation, allowed by which risk rules
with which observed values against which limits, filled at what price with which fees.

```bash
tb replay --fill <fill_id>     # reconstruct the entire decision from the ledger
```

Every broker response is archived raw as well, including failures. For a beta
API whose documentation is not reliably reachable, that archive is the
difference between diagnosing a shape change and guessing at one:

```bash
tb broker drift                # responses that stopped parsing
tb broker replay <msg_id>      # the exact body that arrived
```

## Status

Built in milestones, ordered so the project-killing unknowns die first — *can I trust the
data*, then *can I reconcile broker state*, then *is there any edge after costs*.

| | Milestone | State |
|---|---|---|
| M0 | Ledger, hard limits, kill switch, run state | **done** |
| M1 | Broker adapter (read-only), rate governor, reconciler, symbol map | **done** |
| M2 | Point-in-time data layer, provider bake-off | **done** |
| M3 | Cost model, non-cheating backtester, strategy DSL | |
| M4 | Risk engine, live loop, crash drills | |
| M5 | Registry, promotion gate, capital allocator | |
| M6 | Self-strategising search (LLM optional) | |
| M7 | ML signal layer | |
| M8 | Live at floor size, dashboard, daily journal | |
| M9 | RL — interface stub only, deferred deliberately | |

## Setup

```bash
uv sync --all-extras
cp .env.example .env          # then fill in your keys
tb init                       # creates the ledger and writes the genesis event
tb doctor                     # checks limits, permissions, secrets hygiene
tb status
```

Then characterise the API against your **demo** account. Everything in the
endpoint table and the response models is a reconstruction — the official
reference was not reachable when this was built — so the probe is how those
assumptions become observations:

```bash
tb broker probe               # ~150s: shapes, rate limits, auth header format
tb broker limits              # configured vs observed, and what it implies
tb symbols audit              # Trading 212 tickers -> market-data symbols
tb reconcile                  # what the account actually holds
```

`tb broker probe` refuses to run against a real-money account. It takes a
couple of minutes because the rate governor assumes its budget is spent on a
cold start, so the first call to a one-per-fifty-seconds endpoint waits a full
period — it prints progress so you can tell it apart from a hang.

Nothing in M1 can place or cancel an order, and that is enforced rather than
trusted: the client rejects any endpoint outside its read-only set.

Then build the dataset. These run in this order because each needs the one
before it:

```bash
tb universe build             # pick the symbols, record a dated snapshot
tb data backfill              # fetch history (Yahoo for daily depth)
tb data backfill --provider alpaca --resolution minute
tb data actions               # splits and dividends as dated facts
tb data audit                 # exits 1 if anything blocks
tb data bakeoff               # the paid-data decision, as arithmetic
tb data seal                  # a vintage_id M3 can cite
```

`--symbols` takes Trading 212 tickers and resolves them through the symbol map,
which is the path the trading loop uses. `--data-symbols` takes provider symbols
directly and skips the map, for research only — the bake-off measures *data
providers*, not the broker, so requiring broker credentials to run it was a
coupling that made the measurement impossible without them:

```bash
tb data backfill --provider yahoo  --resolution minute --data-symbols AAPL,MSFT,SPY
tb data backfill --provider alpaca --resolution minute --data-symbols AAPL,MSFT,SPY
tb data bakeoff  --resolution minute
```

Bars fetched that way are keyed `sym:TICKER` instead of by ISIN, which is
self-marking on purpose: the trading path resolves to an `isin:` or `t212:` uid
and never matches one, and `tb data audit` reports them so a vintage containing
research fixtures cannot be mistaken for point-in-time evidence.

There is also a **`Data bake-off` workflow** (`.github/workflows/data-bakeoff.yml`,
manual dispatch) that runs the whole sequence on a GitHub runner using
`ALPACA_DATA_KEY_ID` / `ALPACA_DATA_SECRET_KEY` from repository secrets, prints
the verdict to the run summary and uploads the store. It exists because a
development sandbox may have no egress to either provider, and because a
measurement worth citing should be reproducible rather than something someone
ran once by hand. It holds **no** Trading 212 credentials, so nothing in it can
reach an order endpoint.

Three things about the data layer that are load-bearing rather than
decorative:

**Every bar carries three time axes.** Event time, *knowledge* time
(`bar_close + provider_delay + ingest`, stored rather than recomputed), and
vintage time. The staleness bound is evaluated against knowledge time, never
bar time — a fifteen-minute-delayed feed produces bars whose timestamps look
current, and measuring the wrong axis is exactly what makes such a feed appear
usable.

**A backtest whose `vintage_id` is not in the ledger is not admissible
evidence.** Yahoo back-adjusts history as a matter of course, so re-running the
same backtest against "the store" next month legitimately produces different
numbers with nothing recording why. `tb data seal` freezes a named, hashed set
of files — plus the action, FX, calendar and universe hashes — and loading one
re-hashes every file and refuses on a mismatch.

**`tb data bakeoff` answers "is this the bar the market saw", not "did the bar
arrive in time".** Alpaca's free tier is IEX-only, ~2% of consolidated volume,
so its disagreement with the tape sits in the same 5-20bps band as the whole
gross edge. The output is one number — the gross edge a strategy would need
before the feed's own error is small enough to trade through — set against the
~30bps a round trip already costs. Until that measurement passes,
`data.allowed_live_resolutions` stays `[daily]`.

Alpaca needs two keys, read from `ALPACA_DATA_KEY_ID` /
`ALPACA_DATA_SECRET_KEY`. Generate them at
[alpaca.markets](https://alpaca.markets/data) — sign up, select the **paper**
account, then *API Keys* → *Generate New Keys*; the secret is shown once.

Generate them on a paper account specifically. Alpaca issues keys **per
account, not per scope**, so there is no data-only key to ask for — a key made
on a funded live account can place orders on it. Paper keys serve this same
market-data API on the free plan and cannot move real money, which is the
property the `ALPACA_DATA_*` naming is reaching for but cannot itself enforce.
The SDK-standard `APCA_API_KEY_ID` is deliberately *not* read as a fallback for
the same reason, and a key found under that name is reported by `tb doctor`
rather than used. The free Basic plan is IEX-only, which is what `tb data
bakeoff` exists to measure.

Trading 212 issues a **separate API key per environment**, and the app must be switched to
Practice mode *before* you generate the demo key or you will get a live one. The two keys
are read from deliberately different variables — `T212_DEMO_API_KEY` and
`T212_LIVE_API_KEY` — and the base URL is derived from which one is present. There is no
mode flag, because a single key plus a flag is one typo away from real money. Research and
backtest processes are asserted to run *without* the live key in their environment at all.

## Risk and honest limitations

- **The fee schedule beats most intraday ideas before they start.** See constraint 2 above.
  The system is built to tell you this honestly via the cost gate rather than hide it in an
  optimistic backtest.
- **Both free data feeds are weak for true intraday.** yfinance serves roughly 30 days of
  1-minute history and silently back-adjusts it, which breaks reproducibility and is itself
  a lookahead channel; Alpaca's free tier is IEX only, a few percent of consolidated volume.
  `tb data bakeoff` exists to make the "should I pay for data" decision on evidence.
- **Promotion to live capital is autonomous.** A strategy clearing the gate goes live at
  floor notional with no human approval. The gate is therefore the only thing between a
  noise strategy and real money, which is why it carries a sealed holdout, deflated-Sharpe
  accounting over the full trial count including rejections, and a release-gating test that
  asserts a population of randomly generated strategies gets promoted at approximately zero
  rate. Set `promotion.paper_shadow_sessions` above zero to reinstate a demo period first.
- **The unprotected window is real.** With no bracket orders, an entry fill always precedes
  its protective stop. Position size is capped so a gap across that window is survivable
  inside the daily loss budget; the config validator refuses limits where it is not.
- **Backtest results are not predictions**, and a strategy that cleared a gate is a strategy
  that cleared a gate.
