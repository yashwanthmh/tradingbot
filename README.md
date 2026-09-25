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
| M3 | Cost model, non-cheating backtester, strategy DSL | **done** |
| M4 | Risk engine, live loop, crash drills | **done** |
| M5 | Registry, promotion gate, capital allocator | **done** |
| M6 | Self-strategising search (LLM optional) | **done** |
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

Then check that the backtester can be trusted, before trusting anything it
says:

```bash
tb backtest costs             # the venue arithmetic, per jurisdiction
tb backtest calibrate         # exits 1 if a null strategy earned an edge
tb backtest calibrations      # what the engine has been judged on
```

`tb backtest costs` prints the number the whole project turns on. From a GBP
account, per 1,000 of notional:

| instrument | round trip | gross edge needed |
|---|---|---|
| US large-cap | 40bps | **121bps** |
| UK share | 60bps | **182bps** |
| Irish share | 140bps | **424bps** |

against 5-20bps of gross edge on minute-bar signals in liquid names. Minute-by-
minute trading needs six to twenty times more edge than exists, which is why
the design target is minute-resolution *features* with hour-to-day position
changes — and why the cost gate is a pre-trade rejection rather than a report.
Irish-incorporated names are not tradable by anything in this system's class.

`tb backtest calibrate` is a release gate rather than a diagnostic. It runs
strategies with no edge by construction — always-flat, always-long,
alternating, and eight seeded coin flips — over the store and fails if any of
them earned one. A coin flip with a positive net Sharpe is a fill-timing error,
a mark taken from a bar the position could not see, or a cost charged on one
leg. It is never a discovery.

The assertion is two-sided, and the second half is the part usually left out:
the population's cost drag must also be materially non-zero, because a run
where nothing was charged passes the Sharpe test trivially while proving
nothing. That check earned its place immediately — it caught a real bug in the
first version of the engine, which compared bar time instead of knowledge time
when picking a fill bar and therefore filled nothing at all.

Three things about the backtester that are load-bearing:

**A fill never uses a price the decision could see.** An order decided on bar
`t` fills at the open of bar `t+1`, and the engine obtains that price by
advancing the reader — so it is structurally unavailable to the decision that
caused it. Filling at the decision bar's close is the most common way a
backtest manufactures returns, worth roughly the entire gross edge at minute
resolution. A decision on the last bar has no next bar, so it is dropped and
counted rather than filled at a price it saw.

**There is exactly one feature pipeline.** The same object, taking the same
`BarWindow` type, in backtest, paper and live. A test asserts both call paths
produce identical snapshot hashes, because two implementations that agree today
drift — and the drift surfaces as live underperforming its backtest months
after it was introduced.

**A strategy that will not declare its expected edge cannot trade.** The cost
gate divides by that declared number, so `costs.max_expected_edge_bps` bounds
it: without a ceiling, a spec claiming 10,000bps would pass the gate trivially
and the one control keeping the search loop out of the fee trap would be
defeatable by the search loop.

Strategy specs are validated data, never code. A test walks the AST of
`src/tb/strategy/` asserting no `eval`, `exec`, `compile` or dangerous import
exists anywhere in the interpretation path — features are selected from a fixed
library table by name, which is what makes interpreting generated specs safe.

Then the pipeline that decides whether a strategy gets money:

```bash
tb registry register spec.json   # a candidate, and nothing more
tb research holdout <id> <vint>  # spend its one out-of-sample evaluation
tb promote evaluate <id>         # every gate: pass/fail, observed, threshold
tb allocator explain             # the prior/realised shrinkage behind each size
tb registry review               # KEEP / KILL / ITERATE / SCALE on the live book
tb research null-gate            # re-measure the false-promotion rate
```

`paper_shadow_sessions` is 0, so a strategy that clears `tb promote evaluate`
is funded at floor notional with no human approval. Four things carry that
weight:

**The gate runs every check and does not short-circuit.** "Refused by one check
at 99% of its threshold" and "refused by six" call for opposite responses from
a search loop, and a gate that stopped at the first could not tell them apart.
An unmeasurable input — a deflated probability on too short a sample, PBO on
too few trials, a feed-noise figure with no bake-off — **refuses**, because
otherwise a candidate clears the gate by arranging for a computation to fail,
which is easier than clearing it on merit.

**The holdout is enforced by the data layer and spent once.** A research
process gets a `BarSource` holding no bar past the boundary and a reader that
raises rather than returning a short window, and `UNIQUE(strategy_id, version)`
makes a second evaluation impossible rather than discouraged. A holdout that
can be re-evaluated is a slower training set: "failed, tweak, resubmit" fits it
one bit per attempt.

**Multiplicity is counted over the whole search, not the lineage.** Deflated
Sharpe divides out the best result a search of N trials would produce from
noise, so N has to be honest. Every trial is logged including rejections and
errors, and the count is stamped when the trial happens — a searcher that gave
each candidate its own lineage would otherwise face no haircut at all.

**The release gate is a number, not a claim.** `tb research null-gate` — and a
test that runs the same function over 1,000 specs — draws random specs from the
real grammar, backtests them on a random walk with zero drift, and counts how
many the gate promotes. The result is 0 of 1,000 against a 1% ceiling, with 477
candidates reaching the statistical checks with measured numbers. That second
figure matters as much as the first: a gate that refused everything because
nothing traded would meet any ceiling while proving nothing, so the suite
asserts both, and separately that a genuinely good strategy still promotes.

And then the part that makes all of it true of the account rather than only of
the database — `tb run` trades the **promoted book**:

```bash
tb run --mode paper                   # every promoted strategy, at its funded size
tb run --mode paper --strategy trivial  # the hand-written one, for drilling the loop
```

The loop reads `registry.promoted()`, loads each spec, builds **each strategy's
own pipeline from its own spec**, and sizes every order at
`min(allocation, rung notional)` — which reaches the order path as a risk
verdict row like every other cap, so "why is this position small" is answerable
from the same place as "why was this order refused". A missing allocation on a
promoted strategy blocks rather than sizing by nothing: on the order path an
absent number is a wiring error, not an unlimited budget.

Three consequences worth stating, because each is a thing that can only be got
wrong once:

**Nothing promoted is a refusal to start, not an idle loop.** A run that
started with an empty book would write a run record, a heartbeat and a cycle
saying nothing traded — which is exactly what a correct system looks like on a
quiet day.

**A position belongs to the strategy whose entry opened it**, resolved by
joining `order_intents` to `decisions`, because the broker reports a position
per instrument and knows nothing about strategies. Only the owner is asked about
it, so two strategies cannot take turns deciding one holding, and an add is
charged against the allocation that paid for the rest of it. Two strategies
wanting the same *flat* instrument is resolved by book order — which is sorted
by identity, so a replay resolves it the same way the live run did.

**A position no funded strategy owns is flattened.** Retiring a strategy while
it holds is the searcher's normal outcome, and nothing in the book would ever
produce an exit for what it left behind. An open position with no bracket order
behind it and nothing managing it is the state this whole design exists to
avoid, so the loop closes it itself and records which of the two causes it was:
an owner that is no longer funded, or an entry that cannot be attributed at all.

Where candidates come from — the search:

```bash
tb research cycle <vint>                        # dry run: search, record every trial
tb research cycle <vint> --out specs.jsonl      # ...and write every spec it produced
tb research cycle <vint> --apply                # register the survivors as candidates
tb research cycle <vint> --from <id> --apply    # refine a registered strategy
tb research cycle <vint> --proposer llm         # first generation from a Claude model
```

A cycle draws specs from the grammar (or mutates registered ones), refuses what
cannot be evaluated or afforded before spending a backtest on it, backtests the
rest on the training window only, and breeds the next generation from the best
— one per feature signature, so a generation cannot collapse into twelve
variants of one lookback. It registers candidates and nothing more: the holdout
and the gate are separate commands, because a process that both chose a
strategy and checked the choice would be marking its own homework.

**The budget is the design, not a limit.** The report states the out-of-sample
Sharpe a search of that size must show to clear the deflated-Sharpe gate —
about 2.1 at ten trials, 3.8 at a thousand — so a thousand-spec sweep has
already failed before it runs. Many small searches are cheaper than one large
one, and they are not an evasion: a refinement stays in its parent's lineage,
whose trial count accumulates across every search that touches it.

**Every trial is recorded, dry runs included.** Refused, broken and evaluated
alike, each stamped with the size of the whole batch it was selected from. A
dry run that recorded nothing would let ten runs and a hand-registered winner
pass as a search of one.

**The optional model is told nothing it could date.** A model that has
memorised market history is a lookahead channel no schema can catch — shown a
date and a price, it knows what happened next. So `--proposer llm` builds its
prompt from two abstract inputs only: the feature dictionary with units, and
whether the index is above or below its long average, read through the sealed
source. The types admit nothing else; a tripwire refuses any rendered prompt
holding a date, a price, an instrument or a credential; and a test seals two
vintages more than a decade apart, in different instruments at prices an order
of magnitude apart, and asserts their prompts are byte-identical. What comes
back is untrusted data: parsed as JSON with every number a `Decimal`, validated
by the same schema as everything else, allowed to decide only the entry, the
exit and the declared edge, and never executed. It meets the same validator,
the same backtest and the same selection as a random draw — the model changes
where a search starts and nothing about how its candidates are judged. A
refusal or a failed call stops the cycle before anything is evaluated, rather
than quietly becoming a random search under the model's name; and since a
model cannot be replayed from a seed, every exchange is ledgered with its exact
prompts and the full text of every spec it produced. Refusal fallbacks are on
by default (`--no-fallback` turns them off), and the command will not run in a
process holding `T212_LIVE_API_KEY`. It needs the optional extra
(`uv sync --extra llm`) and `ANTHROPIC_API_KEY`; nothing else in the system
does.

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
- **The size of a search decides what edge it can prove.** The multiplicity haircut is the
  best Sharpe a search of N trials would produce from noise alone, so it grows with N: at a
  trial dispersion of 1.0 it is about 1.6 Sharpe for a search of ten and about 3.3 for a
  search of a thousand. An out-of-sample Sharpe of 3.0 — an excellent real result — is
  therefore promotable out of a focused search and is not out of a thousand-spec sweep. That
  is the arithmetic working rather than a threshold to loosen, and its consequence for M6 is
  concrete: many small searches, with the trial count spent as a budget. See
  `docs/decisions/0002-search-size-and-provable-edge.md`.
- **Backtest results are not predictions**, and a strategy that cleared a gate is a strategy
  that cleared a gate.
