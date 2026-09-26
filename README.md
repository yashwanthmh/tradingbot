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
with which observed values against which limits, filled at what price with which fees. For a
strategy that reads a model, it also answers which artifact scored it, and replay re-scores
the fill with that model — loaded through the store's hash checks — rather than quoting the
recorded number.

```bash
tb replay --fill <fill_id>     # reconstruct the entire decision from the ledger, and check it
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
| M7 | ML signal layer — walk-forward, shuffled-label null, hash-pinned models | **done** |
| M8 | Session record, journal, alerts, backups, drills, live arming, dashboard | **done** — the evidence it asks for is [the operator's to collect](#what-remains-is-the-operators) |
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

Four things about the data layer that are load-bearing rather than
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

**One bar per session, and every split applied from the scale that bar is
already on.** A symbol held from both feeds is read from Alpaca wherever it has
a bar and from Yahoo only where it does not. Yahoo has already divided its
history by every split up to the day it was fetched, so its bars are adjusted
only for splits after that; a raw bar is adjusted for every split after its own
session. The loop, the regime gate, the backtester, the search and the holdout
evaluation all read the recorded actions — so run `tb data actions` before
`tb data seal`, or a split reads as a 75% fall in every feature across it and a
backtest holding through one books the old share count at the new price. A
vintage's backtests use the actions recorded when it was sealed, so a later
backfill cannot change a re-run.

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
is funded at floor notional with no human approval — on paper and demo. Real
money waits for a person as well (see [Arming real money](#arming-real-money)).
Four things carry that weight:

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

`--mode paper` is the simulated venue, marked to the newest close the loop can
see: a fill is at the price the order was sized from, equity moves with the
market so the loss breakers are live, and a stop fires — at the market, gap
included — once the price reaches it. The account lives in the process, so each
paper run starts flat.

Each cycle opens by settling what finished since the last. Every order that has
left the venue's open list is read from order history — one rationed call, and
none when nothing has finished — its fill recorded at the venue's price and its
intent resolved. Every closing fill is then charged to the strategy whose
decision placed it, at average cost from the recorded fills: the realised record
that `tb review`, the allocator and the lineage loss budgets read. A result that
depends on a fill with no reported price is recorded and charged to nobody.

And the first cycle of each trading session reviews the portfolio before it
decides anything: every promoted strategy gets a KEEP / KILL / ITERATE / SCALE
verdict on its realised record, its rung moves on the evidence of that rung —
up only on the ladder's slow terms, down two at once on a KILL or ITERATE — and
the allocator shrinks each prior toward its realised edge. The loop then trades
a book rebuilt from all of it, so a run left going for weeks sizes by today's
evidence. Once a session, however often `tb run` is started; retiring a
strategy stays a human's `tb review --apply`.

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

Where a model comes in — the ML signal layer:

```bash
tb ml train <vint> --feature return_pct:5 --feature zscore:20   # dry run: train, record, count
tb ml train <vint> --feature ... --apply                        # register the surviving specs
tb ml models                                                    # every recorded model
tb ml show <mdl_id>                                             # its record, and a verified load
tb ml verify                                                    # every artifact against the ledger
tb ml calibrate                                                 # the shuffled-label release gate
```

A model is not a strategy. It is a scorer inside the one feature pipeline, and
its score is one more term a spec can read — `{"kind": "model", "model_id",
"artifact_sha256"}` beside features and constants — so the registry, the trials,
the holdout, the gate, the allocator, the loop and replay treat a strategy that
reads a model exactly as they treat any other, and none of them needed a second
code path to do so. The interpreter never runs a model; it reads the score from
the snapshot like a moving average, hashed with everything else.

**Trained on rows the loop would have produced.** Every sample's features come
from the pipeline's own `compute` on the forward-only reader at the decision
time, and every label is the trade a strategy acting on it would have made: in
at the next bar's open, out at the open `horizon` bars later, net of the cost
model's round trip, following any split in between. Through the sealed reader,
a label whose exit falls in the holdout never completes — those prices are never
in memory — and a model whose labels reached the seal is refused at record time.

**Scored walk-forward, purged and embargoed.** A label decided today is known a
week later, so ordinary cross-validation trains on labels computed from the
prices it is scored on. Each fold here is scored by a model fitted only on
samples decided before it *and known* an embargo before it, with a decision
time never split across a boundary; the properties are tested over random spans
rather than hand-picked ones. The specs a training run proposes — thresholds on
the model's out-of-sample scores, three by default, each a trial — are pinned to
the final model but backtested with the fold models, because the final model
has seen every label in the window and a backtest of it there would be
in-sample. Its own out-of-sample test is the sealed holdout, spent once, which
refuses a model whose training labels reach into the window it would be scored
on.

**The null runs on every fit.** The same folds on shuffled labels must show no
skill: an AUC within four standard errors of 0.5, the error computed for the
sample at hand. Skill there is the evaluation's own — a fold scored on rows it
was fitted on is the classic cause — so a failing null records nothing at all:
no model, no trial. `tb ml calibrate` is the same check as a release gate, run
through the real reader, pipeline and labels on synthetic bars, beside a planted
pattern the real labels must find; a null that passes because the model can
find nothing is not a null.

**A model exists when the ledger says so.** Artifacts are LightGBM's text
format, parsed on load and never unpickled, trained deterministically so the
same rows give the same bytes, and stored under their sha256. The
`model.recorded` event, not the file, admits one: a file with no event is never
loaded, and before any artifact is parsed its recording event must hash to its
stored values and name the artifact the catalog row names, and the file must
hash to it. An edited, deleted, planted or re-pointed model is refused by name,
and a spec pinned to one hash is never handed another under the same id — a
retrained model does not inherit a promotion it never earned.

**Retraining is one growing search.** A training run's specs are filed under
the idea's lineage — its features and label, whatever the parameters — so every
retrain of one idea adds to one lineage's trial count, and "retrain until it
passes" is deflated as the search it is. The searcher, for its part, may never
propose a spec that reads a model: a model's trials are counted where it was
trained, and a search composing models would stack a second, uncounted search
on top of them. LightGBM is the optional `ml` extra (`uv sync --extra ml`),
imported only where a model is fitted or parsed; strategies that read no model
trade without it.

## Operating it

Everything above decides *what* trades. This is how a person runs it from day to
day, and what the ledger must show before any of it touches real money. It all
reads the one ledger; nothing here keeps a second record of its own.

A trading day is four processes side by side, each able to stop without taking
the others with it:

```bash
tb watchdog                           # the other half of the dead-man switch
tb run --mode demo                    # the loop, on the demo account
tb alerts --follow --sink webhook     # what a person must act on, delivered once
tb dashboard                          # the same record, in a browser
```

and after the close:

```bash
tb sessions                           # was today clean, and how long is the streak
tb journal write --commit             # the day's page, committed with the chain head
tb backup create                      # a verified copy of everything the ledger names
```

**A session is clean or it is not, and the ledger decides.** `tb sessions`
judges each trading day once it is over, per mode: *clean* when the loop
covered at least 90% of the regular session and nothing halted, crashed,
drifted or left a position unprotected past its bound; *faulted* when something
did; *incomplete* when the loop simply was not there enough. The live gate
counts the streak with the same function from the same ledger, so the number an
operator reads each morning is the number `tb arm` will count. A day with a
passed drill is judged *drill*: neither clean nor a break.

**The journal is the ledger written down, and checkable.** One markdown page per
session under `journal/`: what ran, the orders, fills and stops, the trades
closed, the safety events and the account. Each page names the sequence number
and chain hash it was written from, so `tb journal verify` regenerates it byte
for byte and names a page edited after the fact — or a ledger rewritten beneath
one. `--commit` puts the pages and the chain head in git, which is the anchor
across a trust boundary that a hash chain needs.

**Alerts run out of process, read-only.** A notifier inside the trader would go
quiet exactly when the trader wedged. `tb alerts --follow` reads from a saved
cursor and delivers at least once (a webhook that is down gets the batch again
next pass), counts a repeat instead of re-sending it, and on its first run
starts from now: a year of history replayed would teach anyone to mute the
channel. The webhook URL, usually a credential itself, is read from
`TB_ALERT_WEBHOOK_URL` only and never printed.

**The dashboard can stop trading and can do nothing else.** `tb dashboard`
(the `api` extra) shows the status panel, the equity curve with the three
breaker readings, the risk budget read the way each breaker reads it,
per-strategy attribution within one account, the session record, a live event
tail and the journal. Every view opens the ledger read-only. The one write is a
kill switch button, which throws the switch and records a manual halt exactly
as `tb halt` does; there is no release, so resuming stays `tb resume` at a
terminal, with a reason. It listens on loopback. Setting `TB_DASHBOARD_TOKEN`
(24 characters or more) turns the button on and makes every request need the
token; without one the button is off, only loopback is served, and `tb
dashboard` refuses to listen anywhere else. To watch from another machine,
tunnel rather than bind: `ssh -L 8765:127.0.0.1:8765 this-host`. Ledger text —
a ticker, a model's words on a proposal — reaches the page as text, never as
markup, under a policy that runs no script but the page's own.

**A backup is evidence once it has been restored somewhere else.**

```bash
tb backup create                          # ledger, bar files, models and limits, hashed
tb backup verify var/backups/bkp_...      # every file, the chain, and the catalogues
# copy the directory to another machine, and there:
tb backup restore bkp_... --to /srv/tb    # refuses a non-empty target; replays recent fills
# then, back where the backup was made:
tb backup receipt restore-receipt.json    # the proof, recorded where the live gate reads it
```

A restore on the machine that made the backup proves the files are intact, not
that the state survives losing that machine, so the gate does not count one.
Machines are told apart by host name, so give the second one a name of its
own: two machines sharing a name count as one. The name in a receipt is the
restoring machine's own word, so this catches a mistake, not someone set on
fooling the gate. CI does the whole round trip across two runners on every
push (`backup` → `restore-elsewhere` → `receipt-home`); GitHub's runners all
share one host name, so the second is renamed first, as yours would need to be.

**Drills fire the real mechanisms at a real loop.** Each runs against a demo
loop that holds the lease, in market hours, with a position held and every
position covered by a broker-side stop. A kill switch tested on a quiet evening
proves that a file is read; one tested with a position open in a moving market
proves that stopping the bot leaves what it holds protected.

```bash
tb drill killswitch     # engage the switch: the loop must halt, send nothing, keep every stop
tb drill watchdog       # freeze the loop: the watchdog must notice and engage the switch
tb drill list           # every drill, and whether it passed
```

Neither restarts trading. Each ends with the switch engaged and the loop
stopped, and resuming is a person's call.

### Arming real money

Three things stand between this system and a real trade, and no one of them is
enough on its own:

1. **The limits file enables it.** `live.enabled` is `false` as shipped. The
   file is read-only to the bot and hash-pinned, so turning it on is a person's
   edit that every run records.
2. **A person arms it, against evidence the ledger holds.** `tb arm` prints each
   requirement, observed against required, and exits 1 while any is unmet:
   thirty clean demo sessions in a row with at least five trades closed across
   them; a passed kill-switch drill and a passed watchdog drill, each on demo, in
   market hours, with a position held, within 30 days; a restore verified on
   another machine within 90 days; a chain that verifies. `tb arm --live
   --strategy S` says what will happen and arms only on the typed phrase
   `arm live`. An arming names at most one promoted strategy, holds it to rung 0
   — floor notional — lapses after seven days, and is bound to the limits hash
   in force: change the limits and it no longer holds. `tb disarm --reason`
   ends it at any time.
3. **The live key, and only it.** `tb run --mode live` refuses without a
   current arming under exactly these limits, with `--strategy` or
   `--no-watchdog`, or with the demo key. While it runs it re-checks the arming
   every cycle, so a lapse, a disarm or a change of limits halts it at the next
   cycle, with its stops left at the broker.

The thresholds are the `live:` section of `config/hard_limits.yaml`, beside
every other number that gates money.

### What remains is the operator's

The M8 software is built and each property above is tested; what the plan's M8
verification asks for is evidence that only a real demo account can produce. In
order:

1. Run the loop on demo — `tb watchdog`, `tb run --mode demo`, `tb alerts
   --follow` — until `tb sessions` shows **thirty clean sessions in a row** with
   at least five trades closed.
2. In market hours, with a position open: **`tb drill killswitch`**, then
   `tb resume`; **`tb drill watchdog`**, then `tb resume`.
3. **Restore a backup on another machine** and bring the receipt home.
4. `tb arm`: every row met except the limits file.
5. Review, and set `live.enabled: true` in `config/hard_limits.yaml` yourself.
6. `tb arm --live --strategy <one gate-cleared strategy>`, then `tb run --mode
   live` with `T212_LIVE_API_KEY` alone in the environment.

## Risk and honest limitations

- **The fee schedule beats most intraday ideas before they start.** See constraint 2 above.
  The system is built to tell you this honestly via the cost gate rather than hide it in an
  optimistic backtest.
- **Both free data feeds are weak for true intraday.** yfinance serves roughly 30 days of
  1-minute history and silently back-adjusts it, which breaks reproducibility and is itself
  a lookahead channel; Alpaca's free tier is IEX only, a few percent of consolidated volume.
  `tb data bakeoff` exists to make the "should I pay for data" decision on evidence.
- **Promotion is autonomous; real money is not.** A strategy clearing the gate is funded at
  floor notional on the paper and demo books with no human approval, so the gate is the only
  thing between a noise strategy and the demo account's capital. That is why it carries a
  sealed holdout, deflated-Sharpe accounting over the full trial count including rejections,
  and a release-gating test that asserts a population of randomly generated strategies gets
  promoted at approximately zero rate. Real money needs more: the limits file's
  `live.enabled`, thirty clean demo sessions, both drills, a restore on another machine, and
  a person's `tb arm --live` naming the strategy, capped at floor size and lapsing weekly.
  Set `promotion.paper_shadow_sessions` above zero to reinstate a demo period before a
  promotion funds anything at all.
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
- **A model finds patterns, including the ones that are not there.** Walk-forward scoring,
  purging, the embargo and the per-fit null remove the ways an evaluation manufactures skill;
  none of them makes a real pattern persist. On free daily data the expected result of
  `tb ml train` is a model that shows no out-of-sample skill and specs the edge band refuses,
  which the trainer reports and counts rather than hides.
- **Backtest results are not predictions**, and a strategy that cleared a gate is a strategy
  that cleared a gate.
