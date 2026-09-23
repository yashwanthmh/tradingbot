# 0001 — Minute resolution stays out of `allowed_live_resolutions`

**Status:** accepted
**Date:** 2026-09-23
**Applies to:** `data.allowed_live_resolutions` in `config/hard_limits.yaml`

## The question

The plan made minute-resolution live trading contingent on a measurement:

> `tb data bakeoff` — per symbol per minute: Alpaca-vs-Yahoo close disagreement
> (median/p95/p99 bps), fraction of minutes with no IEX print, observed delay
> distribution per provider, and the fraction of decision cycles where no
> provider met the 180s bound. **This is the paid-data go/no-go, stated as
> arithmetic.**

Until this record, `allowed_live_resolutions: [daily]` was a *conservative
default* — the right answer for the wrong reason, which is a thing that gets
quietly reverted by whoever next reads it and sees no evidence attached.

## What was measured

One real Alpaca IEX pull, from a GitHub runner on 2026-09-16, over ten of the
most liquid US names (AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, UNH, SPY)
across roughly 18 sessions. Sealed as vintage `vint_2026-09-16_ddd0fb703a46`
(10 files, 62,724 rows, `point-in-time: vendor_current_view`).

| measurement | value |
|---|---|
| minute bars stored | 62,549 |
| expected minutes with **no IEX print** | 3,751 — about **5.7%** |
| partitions sealed / revisions detected | 10 / 0 |
| Alpaca-vs-Yahoo disagreement (p50/p95/p99) | **not obtained** |

The cross-feed figure is missing for an environmental reason, not an
analytical one: Yahoo's chart endpoint rate-limits by IP and refused all ten
requests from the runner on first contact, twice. That is not a pacing budget
this repo can tune.

## The decision, and the arithmetic behind it

**Minute resolution is refused.** Two independent numbers each refuse it, so
the missing third would only add precision.

**1. One minute in eighteen has no price at all.** 5.7% of expected minutes
had no IEX print, on the ten most liquid names in the US market. The free
Alpaca tier is IEX-only — a few percent of consolidated volume — so a minute
bar from it is not the minute the market saw: its close is not the
consolidated last price and its high/low are not the session extremes. On a
thinner name this is worse, and the universe cap is 25 symbols, not 10.

A strategy deciding on a minute bar that does not exist is deciding on the
previous print, at an unknown and varying age. That is the lookahead-adjacent
failure the three time axes exist to make visible, and here it would be
happening 5.7% of the time by construction.

**2. The cost gate needs 121bps and the resolution offers 5-20.** Straight
from `tb backtest costs`, on a 1,000 GBP round trip:

| instrument | FX | tax | spread | slip | round trip | edge needed |
|---|---|---|---|---|---|---|
| US large-cap | 1.50 | 0.00 | 0.20 | 0.30 | **40.0bps** | **121bps** |
| UK share | 0.00 | 5.00 | 0.20 | 0.30 | 60.0bps | 182bps |
| Irish share | 1.50 | 10.00 | 0.20 | 0.30 | 140.0bps | 424bps |

"Edge needed" is `round trip / 0.33`, the `execution.max_cost_to_edge_ratio`.
Gross minute-bar edge on liquid names is 5-20bps.

So a minute strategy on the *cheapest* instrument available would have to be
six to twenty-four times better than the literature to clear the gate it must
pass — before the 5.7% of missing bars is considered at all. On an Irish
issuer it needs 424bps.

(The cost is mildly notional-dependent: the same US round trip came out at
36bps at the M4 loop's ~£150 order size, which is where that figure in the
commit history comes from. 40bps is the published table's number and the one
to quote.)

Neither number is close. The conclusion does not depend on the disagreement
figure.

## What would reopen this

Exactly one thing: a `free_data_sufficient` verdict from

```
tb data bakeoff --resolution minute --primary alpaca --secondary yahoo
```

on a run where **both** feeds returned bars. That requires an IP Yahoo will
serve — in practice a residential connection rather than a cloud runner — or
a second feed obtained some other way.

The gate is enforced in code, not by this document.
`Verdict.permits_live_resolution` is true only for `FREE_DATA_SUFFICIENT`, and
`may_widen_live_resolutions` refuses `NO_SECOND_FEED` and
`INSUFFICIENT_SAMPLE` explicitly. A measurement that did not happen cannot
license the edit, and `tb data bakeoff` now exits non-zero on any verdict but
a pass — it previously printed the verdict and exited 0 whatever it said.

## What this does not claim

- **It is not a claim that paid data would fix it.** Consolidated data removes
  the 5.7%; it does not move the cost arithmetic, which is the harder of the
  two constraints. A minute strategy on paid data still needs 121bps on the
  cheapest instrument there is.
- **It is not a claim about the disagreement figure.** We have no evidence on
  it. The honest state is "unmeasured", which is why the bake-off reports
  `no_second_feed` rather than a verdict.
- **It says nothing about daily.** Daily closes are consolidated and
  revision-stable, and the cost arithmetic is survivable at a holding period
  measured in weeks — which is why the M4 strategy declares 250bps and clears
  the gate at a ratio of 0.14.

## Consequences

- `allowed_live_resolutions` stays `[daily]`, now with the evidence recorded
  beside it in `config/hard_limits.yaml`.
- The design target the plan set — *minute-resolution features, hour-to-day
  position changes* — is confirmed rather than revised. Features may be
  computed on minute bars; positions may not change on them.
- M5's promotion gate should refuse a spec whose implied holding period is
  shorter than the resolution permits, rather than relying on the cost gate to
  catch it per trade.
