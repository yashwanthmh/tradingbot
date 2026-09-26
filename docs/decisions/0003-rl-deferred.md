# 0003 — Reinforcement learning stays an interface

**Status:** accepted
**Date:** 2026-09-26
**Applies to:** M9, `src/tb/strategy/rl/`, and anything that would train,
store or fund a policy

## The question

The plan asked for all three learning layers behind one `Strategy` interface —
a DSL searched by a deterministic searcher, an ML signal layer, and RL — and
then, for the third, recommended against itself:

> M9 — RL: interface stub only. `strategy/rl/policy.py` conformance stub.
> **Recommend deferring indefinitely** — RL on ~30 days of minute bars,
> long-only, against a simulator whose fidelity cannot be validated, is a
> research project rather than a feature. The interface stays so it can slot
> in later.

This record makes that recommendation a decision with its reasons attached,
and says what would have to be true before it is reopened. It is written now
because a deferral with no stated conditions gets reversed by the first person
who finds an RL tutorial, and one with vague ones never gets revisited.

## Why not now

Four reasons, each enough on its own.

**1. The evidence would not survive the gate, by the gate's own arithmetic.**
Training a policy selects among a great many candidates on the same data: every
checkpoint evaluated, every hyperparameter setting, every seed. Counted
honestly, as every trial in this system is, that is a search of thousands. The
multiplicity haircut the promotion gate applies (0002) grows with the count, at
a trial-Sharpe dispersion of 1.0:

| trials | E[max SR] from noise | out-of-sample Sharpe needed |
|---|---|---|
| 10 | 1.57 | 2.07 |
| 1,000 | 3.26 | 3.76 |
| 5,000 | 3.69 | 4.19 |
| 50,000 | 4.24 | 4.74 |

(`tb.research.selection.expected_max_sharpe`, with
`promotion.min_oos_deflated_sharpe` at 0.5.) A daily long-only equity strategy
with an out-of-sample Sharpe above 4 is not a thing anyone should expect to
find. So a trained policy either cannot be promoted, or it is promoted by
counting its training as one trial — the second is the failure this system is
built to refuse.

**2. The simulator it would learn against has never been measured.** RL
optimises against its environment, including the environment's mistakes. The
paper broker fills at the price an order was sized from, fires stops at the
market with the gap included, and has no impact, no partial fills and no queue.
Against 5-20bps of gross edge and 40-140bps of round-trip cost, an error of a
few basis points in the simulator is the whole edge, and a policy trained on it
will find that error before it finds anything real. How far paper diverges from
the demo account is not yet known: the thirty demo sessions M8 asks for are the
first measurement there will be.

**3. The action space leaves RL little to add.** Here a decision is flat or long
per instrument, with a 60-minute minimum hold, sized by the allocator, gated on
cost. What RL adds over a classifier is sequential credit assignment — valuing
an action by what it makes possible later — and with two states, long holds
and a fee that dominates the edge, there is little later to value. The case it
would cover, a learned signal turned into entries and exits, is already covered
by M7: a model's score as a term in a spec, walk-forward validated, with a
shuffled-label null on every fit.

**4. The reward is sparse where it matters.** At floor size a strategy makes
10-20 trades a month. Realised P&L on that many trades is the signal the plan
already calls noise for KEEP/KILL decisions, and a policy rewarded on it would
learn from noise faster than the review cycle could tell.

## What exists instead

`tb.strategy.rl` fixes the interface, so that a policy could slot in later
without a second code path, and so that the properties every other strategy
has hold for a policy by construction:

- `Policy` — an id, a version, feature specs from the fixed library, a declared
  expected edge, its own minimum hold, and `act(Observation) -> Target`. Nothing
  else: no reward, no update, no hook the loop could call. A policy is frozen
  while it trades; training, if it ever exists, produces a new version.
- `Observation` — the observed features, read-only, never `UNKNOWN`, and the
  policy's own position. Not the window, whose raw bars no snapshot hash covers,
  and not the account, which would let it size itself.
- `Target` — flat or long. No short and no size.
- `PolicyStrategy` — the adapter onto `Strategy`. It asks the policy only when
  every observed feature has a value; a held position with a missing feature is
  exited without asking, as a spec's is. An entry carries the declared edge, so
  the cost gate divides by a constant the policy cannot move per decision.
- `check_policy` — the conformance check: identity, library features, declared
  edge within `costs.max_expected_edge_bps`, minimum hold, action space,
  exceptions, determinism across repeated and reordered asking, a time budget,
  and a refusal to pass vacuously. Reference policies are held to it in
  `tests/test_rl_policy.py`, one broken policy per rule.

A `PolicyStrategy` can be backtested through the real engine. It cannot be
funded: `funded_book` builds strategies from registered specs only, and
`test_nothing_outside_the_rl_package_imports_it` fails if anything outside
`tb.strategy.rl` imports it.

## What would reopen it

All of these, not some:

1. **A measured simulator.** Fills on paper and on the demo account compared
   over at least thirty clean demo sessions, with the divergence in price and
   timing stated and small against the cost gate's margin. Until then there is
   nothing to train against that is known to resemble the venue.
2. **Data deep enough to hold a holdout.** A feed with years of consolidated
   minute history (the bake-off's paid-data verdict, 0001), or a daily
   formulation over far more instruments than the 25 the rate limits allow.
3. **Honest trial accounting for training.** Every checkpoint, hyperparameter
   setting and seed evaluated counts as a trial in the policy's lineage, and the
   gate's haircut is computed from that count — agreed before the first policy
   is evaluated, not after.
4. **A policy store like the model store.** Content-addressed, parsed rather
   than unpickled (the strategy package forbids `pickle` outright), admitted by
   a ledger event, and pinned by hash in the spec that uses it, so a promoted
   policy cannot be swapped under its promotion.
5. **The same path in as models took.** A policy enters the book as a term in a
   registered spec, so the registry, trials, holdout, gate, allocator, loop and
   replay need no second path — and the import test above is changed on purpose,
   in the commit that does it.
6. **`check_policy` passes** over windows from the sealed training vintage.

## What this does not claim

- **It is not a claim that RL cannot work in markets.** It is a claim about this
  venue, this data, this action space, and a gate that counts trials honestly.
- **It does not forbid research.** A policy can be written, checked and
  backtested today. It cannot trade, and the gate would not believe its
  backtest.
