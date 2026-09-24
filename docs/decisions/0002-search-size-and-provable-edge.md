# 0002 — The size of a search decides what edge it can prove

**Status:** accepted
**Date:** 2026-09-24
**Applies to:** `promotion.min_oos_deflated_sharpe`,
`promotion.min_deflated_sharpe_probability`, and the shape of M6's searcher

## The finding

M5's release gate runs 1,000 randomly generated specs through the real
promotion gate. The result:

```
0 of 1000 random specs promoted (0.00%); 494 traded, 477 reached the
statistical checks with measured numbers; PBO 0.043, haircut from 1000 trials
at a Sharpe dispersion of 0.92
```

Zero promotions against a 1% ceiling. But the *reason* is the finding, and it
is not the one I expected: the deflated Sharpe alone refused all 1,000. It did
not need help from PBO, the trade count or the drawdown bound.

The arithmetic is straightforward once seen. The multiplicity haircut is the
expected maximum Sharpe of N trials (Bailey and López de Prado), which scales
with the cross-sectional dispersion of the trials' Sharpes:

| trials | E[max SR] at dispersion 1.0 | OOS Sharpe needed to clear 0.5 |
|---|---|---|
| 1 | 0.00 | 0.50 |
| 10 | 1.57 | 2.07 |
| 100 | 2.53 | 3.03 |
| 1,000 | 3.26 | 3.76 |
| 10,000 | 3.86 | 4.36 |

The measured dispersion across the null population was 0.92, so a candidate
out of that search needed an out-of-sample Sharpe of about **3.5** to clear the
level, and rather more to clear the 0.95 probability bar on a realistic sample
length. The best holdout Sharpe in the whole population was 1.29.

## What this means, stated plainly

**A thousand-spec sweep over one window cannot promote anything real.** An
out-of-sample Sharpe of 3.0 is an excellent result for a daily long-only equity
strategy. Out of a search of ten it is promotable; out of a search of a
thousand it is below what noise would have produced, so it is refused — and
correctly, because a search of that size genuinely cannot distinguish the two.

That is the technique doing its job. The plan predicted it in words: "Raw OOS
Sharpe over thousands of trials is a rubber stamp." The number above is the
same statement as arithmetic.

## The decision

**The thresholds stay where they are.** Both are in the hash-pinned limits
file, and lowering either to admit candidates out of a large search would be
lowering the one control that makes a large search safe. The honest response to
"nothing promotes" is to run a different search, not to accept weaker evidence.

**M6's searcher spends its trial count as a budget.** Three consequences,
recorded here so they are design inputs rather than discoveries:

1. **Many small searches, not one large one.** Multiplicity is counted over the
   `search_id` as well as the lineage (see `tb.research.trials`), so a searcher
   cannot dodge the haircut by splitting one sweep across a thousand lineages.
   It can, legitimately, run a genuinely separate search per family per window
   — a search of thirty candidates carries a haircut of about 2.0 rather than
   3.3, and that difference is the whole difference between "can promote" and
   "cannot".
2. **A wider window buys more than more candidates.** The probability form
   scales with `sqrt(T - 1)`, so lengthening the holdout raises confidence
   without touching the haircut. On daily bars, two years of holdout is roughly
   the point where a Sharpe gap of 1.5 clears 95%. Ten years of daily history
   is available on free data; a thousand more specs is not worth one more year.
3. **The trial count is worth reporting to the searcher.** A spec proposed as
   trial 900 of a search is being asked to clear a bar that trial 10 did not
   face. A searcher that knows this can stop a sweep that has already made its
   own survivors unpromotable.

**What would change this decision.** A measured trial-Sharpe dispersion well
below 1.0 would shrink every haircut proportionally — the table above scales
linearly in it. A more constrained grammar, or a feature library whose members
are less able to fit noise, would produce that. It is a reason to *narrow* the
DSL rather than to widen it, which is the opposite of the instinct a search
loop creates.

## What this does not claim

- **It is not a claim that the gate is too strict.** A gate that promotes
  nothing from noise and promotes a Sharpe of 4.0 out of a search of forty is
  calibrated, and `test_a_genuine_edge_still_promotes` asserts the second half
  so the first cannot be met by a gate that is simply closed.
- **It is not a claim about PBO.** PBO came out at 0.043 on this fixture, which
  reads as "the in-sample winner held up out of sample" — plausible on a single
  random-walk realisation that drifted, since a persistently-long spec tracks
  the drift in both halves. PBO earns its place on populations where the
  selection is the artefact; here the deflation did the work.
- **It says nothing about whether any edge exists.** The population had none by
  construction. Whether the DSL can express a real one is M6's question, and
  this record only bounds how many times it may ask.

## Consequences

- `promotion.min_oos_deflated_sharpe` and
  `promotion.min_deflated_sharpe_probability` are unchanged and should not be
  changed to make a sweep promotable.
- `tb research null-gate` exists so this measurement can be re-run after any
  edit to `promotion.*` or `costs.*`, rather than living only in the suite.
- `test_the_search_size_decides_what_edge_is_provable` asserts the table above
  at two points, so a change that flattened the haircut fails loudly.
- M6's `SpecProposer` interface should carry the search's trial count, and its
  loop should treat that count as a spend rather than as a statistic.
