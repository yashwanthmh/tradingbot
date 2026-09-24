"""Families, measured from what strategies actually held.

Long-only means every strategy here is a long-equity beta expression. "Ten
uncorrelated strategies" is therefore a claim to be measured rather than an
assumption to be made, and the failure it hides is specific: ten strategies
discover the same trade, the portfolio is one position with ten names on it,
and the per-position cap — which was doing the work of limiting single-name
exposure — silently stops binding.

**Overlap is measured from realised positions, not from declared families or
from return correlation.** Two reasons, and both come from what free data and
a small account actually give you:

* A *declared* family is a label the searcher chose. A searcher that mutates a
  spec into something that trades identically has produced a new label and the
  same position, and grouping by label would not notice.
* *Return* correlation needs a return series per strategy, and at 10-20 trades
  a month per strategy the sample is far too short for a correlation to mean
  anything. Position overlap needs no such sample: two strategies holding AAPL
  on the same day overlapped on that day, as a fact rather than an estimate.

**Grouping is single-linkage on purpose.** If A overlaps B and B overlaps C,
all three are one family even when A and C never held the same name. That is
the conservative direction: the cap is about concentration, and a chain of
overlapping positions concentrates exactly as much as a clique does. Requiring
every pair to overlap would split the chain into three families and cap
nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

# Above this Jaccard overlap, two strategies are one family.
#
# A half rather than something tighter. The number answers "would capping these
# two together be the right call", and at 0.5 two strategies held the same
# instrument on the same session for half of all the sessions either of them
# held anything — which is not two strategies for the purpose of a
# concentration cap. Lower would merge strategies that share a few crowded
# names, which is normal on a 25-symbol universe and not concentration.
FAMILY_OVERLAP_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Holding:
    """One strategy holding one instrument on one session.

    Session-grained rather than instant-grained. The positions here are held
    for days, so a finer grain would measure clock alignment rather than
    overlap, and two strategies entering the same name six hours apart are the
    same position for a concentration cap.
    """

    strategy_id: str
    instrument_uid: str
    session_date: date

    @property
    def key(self) -> tuple[str, date]:
        return (self.instrument_uid, self.session_date)


@dataclass(frozen=True, slots=True)
class Family:
    """A set of strategies whose realised positions overlap."""

    family_id: str
    strategy_ids: tuple[str, ...]
    max_pairwise_overlap: float

    @property
    def size(self) -> int:
        return len(self.strategy_ids)

    @property
    def is_singleton(self) -> bool:
        return self.size == 1

    def summary(self) -> str:
        if self.is_singleton:
            return f"{self.family_id}: {self.strategy_ids[0]} alone"
        return (
            f"{self.family_id}: {self.size} strategies "
            f"({', '.join(self.strategy_ids)}), peak overlap "
            f"{self.max_pairwise_overlap:.2f}"
        )


@dataclass(frozen=True, slots=True)
class CapOutcome:
    """What the family cap did to one allocation round."""

    weights: Mapping[str, Decimal]
    families: tuple[Family, ...]
    capped_families: tuple[str, ...]
    detail: str = ""

    @property
    def n_capped(self) -> int:
        return len(self.capped_families)


def overlap(a: Iterable[Holding], b: Iterable[Holding]) -> float:
    """Jaccard overlap of two strategies' holdings.

    Intersection over union of `(instrument, session)` pairs. Jaccard rather
    than "fraction of A that is also in B", because the asymmetric form makes a
    strategy that trades rarely look like a subset of one that trades
    constantly — it would report an overlap of 1.0 for a strategy holding one
    name on one day inside another's year-long book, which is not the fact a
    concentration cap is after.

    Two strategies that held nothing overlap 0.0 rather than 1.0. The empty
    intersection over the empty union is undefined, and the conservative
    reading of "neither has held anything yet" is that nothing has been shown,
    not that they are identical.
    """
    left = {holding.key for holding in a}
    right = {holding.key for holding in b}
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union)


def families(holdings: Sequence[Holding], *, threshold: float | None = None) -> tuple[Family, ...]:
    """Group strategies into families by realised position overlap.

    Single-linkage: a chain of overlapping strategies is one family. See the
    module note — a chain concentrates as much as a clique, and requiring every
    pair to overlap would split it and cap nothing.

    A strategy with no holdings still gets a family of its own rather than
    being dropped. It has an allocation, so it has to appear in the cap
    arithmetic; dropping it would silently exclude it from the total a family
    cap is computed against.
    """
    bound = FAMILY_OVERLAP_THRESHOLD if threshold is None else threshold
    by_strategy: dict[str, list[Holding]] = {}
    for holding in holdings:
        by_strategy.setdefault(holding.strategy_id, []).append(holding)

    ids = sorted(by_strategy)
    parent: dict[str, str] = {node: node for node in ids}

    def find(node: str) -> str:
        # Path halving, so a long chain does not make this quadratic.
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    peak: dict[str, float] = {}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            score = overlap(by_strategy[left], by_strategy[right])
            if score <= bound:
                continue
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
            merged = find(left)
            peak[merged] = max(peak.get(merged, 0.0), score)

    grouped: dict[str, list[str]] = {}
    for strategy_id in ids:
        grouped.setdefault(find(strategy_id), []).append(strategy_id)

    out: list[Family] = []
    for root in sorted(grouped):
        members = tuple(sorted(grouped[root]))
        # Peak is keyed by whichever root won the union, which path compression
        # can change; take the best recorded for any member.
        best = max((peak.get(member, 0.0) for member in members), default=0.0)
        out.append(
            Family(
                family_id=f"fam_{members[0]}",
                strategy_ids=members,
                max_pairwise_overlap=best,
            )
        )
    return tuple(out)


def apply_family_cap(
    *,
    weights: Mapping[str, Decimal],
    holdings: Sequence[Holding],
    max_family_fraction: Decimal,
    threshold: float | None = None,
) -> CapOutcome:
    """Scale down any family holding more than its share.

    Weights are fractions of deployed capital and are expected to sum to at
    most 1. A family over its cap is scaled *proportionally* within itself —
    every member keeps its relative standing, because the cap is a statement
    about the family's total and not a judgement about which member deserves
    it.

    The freed weight is deliberately **not** redistributed to other families.
    Handing it to the next family along would push that one toward its own cap,
    and the sequence of which family gets the surplus would depend on iteration
    order. Leaving the portfolio smaller is the honest outcome: there was
    nothing uncorrelated to put the money into.
    """
    grouped = families(holdings, threshold=threshold)
    known = {strategy_id for family in grouped for strategy_id in family.strategy_ids}
    # A strategy with an allocation but no holdings yet is its own family.
    extra = sorted(set(weights) - known)
    grouped = grouped + tuple(
        Family(
            family_id=f"fam_{strategy_id}", strategy_ids=(strategy_id,), max_pairwise_overlap=0.0
        )
        for strategy_id in extra
    )

    adjusted = dict(weights)
    capped: list[str] = []
    notes: list[str] = []
    for family in grouped:
        members = [s for s in family.strategy_ids if s in adjusted]
        if not members:
            continue
        total = sum((adjusted[s] for s in members), Decimal(0))
        if total <= max_family_fraction or total <= 0:
            continue
        scale = max_family_fraction / total
        for member in members:
            adjusted[member] = adjusted[member] * scale
        capped.append(family.family_id)
        # A family of one is capped too, and saying "1 strategies, peak overlap
        # 0.00" would read as a correlation finding it is not. The cap that
        # bound there is the plainer one: no single family may hold the whole
        # book, so using the full budget takes several that do not overlap.
        shape = (
            f"{members[0]} alone"
            if len(members) == 1
            else (f"{len(members)} strategies, peak overlap {family.max_pairwise_overlap:.2f}")
        )
        notes.append(
            f"{family.family_id} ({shape}) held {total} of deployed capital against a "
            f"family cap of {max_family_fraction}; scaled by {scale}"
        )

    return CapOutcome(
        weights=adjusted,
        families=grouped,
        capped_families=tuple(capped),
        detail="; ".join(notes),
    )
