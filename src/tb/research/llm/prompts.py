"""What a language model is told when it proposes specs — and what it never is.

The plan's constraint, and the reason this module exists separately from the
adapter: **a model that has memorised market history is itself a lookahead
channel no schema can catch.** Show it "AAPL, 2020-03-16, close 60.55" and it
knows what happened next, and a spec it proposes can encode that knowledge in a
threshold that looks like an idea. The sealed holdout cannot help, because the
contamination happened before the data layer was involved.

So the prompt is built from exactly two inputs, both abstract by construction:

* a **feature dictionary** — the names, lookbacks and units of the fixed feature
  library, which is the vocabulary a spec can use and nothing else;
* a **regime description** — whether the reference index is above or below its
  long average, or unknown. Not the index, not its level, not the date: the
  `RegimeReading` it is derived from carries a close, a moving average and a
  timestamp, and `RegimeDescription.from_reading` reads only the state.

Neither type has a field that could hold a date, a price or a ticker, which is
the primary guarantee. `audit_prompt` is the tripwire behind it: it scans the
rendered text for anything shaped like a date, a price, an instrument
identifier or a credential, and raises before the text leaves the process. The
two are layered on purpose — a type is only as good as the next edit to the
function that renders it.

The model's reply is untrusted data. Nothing here interprets it; the adapter
hands it to `StrategySpec.parse`, which is the only thing that decides whether
it is a spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from tb.core.errors import TbError
from tb.data.regime import RegimeReading, RegimeState
from tb.features.pipeline import FEATURE_LIBRARY
from tb.research.mutate import COMPARISONS, ProposalBounds

# What each library feature measures, and in what units. Units matter more than
# they look: the grammar lets a spec compare any feature with any other, and a
# model that does not know `sma` is a price while `zscore` is a z-score will
# happily propose `sma_20 > zscore_20`, which is always true and means nothing.
# A test asserts this table and the library have the same keys.
FEATURE_NOTES: dict[str, tuple[str, str]] = {
    "last": ("price", "the most recent split-adjusted close"),
    "sma": ("price", "simple moving average of the closes over the lookback"),
    "return_pct": ("percent", "total return over the lookback, in percent"),
    "stdev_pct": (
        "percent",
        "sample standard deviation of daily returns over the lookback, in percent",
    ),
    "zscore": (
        "z-score",
        "how far the last close sits from the lookback mean, in standard deviations",
    ),
    "max_drawdown_pct": (
        "percent",
        "worst peak-to-trough fall within the lookback, as a positive percent",
    ),
}


class PromptLeak(TbError):
    """A rendered prompt contains something a model must never see."""


@dataclass(frozen=True, slots=True)
class RegimeDescription:
    """The market's state, with everything that could date it removed.

    Three values and no more. "The index is above its long average" is the
    abstraction the plan asks for; a level, a percentage above the average, or
    a volatility figure would each narrow the candidate periods a model with
    memorised history could match it to.
    """

    trend: Literal["above", "below", "unknown"] = "unknown"

    @classmethod
    def from_reading(cls, reading: RegimeReading) -> RegimeDescription:
        """Reads the state and nothing else.

        The reading carries `last_close`, `moving_average`, `as_of` and a free
        text `detail` that may quote both — all of which would date it. Only
        `state` crosses into the prompt, and an unmeasured state is `unknown`
        rather than a guess, for the same reason the regime gate treats it as
        reduced exposure rather than full.
        """
        if reading.state is RegimeState.RISK_ON:
            return cls(trend="above")
        if reading.state is RegimeState.RISK_OFF:
            return cls(trend="below")
        return cls(trend="unknown")

    def sentence(self) -> str:
        if self.trend == "above":
            return "The broad market index is above its long-run moving average."
        if self.trend == "below":
            return "The broad market index is below its long-run moving average."
        return "The state of the broad market is not known."


def render_system_prompt(bounds: ProposalBounds) -> str:
    """The contract: the grammar, the constraints, and the output shape.

    Static for given bounds, so it is the stable prefix of every request.
    """
    features = ", ".join(sorted(FEATURE_LIBRARY))
    ops = ", ".join(COMPARISONS)
    lookbacks = ", ".join(str(look) for look in bounds.lookbacks)
    return (
        "You propose rule-based trading strategies for a long-only daily-bar equity "
        "system, written in a small declarative grammar. You are one source of "
        "candidate ideas among several; every candidate is backtested, counted, and "
        "judged by statistical gates you cannot see. Propose distinct, plausible ideas "
        "rather than variations of one.\n\n"
        "Grammar. A strategy is a JSON object with exactly these keys:\n"
        '  "entry": a predicate — when to open a position;\n'
        '  "exit": a predicate — when to close it;\n'
        '  "expected_edge_bps": a number — the net return per round trip you expect, '
        "in basis points.\n"
        "A predicate is one of:\n"
        '  {"kind": "compare", "op": OP, "left": TERM, "right": TERM}\n'
        '  {"kind": "all", "operands": [PREDICATE, ...]}   (every operand holds)\n'
        '  {"kind": "any", "operands": [PREDICATE, ...]}   (at least one holds)\n'
        '  {"kind": "not", "operand": PREDICATE}\n'
        "A term is one of:\n"
        '  {"kind": "feature", "name": FEATURE, "lookback": LOOKBACK}\n'
        '  {"kind": "const", "value": "NUMBER"}\n'
        f"OP is one of: {ops}.\n"
        f"FEATURE is one of: {features}.\n"
        f"LOOKBACK is one of: {lookbacks} (in daily bars).\n\n"
        "Constraints:\n"
        "  - Long only: a strategy can hold a position or hold cash, never short.\n"
        "  - Positions are held for at least one trading day.\n"
        f"  - expected_edge_bps must be between {bounds.min_edge_bps} and "
        f"{bounds.max_edge_bps}. A round trip costs roughly 40 basis points, so an "
        "idea that cannot plausibly clear several times that is not worth proposing.\n"
        "  - Compare features only with features or constants of the same units.\n"
        "  - Keep each predicate small: three comparisons or fewer.\n\n"
        "Output: a JSON array of strategy objects and nothing else — no prose, no "
        "code fences, no keys beyond the three above."
    )


def render_user_prompt(
    *,
    n: int,
    regime: RegimeDescription,
    required_sharpe: float | None = None,
    n_trials: int | None = None,
) -> str:
    """The request: how many, the vocabulary, the abstract state, the bar to clear.

    `required_sharpe` and `n_trials` are the one piece of search feedback the
    model gets — ADR 0002's "report the trial count to the searcher". They are
    derived from the budget alone, so they say nothing about the data.

    Counts are written with a thousands separator. That is for the tripwire, not
    for the reader: a budget of 2000 written plainly is indistinguishable from a
    year, and `audit_prompt` would rightly refuse it.
    """
    lines = [f"Propose {n:,} distinct strategies.", "", "Features available:"]
    for name in sorted(FEATURE_NOTES):
        unit, meaning = FEATURE_NOTES[name]
        lines.append(f"  {name} ({unit}): {meaning}")
    lines.extend(["", regime.sentence()])
    if required_sharpe is not None and n_trials is not None:
        lines.extend(
            [
                "",
                f"This search has a budget of {n_trials:,} candidates. Because every "
                "candidate is counted, a strategy selected from a search this size needs "
                f"an out-of-sample Sharpe ratio near {required_sharpe:.2f} to be believed. "
                "Favour ideas with a clear economic reason to persist over ones that "
                "merely fit.",
            ]
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The tripwire
# --------------------------------------------------------------------------
#
# Each pattern names what it catches, because the error is read by whoever just
# edited the prompt builder and the useful message is "you put a date in it".

_LEAKS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("an ISO date", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")),
    ("a calendar year", re.compile(r"\b(?:19[5-9]\d|20[0-4]\d)\b")),
    (
        "a month name",
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|"
            r"November|December)\b"
        ),
    ),
    ("a currency amount", re.compile(r"[$£€¥]\s?\d")),
    ("an ISIN", re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")),
    ("an instrument uid", re.compile(r"\b(?:isin|sym):", re.IGNORECASE)),
    ("a broker ticker", re.compile(r"\b[A-Z0-9]{1,6}_[A-Z]{2}_EQ\b")),
    ("an API key", re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")),
    (
        "a credential variable",
        re.compile(r"\b(?:T212|ALPACA|APCA|ANTHROPIC)_[A-Z_]*(?:KEY|SECRET|TOKEN)\b"),
    ),
    ("a long opaque token", re.compile(r"\b[A-Za-z0-9+/=_-]{40,}\b")),
)


def audit_prompt(text: str) -> None:
    """Raise if the rendered prompt contains a date, a price, an id or a secret.

    A tripwire rather than the guarantee — the guarantee is that the builders
    take only abstract types. This catches the edit that adds "as of {date}" to
    a builder for convenience, which is exactly how a lookahead channel is
    introduced by someone who was not thinking about lookahead.
    """
    for label, pattern in _LEAKS:
        match = pattern.search(text)
        if match is not None:
            raise PromptLeak(
                f"the prompt contains {label} ({match.group(0)!r}). A model with memorised "
                "market history can use a date, a price or an instrument to recognise the "
                "period, and a spec it proposes would then encode what happened next — a "
                "lookahead no schema can catch. Prompts are built from the feature "
                "dictionary and an abstract regime description only."
            )
