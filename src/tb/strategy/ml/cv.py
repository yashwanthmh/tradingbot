"""Walk-forward cross-validation that cannot train on its own answers.

A trading label is not a point in time. "The return over the next five
sessions" is decided at `t` and known only a week later, so neighbouring
samples' labels overlap, and ordinary k-fold CV scatters those overlapping spans
across train and test. The training set then holds samples whose labels were
computed from the very prices the test fold is scored on — a leak that reads as
skill, and the one that makes most published ML backtests worthless. Three rules
close it, all structural rather than conventional:

**Walk-forward.** Each test fold is scored by a model trained only on samples
decided *before* the fold begins: the order a live model meets them in, and the
only order in which "trained on the past, tested on the future" is literally
true rather than approximately.

**Purged.** A training sample is kept only if its label was *known* before the
fold begins — its span's end, not its start, is what is compared. Without this
the last `horizon` samples before every fold carry labels computed from prices
inside it.

**Embargoed.** A further gap before the fold. Features are rolling windows, so
a sample decided the day before the fold shares nearly all of its inputs with
the fold's first rows; the embargo stops the model being scored on
near-duplicates of the last thing it was fitted to.

A decision time is never split between folds. Several instruments are decided
at the same instant, and a fold boundary through the middle of them would put
the same moment in both the training and the test set.

Pure: no data, no model, no clock. The properties are tested exhaustively on
spans alone, which is the point of keeping it that way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from tb.core.errors import TbError


class CrossValidationError(TbError):
    """The samples cannot be split into usable folds."""


@dataclass(frozen=True, slots=True)
class LabelSpan:
    """When a sample's features were taken, and when its label became knowable."""

    decided_at: datetime
    known_at: datetime

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None or self.known_at.tzinfo is None:
            raise CrossValidationError("label spans must be timezone-aware")
        if self.known_at < self.decided_at:
            raise CrossValidationError(
                f"a label known at {self.known_at.isoformat()} before the decision it "
                f"labels at {self.decided_at.isoformat()} is a label from the future"
            )


@dataclass(frozen=True, slots=True)
class Fold:
    """One test fold and the samples a model scoring it may be trained on."""

    index: int
    train: tuple[int, ...]
    test: tuple[int, ...]
    test_start: datetime
    test_end: datetime
    # Decided before the fold, but with a label not known in time (or inside
    # the embargo). Reported, because a fold that purges most of its history
    # is a fold whose score says little.
    n_purged: int


def walk_forward_folds(
    spans: Sequence[LabelSpan],
    *,
    n_folds: int,
    embargo: timedelta,
    min_train: int = 1,
) -> tuple[Fold, ...]:
    """Split samples into walk-forward test folds, each with its purged training set.

    The distinct decision times are cut into `n_folds + 1` contiguous blocks of
    near-equal size. The first is history only; each later block is a test
    fold, trained on every sample decided before it whose label was known at
    least `embargo` before it began. A fold left with fewer than `min_train`
    training samples is dropped rather than scored on a model that has barely
    seen anything; `CrossValidationError` if none survives.
    """
    if n_folds < 1:
        raise CrossValidationError(f"n_folds must be at least 1, got {n_folds}")
    if embargo < timedelta(0):
        raise CrossValidationError(f"a negative embargo ({embargo}) is a lookahead")
    times = sorted({span.decided_at for span in spans})
    if len(times) < n_folds + 1:
        raise CrossValidationError(
            f"{len(times)} distinct decision time(s) cannot make {n_folds} walk-forward "
            "fold(s) with any history before the first"
        )

    blocks = _blocks(times, n_folds + 1)
    folds: list[Fold] = []
    for index, block in enumerate(blocks[1:], start=1):
        start, end = block[0], block[-1]
        test = tuple(i for i, span in enumerate(spans) if start <= span.decided_at <= end)
        before = [i for i, span in enumerate(spans) if span.decided_at < start]
        train = tuple(i for i in before if spans[i].known_at + embargo <= start)
        if len(train) < min_train:
            continue
        folds.append(
            Fold(
                index=index,
                train=train,
                test=test,
                test_start=start,
                test_end=end,
                n_purged=len(before) - len(train),
            )
        )
    if not folds:
        raise CrossValidationError(
            f"no fold keeps {min_train} training sample(s) once labels not known "
            f"{embargo} before each fold are purged: the history is too short for the "
            "label horizon"
        )
    return tuple(folds)


def _blocks(times: Sequence[datetime], n_blocks: int) -> list[list[datetime]]:
    """`times` cut into `n_blocks` contiguous runs, sizes differing by at most one."""
    base, extra = divmod(len(times), n_blocks)
    blocks: list[list[datetime]] = []
    cursor = 0
    for block in range(n_blocks):
        size = base + (1 if block < extra else 0)
        blocks.append(list(times[cursor : cursor + size]))
        cursor += size
    return blocks
