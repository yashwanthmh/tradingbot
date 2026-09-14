"""FX and universe membership.

Two tests here guard the failures that would be hardest to notice.

`test_a_missing_rate_is_unknown_not_one` covers the single most seductive bug
in the FX module: defaulting a missing rate to 1.0 makes everything run, and
makes a USD position look like a GBP position of the same magnitude —
understating exposure against a GBP ceiling by about a quarter, silently, in
the permissive direction.

`test_a_window_before_the_first_snapshot_is_unmeasured` covers survivorship.
Free data cannot fix the bias; the only defence is refusing to pretend the
record exists, and stamping the window so M5 weighs the backtest accordingly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tb.broker.port import Instrument
from tb.data.asof import UNKNOWN, UnknownValueError
from tb.data.fx import (
    FX_FEE_RATE,
    MAX_RATE_AGE_DAYS,
    Conversion,
    FxError,
    FxRate,
    FxStore,
    flat_rate,
    pair_key,
    rates_from_bars,
)
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.data.universe import (
    Exclusion,
    SurvivorshipFlag,
    UniverseError,
    UniverseStore,
    build_candidates,
    dollar_volume_from_bars,
    select,
)
from tb.ledger.events import EventType
from tb.ledger.store import Ledger

NOW = datetime(2026, 3, 5, 12, tzinfo=UTC)
TODAY = date(2026, 3, 5)


def gbpusd(
    *,
    rate: str = "1.27",
    day: date = TODAY,
    known: datetime | None = None,
    ingested: datetime | None = None,
) -> FxRate:
    moment = known or datetime(day.year, day.month, day.day, 22, tzinfo=UTC)
    return FxRate(
        pair="GBPUSD",
        as_of_date=day,
        available_at_utc=moment,
        ingested_at_utc=ingested or moment,
        rate=Decimal(rate),
        provider="yahoo",
        provenance=Provenance.BACKFILL,
    )


@pytest.fixture
def fx(ledger: Ledger) -> FxStore:
    return FxStore(ledger, run_id="run-test")


@pytest.fixture
def universe(ledger: Ledger) -> UniverseStore:
    return UniverseStore(ledger, run_id="run-test")


def instrument(
    ticker: str,
    *,
    isin: str | None = None,
    kind: str = "STOCK",
    currency: str = "USD",
) -> Instrument:
    return Instrument(
        ticker=ticker,
        instrument_type=kind,
        isin=isin,
        currency_code=currency,
        short_name=ticker.split("_")[0],
    )


# ==========================================================================
# FX
# ==========================================================================


def test_pair_keys_are_canonical() -> None:
    assert pair_key("gbp", "usd") == "GBPUSD"
    assert pair_key(" GBP ", "USD") == "GBPUSD"


def test_a_self_pair_is_not_a_conversion() -> None:
    with pytest.raises(FxError, match="not a conversion"):
        pair_key("GBP", "gbp")


def test_a_non_positive_rate_is_refused() -> None:
    with pytest.raises(FxError, match="non-positive rate"):
        gbpusd(rate="0")


def test_a_rate_states_its_direction() -> None:
    """One pound buys 1.27 dollars — not the other way round.

    An inverted rate is a 60% sizing error that still looks like a plausible
    number, which is the worst kind of wrong.
    """
    rate = gbpusd(rate="1.27")
    assert (rate.base, rate.quote) == ("GBP", "USD")
    assert rate.inverse.pair == "USDGBP"
    assert rate.inverse.rate == pytest.approx(Decimal(1) / Decimal("1.27"))


def test_a_round_trip_through_the_inverse_does_not_drift() -> None:
    """The reciprocal of 1.27 does not terminate.

    Rounding it to a display scale before use would make converting there and
    back land somewhere other than where it started.
    """
    rate = gbpusd(rate="1.27")
    assert rate.inverse.inverse.rate == pytest.approx(rate.rate)


def test_storing_and_reading_a_rate_keeps_it_exact(fx: FxStore) -> None:
    """Text, not REAL. The ceiling check is an exact comparison."""
    fx.record([gbpusd(rate="1.2712345678")])
    found = fx.rate_at("GBP", "USD", as_of=NOW + timedelta(days=1))
    assert found is not None
    assert str(found.rate) == "1.2712345678"


def test_the_inverse_direction_is_derived_not_stored(fx: FxStore) -> None:
    """One stored direction per pair, or the two drift against each other.

    Two independently stored directions eventually disagree, and then sizing
    and P&L use different rates for the same conversion.
    """
    fx.record([gbpusd(rate="1.25")])
    assert fx.pairs_held() == ("GBPUSD",)
    flipped = fx.rate_at("USD", "GBP", as_of=NOW + timedelta(days=1))
    assert flipped is not None
    assert flipped.pair == "USDGBP"
    assert flipped.rate == pytest.approx(Decimal("0.8"))


def test_a_rate_is_invisible_before_its_knowledge_time(fx: FxStore) -> None:
    fx.record([gbpusd(day=TODAY, known=datetime(2026, 3, 5, 22, tzinfo=UTC))])
    assert fx.rate_at("GBP", "USD", as_of=datetime(2026, 3, 5, 21, tzinfo=UTC)) is None
    assert fx.rate_at("GBP", "USD", as_of=datetime(2026, 3, 5, 23, tzinfo=UTC)) is not None


def test_a_restated_rate_is_invisible_to_an_earlier_as_of(fx: FxStore) -> None:
    """Published rates get revised, and a revision is future information.

    Filtering on `ingested_at_utc` as well as `available_at_utc` is easy to
    forget and is exactly the leak: a rate restated last week must not appear in
    a backtest of last month.
    """
    original = gbpusd(rate="1.25", day=date(2026, 3, 2))
    restated = FxRate(
        pair="GBPUSD",
        as_of_date=date(2026, 3, 2),
        available_at_utc=original.available_at_utc,
        ingested_at_utc=datetime(2026, 3, 20, tzinfo=UTC),
        rate=Decimal("1.26"),
        provider="yahoo",
        provenance=Provenance.BACKFILL,
    )
    written, revisions = fx.record([original, restated])
    assert written == 2
    assert len(revisions) == 1

    early = fx.rate_at(
        "GBP", "USD", as_of=datetime(2026, 3, 10, tzinfo=UTC), on_or_before=date(2026, 3, 2)
    )
    late = fx.rate_at(
        "GBP", "USD", as_of=datetime(2026, 3, 25, tzinfo=UTC), on_or_before=date(2026, 3, 2)
    )
    assert early is not None and early.rate == Decimal("1.25")
    assert late is not None and late.rate == Decimal("1.26")


def test_a_stale_rate_is_no_rate(fx: FxStore) -> None:
    """Beyond a few days the world has moved and sizing is guesswork."""
    fx.record([gbpusd(day=TODAY - timedelta(days=MAX_RATE_AGE_DAYS + 2))])
    assert fx.rate_at("GBP", "USD", as_of=NOW, on_or_before=TODAY) is None


def test_a_weekend_gap_is_within_tolerance(fx: FxStore) -> None:
    """Friday's rate is a fine rate on Monday morning."""
    friday = date(2026, 3, 6)
    fx.record([gbpusd(day=friday)])
    monday = datetime(2026, 3, 9, 12, tzinfo=UTC)
    found = fx.rate_at("GBP", "USD", as_of=monday, on_or_before=date(2026, 3, 9))
    assert found is not None
    assert found.as_of_date == friday


def test_converting_charges_the_fee_separately(fx: FxStore) -> None:
    """0.15% per conversion, and a US round trip pays it twice.

    Kept separate from the converted amount so it cannot silently disappear
    into the figure the cost gate is comparing against the edge.
    """
    fx.record([gbpusd(rate="1.25")])
    result = fx.convert(Decimal("100.00"), base="GBP", quote="USD", as_of=NOW + timedelta(days=1))
    assert isinstance(result, Conversion)
    assert result.amount == Decimal("125.00")
    # 125.00 * 0.0015 = 0.1875, quantized half-even to the money scale.
    assert result.fee == Decimal("0.19")
    assert str(FX_FEE_RATE) == "0.0015"
    assert result.net == Decimal("124.81")
    assert result.rate == Decimal("1.25")


def test_a_missing_rate_is_unknown_not_one(fx: FxStore) -> None:
    """The seductive bug, refused structurally.

    A 1.0 fallback makes a USD position look like a GBP position of the same
    magnitude — understating exposure against the GBP ceiling by about a
    quarter. `UNKNOWN` raises on arithmetic, so the sizing path cannot use it
    by accident.
    """
    result = fx.convert(Decimal("100.00"), base="GBP", quote="USD", as_of=NOW)
    assert result is UNKNOWN
    with pytest.raises(UnknownValueError):
        _ = bool(result)


def test_a_float_amount_is_refused(fx: FxStore) -> None:
    fx.record([gbpusd()])
    with pytest.raises(FxError, match="must be Decimal"):
        fx.convert(
            100.0,  # type: ignore[arg-type]
            base="GBP",
            quote="USD",
            as_of=NOW + timedelta(days=1),
        )


def test_rates_are_built_from_ordinary_provider_bars(fx: FxStore) -> None:
    """The FX feed reuses the provider layer rather than growing a second one.

    Yahoo serves `GBPUSD=X` as a daily bar, so the rate inherits the bar's
    validation, its knowledge time and its revision detection — a separate,
    weaker ingest path is how one of the two ends up without them.
    """
    bar_open = datetime(2026, 3, 4, tzinfo=UTC)
    bar = Bar(
        instrument_uid="sym:GBPUSD=X",
        resolution=Resolution.DAILY,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(days=1),
        ingested_at_utc=bar_open + timedelta(days=1),
        provider="yahoo",
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=Decimal("1.2700"),
        high=Decimal("1.2750"),
        low=Decimal("1.2680"),
        close=Decimal("1.2730"),
        # Not reported, which is what spot FX has. Yahoo sends a literal 0 and
        # the provider maps it to None — see `symbol_has_volume`.
        volume=None,
    )
    (rate,) = rates_from_bars([bar], base="GBP", quote="USD", provider="yahoo")
    assert rate.rate == Decimal("1.2730")
    # Knowledge time is the bar's, not the bar's date.
    assert rate.available_at_utc == bar.available_at_utc


def test_a_flat_rate_is_marked_as_hand_entered() -> None:
    """So an audit can find every number a human typed."""
    rates = flat_rate(
        base="GBP", quote="USD", rate=Decimal("1.25"), days=[TODAY, TODAY + timedelta(days=1)]
    )
    assert len(rates) == 2
    assert all(rate.provider == "manual" for rate in rates)


def test_the_fx_table_hash_changes_with_its_contents(fx: FxStore) -> None:
    """A run whose FX table changed is not the same run."""
    empty = fx.table_hash()
    fx.record([gbpusd()])
    assert fx.table_hash() != empty


def test_recording_no_rates_writes_nothing(fx: FxStore) -> None:
    assert fx.record([]) == (0, ())


# ==========================================================================
# Universe
# ==========================================================================


def test_only_plain_equity_and_etfs_are_candidates() -> None:
    """Everything else carries leverage, an expiry, or a swap counterparty.

    None of those is something a bot sized against a GBP 500 ceiling should
    discover by accident.
    """
    candidates, rejections = build_candidates(
        [
            instrument("AAPL_US_EQ", isin="US0378331005"),
            instrument("SPY_US_EQ", isin="US78462F1030", kind="ETF"),
            instrument("TQQQ_US_EQ", isin="US74347X8314", kind="LEVERAGED"),
        ],
        symbol_for={"AAPL_US_EQ": "AAPL", "SPY_US_EQ": "SPY", "TQQQ_US_EQ": "TQQQ"},
    )
    assert {c.t212_ticker for c in candidates} == {"AAPL_US_EQ", "SPY_US_EQ"}
    assert any(r.reason is Exclusion.WRONG_TYPE for r in rejections)


def test_an_unmapped_instrument_is_not_a_candidate() -> None:
    """A member whose prices cannot be fetched consumes a poll slot forever."""
    candidates, rejections = build_candidates(
        [instrument("VOD_EQ", isin="GB00BH4HKS39", currency="GBX")], symbol_for={}
    )
    assert candidates == ()
    assert rejections[0].reason is Exclusion.UNMAPPED


def test_a_ticker_only_identity_is_included_but_flagged() -> None:
    """ISIN is occasionally missing from the broker's own list.

    Such a member is allowed, and recorded as a place where a reused ticker
    could splice two companies' histories.
    """
    candidates, rejections = build_candidates(
        [instrument("XYZ_US_EQ", isin=None)], symbol_for={"XYZ_US_EQ": "XYZ"}
    )
    assert len(candidates) == 1
    assert not candidates[0].has_stable_id
    assert any(r.reason is Exclusion.NO_STABLE_ID for r in rejections)


def test_selection_ranks_by_dollar_volume_and_caps_the_count() -> None:
    """The cap is the rate-limit budget, not a preference.

    Every symbol costs poll capacity each cycle; a universe too large to poll
    inside one decision interval has stale prices by construction.
    """
    candidates, _ = build_candidates(
        [instrument(f"S{index}_US_EQ", isin=f"US000000000{index}") for index in range(5)],
        symbol_for={f"S{index}_US_EQ": f"S{index}" for index in range(5)},
        dollar_volume={
            f"S{index}_US_EQ": Decimal(index) * Decimal(1_000_000) for index in range(5)
        },
    )
    snapshot = select(candidates, max_symbols=2, taken_at=NOW)
    assert snapshot.tickers == ("S4_US_EQ", "S3_US_EQ")
    assert [member.rank for member in snapshot.members] == [1, 2]
    assert any(r.reason is Exclusion.BELOW_RANK_CAP for r in snapshot.rejections)


def test_an_unmeasured_name_is_treated_as_illiquid() -> None:
    """Conservative direction: no measurement means no slippage estimate.

    Treating an unmeasured name as liquid would let the cost gate compare an
    edge against a guess.
    """
    candidates, _ = build_candidates(
        [instrument("THIN_US_EQ", isin="US0000000099")], symbol_for={"THIN_US_EQ": "THIN"}
    )
    snapshot = select(
        candidates, max_symbols=25, min_dollar_volume=Decimal(1_000_000), taken_at=NOW
    )
    assert len(snapshot) == 0
    assert snapshot.rejections[0].reason is Exclusion.ILLIQUID
    assert "cannot be estimated" in snapshot.rejections[0].detail


def test_the_account_currency_is_a_tiebreak_not_a_filter() -> None:
    """0.15% per conversion is ~30bps on a round trip, against a 5-20bps edge.

    A hard filter would empty a US-large-cap universe held from a GBP account,
    so the preference only breaks ties.
    """
    candidates, _ = build_candidates(
        [
            instrument("A_US_EQ", isin="US0000000001", currency="USD"),
            instrument("B_EQ", isin="GB0000000002", currency="GBP"),
        ],
        symbol_for={"A_US_EQ": "A", "B_EQ": "B"},
        dollar_volume={"A_US_EQ": Decimal(5_000_000), "B_EQ": Decimal(5_000_000)},
    )
    snapshot = select(candidates, max_symbols=2, preferred_currency="GBP", taken_at=NOW)
    assert snapshot.tickers[0] == "B_EQ"
    assert "account currency" in snapshot.members[0].selection_reason
    # But the USD name is still selectable.
    assert len(snapshot) == 2


def test_blocked_symbols_never_enter_the_universe() -> None:
    candidates, _ = build_candidates(
        [instrument("BAD_US_EQ", isin="US0000000003")], symbol_for={"BAD_US_EQ": "BAD"}
    )
    snapshot = select(candidates, max_symbols=25, blocked=["BAD_US_EQ"], taken_at=NOW)
    assert len(snapshot) == 0
    assert snapshot.rejections[0].reason is Exclusion.BLOCKED


def test_selection_is_deterministic() -> None:
    """Two runs of the same backtest must select the same names.

    Ties broken by ticker rather than left to dict ordering; otherwise the
    universe itself becomes a source of run-to-run variance.
    """
    candidates, _ = build_candidates(
        [instrument(f"T{index}_US_EQ", isin=f"US000000010{index}") for index in range(6)],
        symbol_for={f"T{index}_US_EQ": f"T{index}" for index in range(6)},
        dollar_volume={f"T{index}_US_EQ": Decimal(1_000_000) for index in range(6)},
    )
    first = select(candidates, max_symbols=3, taken_at=NOW)
    second = select(list(reversed(candidates)), max_symbols=3, taken_at=NOW)
    assert first.tickers == second.tickers
    assert first.snapshot_id == second.snapshot_id


def test_a_zero_cap_is_refused() -> None:
    with pytest.raises(UniverseError, match="at least 1"):
        select([], max_symbols=0)


def test_recording_a_snapshot_emits_its_event(universe: UniverseStore, ledger: Ledger) -> None:
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    snapshot = select(candidates, max_symbols=25, taken_at=NOW)
    assert universe.record(snapshot)

    events = list(ledger.iter_events(event_type=EventType.DATA_UNIVERSE_SNAPSHOT_TAKEN))
    assert len(events) == 1
    assert events[0]["aggregate_id"] == snapshot.snapshot_id

    stored = universe.get(snapshot.snapshot_id)
    assert stored is not None
    assert stored.uids == ("isin:US0378331005",)


def test_re_recording_the_same_snapshot_is_a_no_op(universe: UniverseStore) -> None:
    """Content-derived ids, so running the selector twice a day is idempotent."""
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    snapshot = select(candidates, max_symbols=25, taken_at=NOW)
    assert universe.record(snapshot)
    assert not universe.record(snapshot)


def test_membership_as_of_an_instant_ignores_later_snapshots(
    universe: UniverseStore,
) -> None:
    early, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    later, _ = build_candidates(
        [
            instrument("AAPL_US_EQ", isin="US0378331005"),
            instrument("MSFT_US_EQ", isin="US5949181045"),
        ],
        symbol_for={"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"},
    )
    universe.record(select(early, max_symbols=25, taken_at=NOW))
    universe.record(select(later, max_symbols=25, taken_at=NOW + timedelta(days=30)))

    at_first = universe.as_of(NOW + timedelta(days=1))
    assert at_first is not None
    assert len(at_first) == 1

    at_second = universe.as_of(NOW + timedelta(days=40))
    assert at_second is not None
    assert len(at_second) == 2


def test_an_instant_before_every_snapshot_has_no_membership(
    universe: UniverseStore,
) -> None:
    """The survivorship bug in one line, refused.

    Falling back to the newest snapshot would answer a question about 2019 with
    today's survivors and look like a successful lookup.
    """
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    universe.record(select(candidates, max_symbols=25, taken_at=NOW))
    assert universe.as_of(NOW - timedelta(days=365)) is None


def test_a_window_before_the_first_snapshot_is_unmeasured(
    universe: UniverseStore,
) -> None:
    """Labelled, not forbidden.

    On free data, `UNMEASURED` is the only option for the ten-year window the
    promotion gate needs. The unacceptable outcome is not a biased backtest —
    it is a biased backtest nobody knew was biased.
    """
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    universe.record(select(candidates, max_symbols=25, taken_at=NOW))

    flag, note = universe.survivorship_for(
        window_start=NOW - timedelta(days=3650), window_end=NOW - timedelta(days=30)
    )
    assert flag is SurvivorshipFlag.UNMEASURED
    assert not flag.admissible_for_promotion
    assert "today's survivors" in note


def test_a_window_straddling_the_first_snapshot_is_partial(
    universe: UniverseStore,
) -> None:
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    universe.record(select(candidates, max_symbols=25, taken_at=NOW))

    flag, note = universe.survivorship_for(
        window_start=NOW - timedelta(days=3650), window_end=NOW + timedelta(days=30)
    )
    assert flag is SurvivorshipFlag.PARTIAL
    assert "inside the window" in note


def test_a_window_fully_covered_by_snapshots_is_measured(
    universe: UniverseStore,
) -> None:
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    universe.record(select(candidates, max_symbols=25, taken_at=NOW))

    flag, _ = universe.survivorship_for(
        window_start=NOW + timedelta(days=1), window_end=NOW + timedelta(days=30)
    )
    assert flag is SurvivorshipFlag.MEASURED
    assert flag.admissible_for_promotion


def test_with_no_snapshots_at_all_every_window_is_unmeasured(
    universe: UniverseStore,
) -> None:
    flag, note = universe.survivorship_for(window_start=NOW, window_end=NOW)
    assert flag is SurvivorshipFlag.UNMEASURED
    assert "no universe snapshots exist" in note


def test_changes_between_snapshots_report_departures(universe: UniverseStore) -> None:
    """A name leaving is the interesting direction.

    It may have been delisted, acquired, or just fallen below the liquidity
    floor — and only the first two mean a held position needs attention.
    """
    both, _ = build_candidates(
        [
            instrument("AAPL_US_EQ", isin="US0378331005"),
            instrument("MSFT_US_EQ", isin="US5949181045"),
        ],
        symbol_for={"AAPL_US_EQ": "AAPL", "MSFT_US_EQ": "MSFT"},
    )
    one, _ = build_candidates(
        [instrument("MSFT_US_EQ", isin="US5949181045")], symbol_for={"MSFT_US_EQ": "MSFT"}
    )
    first = select(both, max_symbols=25, taken_at=NOW)
    second = select(one, max_symbols=25, taken_at=NOW + timedelta(days=30))
    universe.record(first)
    universe.record(second)

    added, removed = universe.changes_between(first.snapshot_id, second.snapshot_id)
    assert added == ()
    assert removed == ("isin:US0378331005",)


def test_dollar_volume_prices_the_shares() -> None:
    """A 5-dollar stock on ten million shares is less liquid than a 400-dollar
    stock on a million, and position sizing is in currency."""
    bars = [
        Bar(
            instrument_uid="isin:US0378331005",
            resolution=Resolution.DAILY,
            bar_open_utc=datetime(2026, 3, 1, tzinfo=UTC) + timedelta(days=index),
            available_at_utc=datetime(2026, 3, 2, tzinfo=UTC) + timedelta(days=index),
            ingested_at_utc=datetime(2026, 3, 2, tzinfo=UTC) + timedelta(days=index),
            provider="alpaca",
            provenance=Provenance.BACKFILL,
            session=Session.REGULAR,
            open=Decimal("100"),
            high=Decimal("100"),
            low=Decimal("100"),
            close=Decimal("100"),
            volume=1_000,
        )
        for index in range(3)
    ]
    assert dollar_volume_from_bars(bars) == Decimal("100000")


def test_dollar_volume_of_nothing_is_none_not_zero() -> None:
    """None means unmeasured, which `select` treats as illiquid.

    Zero would mean measured-and-illiquid, which is a different claim.
    """
    assert dollar_volume_from_bars([]) is None


def test_the_universe_table_hash_changes_with_membership(
    universe: UniverseStore,
) -> None:
    empty = universe.table_hash()
    candidates, _ = build_candidates(
        [instrument("AAPL_US_EQ", isin="US0378331005")], symbol_for={"AAPL_US_EQ": "AAPL"}
    )
    universe.record(select(candidates, max_symbols=25, taken_at=NOW))
    assert universe.table_hash() != empty
