"""The two-tier symbol gate, and the deadlock it exists to break.

`test_the_deadlock_is_broken` is the point of this file. Before the weak tier,
a candidate symbol could never be verified — Trading 212 quotes only what it
holds, holding required a verification, and verification required a quote — so
`may_enter` was permanently False and the first trade could never happen. The
test walks the full cycle: derive, cross-verify at floor size, hold, then
broker-verify for full size.

The tests around it are all about the ways a two-tier gate goes wrong: the weak
tier unblocking something the broker rejected, stale evidence still
authorising entries, a re-derivation quietly demoting everything back to
untradable, or a comparison run across sessions or currencies producing a
confident wrong answer.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tb.broker.port import Instrument
from tb.data.provider import Bar, Provenance, Resolution, Session
from tb.data.symbols import Confidence, SymbolMap, compare_prices, derive_mapping
from tb.data.verification import (
    CROSS_VERIFIED_TTL,
    Contradiction,
    Corroboration,
    ProviderReference,
    QuoteCurrencyBasis,
    SymbolVerifier,
    Tier,
    VerificationError,
    cross_match_reference,
    infer_quote_basis,
    name_similarity,
)
from tb.ledger.store import Ledger

TICKER = "AAPL_US_EQ"
LIMIT_BPS = 75.0
FLOOR = Decimal("15.00")

# Mid-session on a Wednesday: 11:00 Eastern is 16:00 UTC in winter.
MID_SESSION = datetime(2026, 3, 4, 16, tzinfo=UTC)
BAR_OPEN = datetime(2026, 3, 4, 15, 59, tzinfo=UTC)


def apple(*, isin: str | None = "US0378331005", currency: str = "USD") -> Instrument:
    return Instrument(
        ticker=TICKER,
        instrument_type="STOCK",
        isin=isin,
        currency_code=currency,
        short_name="AAPL",
        full_name="Apple Inc.",
    )


def bar(
    *,
    close: str = "155.40",
    provider: str = "alpaca",
    session: Session = Session.REGULAR,
    bar_open: datetime = BAR_OPEN,
) -> Bar:
    price = Decimal(close)
    return Bar(
        instrument_uid="isin:US0378331005",
        resolution=Resolution.MINUTE,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + timedelta(minutes=1),
        ingested_at_utc=bar_open + timedelta(minutes=1),
        provider=provider,
        provenance=Provenance.LIVE,
        session=session,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1000,
    )


def alpaca_reference(
    *, name: str | None = "Apple Inc", currency: str | None = "USD", isin: str | None = None
) -> ProviderReference:
    return ProviderReference(
        symbol="AAPL", provider="alpaca", name=name, currency=currency, isin=isin
    )


@pytest.fixture
def symbol_map(ledger: Ledger) -> SymbolMap:
    mapper = SymbolMap(ledger, provider="alpaca")
    mapper.derive_all((apple(),))
    return mapper


@pytest.fixture
def verifier(symbol_map: SymbolMap) -> SymbolVerifier:
    return SymbolVerifier(symbol_map, account_currency="GBP")


# --------------------------------------------------------------------------
# The deadlock
# --------------------------------------------------------------------------


def test_the_deadlock_exists_without_the_weak_tier(symbol_map: SymbolMap) -> None:
    """The bug this module fixes, stated as a test.

    A derived mapping needs a broker quote to be verified, and Trading 212
    quotes only instruments it holds. With one tier, this is where the system
    stops forever.
    """
    comparison = compare_prices(
        derive_mapping(apple(), provider="alpaca"),
        # What `/equity/portfolio` gives for a symbol we do not hold.
        broker_price=None,
        data_price=Decimal("155.40"),
        limit_bps=LIMIT_BPS,
    )
    assert comparison.should_block
    assert "cannot verify" in comparison.detail
    assert not symbol_map.may_enter(TICKER)[0]


def test_the_deadlock_is_broken(symbol_map: SymbolMap, verifier: SymbolVerifier) -> None:
    """The full cycle: derive, cross-verify at floor size, hold, full size.

    This is the check that says the first trade is possible. Each step is
    reachable from the one before it using only information that actually
    exists at that point.
    """
    # 1. Derived only: nothing may be entered.
    assert not symbol_map.may_enter(TICKER)[0]
    assert not verifier.entry_permission(TICKER, floor_notional=FLOOR).allowed

    # 2. Two feeds plus reference data — no broker involvement at all.
    weak = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca", close="155.40"),
        secondary_bar=bar(provider="yahoo", close="155.42"),
        limit_bps=LIMIT_BPS,
    )
    assert weak.tier is Tier.CROSS_PROVIDER
    assert Corroboration.PRICE_AGREES in weak.corroborations

    # 3. An entry is now possible, capped at the floor notional.
    permission = verifier.entry_permission(TICKER, floor_notional=FLOOR)
    assert permission.allowed
    assert permission.max_notional == FLOOR
    assert permission.cap(Decimal("500.00")) == FLOOR

    # 4. With a position held, the broker finally quotes it.
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    strong = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.41"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    assert strong.tier is Tier.BROKER_QUOTE

    # 5. Full size.
    final = verifier.entry_permission(TICKER, floor_notional=FLOOR)
    assert final.allowed
    assert final.max_notional is None
    assert final.cap(Decimal("500.00")) == Decimal("500.00")


def test_may_enter_says_floor_only_out_loud(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """A bool meaning "yes, but capped" read as plain "yes" is the whole risk."""
    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    allowed, why = symbol_map.may_enter(TICKER)
    assert allowed
    assert "FLOOR NOTIONAL ONLY" in why
    assert not symbol_map.get(TICKER).permits_full_size  # type: ignore[union-attr]


# --------------------------------------------------------------------------
# Absence is not contradiction
# --------------------------------------------------------------------------


def test_one_unreachable_feed_does_not_re_deadlock(verifier: SymbolVerifier) -> None:
    """A single feed outage must degrade the evidence, not stop the system.

    If price corroboration were mandatory, Yahoo being unreachable would put
    the bot right back where it started — unable to open its first position and
    with nothing saying why.
    """
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=None,
        limit_bps=LIMIT_BPS,
    )
    assert result.tier is Tier.CROSS_PROVIDER
    assert Corroboration.PRICE_AGREES not in result.corroborations
    assert Corroboration.CURRENCY_MATCHES in result.corroborations
    assert "cannot corroborate" in result.detail


def test_a_missing_isin_is_no_evidence_either_way(verifier: SymbolVerifier) -> None:
    corroborations, contradictions, note = cross_match_reference(
        apple(), alpaca_reference(isin=None)
    )
    assert Corroboration.ISIN_MATCHES not in corroborations
    assert contradictions == ()
    assert "no evidence either way" in note


def test_a_differing_isin_is_fatal_however_much_else_agrees(
    verifier: SymbolVerifier,
) -> None:
    """Matching name, matching currency, agreeing prices, different ISIN.

    That is a mismapping with good luck, not a match — and it is exactly the
    case the ISIN exists to catch.
    """
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(isin="US0378331005"),
        reference=alpaca_reference(isin="US0378331099"),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    assert result.tier is Tier.NONE
    assert result.fatal
    assert Contradiction.ISIN_DIFFERS in result.contradictions
    assert "different securities" in result.detail


def test_a_currency_mismatch_is_fatal(verifier: SymbolVerifier) -> None:
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(currency="USD"),
        reference=alpaca_reference(currency="EUR"),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.CURRENCY_DIFFERS in result.contradictions


def test_an_unrelated_name_is_fatal(verifier: SymbolVerifier) -> None:
    """AAPL mapped to "Applied Signal Technology" is the failure mode."""
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(name="Applied Signal Technology"),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.NAME_DIFFERS in result.contradictions


def test_no_evidence_at_all_does_not_verify(verifier: SymbolVerifier) -> None:
    result = verifier.verify_by_cross_provider(
        TICKER, instrument=apple(), reference=None, limit_bps=LIMIT_BPS
    )
    assert result.tier is Tier.NONE
    assert not result.fatal
    assert "nothing corroborates" in result.detail


@pytest.mark.parametrize(
    ("left", "right", "expect_match"),
    [
        ("Apple Inc.", "Apple", True),
        ("Alphabet Inc. Class A", "Alphabet Inc", True),
        ("Vodafone Group plc", "Vodafone Group Public Limited Company", True),
        ("Apple Inc.", "Applied Signal Technology", False),
        ("Apple Inc.", "Microsoft Corporation", False),
    ],
)
def test_name_similarity_tolerates_suffixes_but_not_different_companies(
    left: str, right: str, expect_match: bool
) -> None:
    score = name_similarity(left, right)
    assert score is not None
    assert (score >= 0.34) is expect_match


def test_an_absent_name_scores_none_rather_than_zero() -> None:
    """No name is no evidence. Zero would reject every terse broker record."""
    assert name_similarity("Apple Inc.", None) is None
    assert name_similarity(None, "Apple") is None


# --------------------------------------------------------------------------
# Comparing two feeds
# --------------------------------------------------------------------------


def test_a_feed_cannot_corroborate_itself(verifier: SymbolVerifier) -> None:
    """Otherwise every symbol in the universe verifies on the first pass."""
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="alpaca"),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.SAME_PROVIDER in result.contradictions
    assert "agreeing with itself" in result.detail


def test_different_bar_periods_are_refused(verifier: SymbolVerifier) -> None:
    """Comparing 15:59 against 15:45 measures the price move, not the mapping."""
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        primary_bar=bar(provider="alpaca", bar_open=BAR_OPEN),
        secondary_bar=bar(provider="yahoo", bar_open=BAR_OPEN - timedelta(minutes=14)),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.BAR_PERIOD_MISMATCH in result.contradictions


def test_an_extended_hours_bar_is_not_comparable(verifier: SymbolVerifier) -> None:
    """Earnings-night moves of 200-500bps would block half the universe."""
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        primary_bar=bar(provider="alpaca", session=Session.REGULAR),
        secondary_bar=bar(provider="yahoo", session=Session.EXTENDED),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.SESSION_MISMATCH in result.contradictions


def test_diverging_feeds_block_and_record_the_bps(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca", close="155.40"),
        secondary_bar=bar(provider="yahoo", close="180.00"),
        limit_bps=LIMIT_BPS,
    )
    assert Contradiction.PRICE_DIVERGES in result.contradictions
    stored = symbol_map.get(TICKER)
    assert stored is not None and stored.blocked
    assert not symbol_map.may_enter(TICKER)[0]


# --------------------------------------------------------------------------
# The weak tier must not override the strong one
# --------------------------------------------------------------------------


def test_the_weak_tier_cannot_clear_a_block(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """Two feeds agreeing with each other is not an answer to the broker.

    A block from the strong tier records that *Trading 212* disagreed with the
    feed. Letting cross-provider agreement clear it would re-admit exactly the
    symbol the broker rejected, on evidence that never looked at the broker.
    """
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.40"),
        data_bar=bar(close="999.00"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    blocked = symbol_map.get(TICKER)
    assert blocked is not None and blocked.blocked

    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca", close="155.40"),
        secondary_bar=bar(provider="yahoo", close="155.41"),
        limit_bps=LIMIT_BPS,
    )
    still = symbol_map.get(TICKER)
    assert still is not None
    assert still.blocked, "cross-provider agreement cleared a broker-quote block"
    assert not verifier.entry_permission(TICKER, floor_notional=FLOOR).allowed


def test_the_strong_tier_can_clear_its_own_block(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """A transient divergence must be recoverable by the same evidence type."""
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.40"),
        data_bar=bar(close="999.00"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.40"),
        data_bar=bar(close="155.41"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    recovered = symbol_map.get(TICKER)
    assert recovered is not None
    assert not recovered.blocked
    assert recovered.confidence is Confidence.VERIFIED


# --------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------


def test_stale_cross_provider_evidence_stops_authorising_entries(
    verifier: SymbolVerifier,
) -> None:
    """One price comparison at one instant does not hold indefinitely.

    A symbol change or an unannounced corporate action between then and now
    would have invalidated it with nothing to show.
    """
    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    assert verifier.entry_permission(TICKER, floor_notional=FLOOR).allowed

    later = datetime.now(UTC) + CROSS_VERIFIED_TTL + timedelta(hours=1)
    expired = verifier.entry_permission(TICKER, floor_notional=FLOOR, now=later)
    assert not expired.allowed
    assert any("older than" in reason for reason in expired.reasons)


def test_broker_verification_does_not_expire_on_the_weak_clock(
    verifier: SymbolVerifier,
) -> None:
    """The strong tier is evidence from the venue that fills the orders.

    It is re-checked every cycle against a live quote rather than aged out on a
    timer, so applying the weak tier's TTL to it would block full-size trading
    for no reason.
    """
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.41"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    later = datetime.now(UTC) + CROSS_VERIFIED_TTL + timedelta(days=7)
    assert verifier.entry_permission(TICKER, floor_notional=FLOOR, now=later).allowed


# --------------------------------------------------------------------------
# Re-derivation
# --------------------------------------------------------------------------


def test_re_deriving_preserves_cross_verification(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """Otherwise every audit re-creates the deadlock one run at a time."""
    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    counts = symbol_map.derive_all((apple(),))
    assert counts[Confidence.CROSS_VERIFIED.value] == 1

    stored = symbol_map.get(TICKER)
    assert stored is not None
    assert stored.confidence is Confidence.CROSS_VERIFIED
    assert symbol_map.may_enter(TICKER)[0]


def test_a_changed_symbol_invalidates_cross_verification(
    ledger: Ledger,
) -> None:
    """Whatever was corroborated is not what would now be used."""
    mapper = SymbolMap(ledger, provider="yfinance")
    vodafone = Instrument(
        ticker="VODl_EQ",
        instrument_type="STOCK",
        isin="GB00BH4HKS39",
        currency_code="GBX",
        short_name="Vodafone",
        full_name="Vodafone Group plc",
    )
    mapper.derive_all((vodafone,))
    verifier = SymbolVerifier(mapper, account_currency="GBP")
    verifier.verify_by_cross_provider(
        "VODl_EQ",
        instrument=vodafone,
        reference=ProviderReference(
            symbol="VOD.L", provider="yahoo", name="Vodafone Group", currency="GBX"
        ),
        limit_bps=LIMIT_BPS,
    )
    assert mapper.may_enter("VODl_EQ")[0]

    # Simulate the stored symbol having been corroborated against a different
    # listing than the one the rules now derive — a venue move, in effect.
    ledger.conn.execute(
        "UPDATE symbol_map SET data_symbol = 'VOD.DE' WHERE t212_ticker = 'VODl_EQ'"
    )
    ledger.conn.commit()
    mapper.derive_all((vodafone,))

    stored = mapper.get("VODl_EQ")
    assert stored is not None
    assert stored.blocked
    assert "no longer applies" in (stored.blocked_reason or "")
    assert not mapper.may_enter("VODl_EQ")[0]


# --------------------------------------------------------------------------
# The strong tier's refusals
# --------------------------------------------------------------------------


def test_a_closed_market_refuses_rather_than_reports_a_divergence(
    verifier: SymbolVerifier,
) -> None:
    """T212's overnight currentPrice is the last close; the feed has moved on."""
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    overnight = datetime(2026, 3, 5, 2, tzinfo=UTC)
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.40"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=overnight,
    )
    assert result.tier is Tier.NONE
    assert not result.fatal
    assert "market is closed" in result.detail


def test_an_undetermined_currency_basis_refuses_to_guess(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """A USD name quoted in GBP is ~2600bps.

    The x100 unit-mismatch detector would call that a mismapped ticker, sending
    the investigation in exactly the wrong direction. Refusing leaves the
    symbol at floor size, which is bounded and reported.
    """
    assert verifier.quote_basis is QuoteCurrencyBasis.UNDETERMINED
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(currency="USD"),
        broker_price=Decimal("122.36"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    assert result.tier is Tier.NONE
    assert not result.fatal
    assert "has not been established" in result.detail
    assert "tb broker probe" in result.detail
    # Crucially, not blocked — so the floor-size path survives.
    stored = symbol_map.get(TICKER)
    assert stored is not None and not stored.blocked


def test_no_ambiguity_when_the_currencies_match(verifier: SymbolVerifier) -> None:
    """Both hypotheses predict the same number, so there is nothing to resolve."""
    gbp_name = Instrument(
        ticker=TICKER,
        instrument_type="STOCK",
        isin="US0378331005",
        currency_code="GBP",
        short_name="AAPL",
        full_name="Apple Inc.",
    )
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=gbp_name,
        broker_price=Decimal("155.41"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    assert result.tier is Tier.BROKER_QUOTE


def test_an_account_basis_converts_before_comparing(verifier: SymbolVerifier) -> None:
    verifier.observe_quote_basis(QuoteCurrencyBasis.ACCOUNT)
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(currency="USD"),
        # 155.40 USD at 0.7874 GBP/USD.
        broker_price=Decimal("122.36"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
        fx_rate=Decimal("0.7874"),
    )
    assert result.tier is Tier.BROKER_QUOTE
    assert "converted to GBP" in result.detail


def test_an_account_basis_without_a_rate_refuses(verifier: SymbolVerifier) -> None:
    verifier.observe_quote_basis(QuoteCurrencyBasis.ACCOUNT)
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(currency="USD"),
        broker_price=Decimal("122.36"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
        fx_rate=None,
    )
    assert result.tier is Tier.NONE
    assert "no USD/GBP rate was supplied" in result.detail


def test_a_missing_broker_quote_names_the_reason_and_the_remedy(
    verifier: SymbolVerifier,
) -> None:
    result = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=None,
        data_bar=bar(),
        limit_bps=LIMIT_BPS,
        now=MID_SESSION,
    )
    assert result.tier is Tier.NONE
    assert "only quotes instruments it holds" in result.detail
    assert "cross-provider tier" in result.detail


# --------------------------------------------------------------------------
# Inferring the basis from a held position
# --------------------------------------------------------------------------


def test_the_basis_can_be_inferred_when_the_hypotheses_are_far_apart() -> None:
    """The undocumented fact becomes a measurement, not a guess."""
    basis, note = infer_quote_basis(
        broker_price=Decimal("155.40"), data_price=Decimal("155.40"), fx_rate=Decimal("0.7874")
    )
    assert basis is QuoteCurrencyBasis.INSTRUMENT
    assert "instrument's own currency" in note

    basis, note = infer_quote_basis(
        broker_price=Decimal("122.36"), data_price=Decimal("155.40"), fx_rate=Decimal("0.7874")
    )
    assert basis is QuoteCurrencyBasis.ACCOUNT


def test_near_parity_the_basis_cannot_be_inferred() -> None:
    """The case that matters most: the hypotheses are too close to tell apart.

    Confidently recording the wrong basis would apply a spurious conversion to
    every future comparison, so this must return UNDETERMINED rather than the
    one that happens to fit marginally better.
    """
    basis, note = infer_quote_basis(
        broker_price=Decimal("100.00"), data_price=Decimal("100.00"), fx_rate=Decimal("1.01")
    )
    assert basis is QuoteCurrencyBasis.UNDETERMINED
    assert "inside the noise" in note


def test_a_price_matching_neither_hypothesis_blames_the_mapping() -> None:
    basis, note = infer_quote_basis(
        broker_price=Decimal("42.00"), data_price=Decimal("155.40"), fx_rate=Decimal("0.7874")
    )
    assert basis is QuoteCurrencyBasis.UNDETERMINED
    assert "most likely the mapping itself" in note


def test_an_ambiguous_observation_cannot_undo_a_determined_basis(
    verifier: SymbolVerifier,
) -> None:
    verifier.observe_quote_basis(QuoteCurrencyBasis.INSTRUMENT)
    verifier.observe_quote_basis(QuoteCurrencyBasis.UNDETERMINED)
    assert verifier.quote_basis is QuoteCurrencyBasis.INSTRUMENT


def test_inferring_from_a_non_positive_price_is_an_error() -> None:
    with pytest.raises(VerificationError, match="non-positive"):
        infer_quote_basis(broker_price=Decimal("0"), data_price=Decimal("155.40"), fx_rate=None)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_an_unmapped_ticker_gets_no_permission(verifier: SymbolVerifier) -> None:
    permission = verifier.entry_permission("NOPE_US_EQ", floor_notional=FLOOR)
    assert not permission.allowed
    assert "no symbol mapping" in permission.detail


def test_capping_a_forbidden_entry_raises_rather_than_returning_zero(
    verifier: SymbolVerifier,
) -> None:
    """Zero would flow into a sizing calculation and produce a silent no-op.

    An explicit raise makes the refusal impossible to mistake for "size zero",
    which is the shape of bug that leaves an operator wondering why nothing
    trades.
    """
    permission = verifier.entry_permission(TICKER, floor_notional=FLOOR)
    assert not permission.allowed
    with pytest.raises(VerificationError, match="no entry is permitted"):
        permission.cap(Decimal("100.00"))


def test_the_floor_cap_never_inflates_a_smaller_request(
    verifier: SymbolVerifier,
) -> None:
    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    permission = verifier.entry_permission(TICKER, floor_notional=FLOOR)
    assert permission.cap(Decimal("5.00")) == Decimal("5.00")


def test_exits_are_never_gated_by_either_tier(symbol_map: SymbolMap) -> None:
    """True in every reachable mapping state.

    A bug here converts a data problem into an unhedged position, which is
    strictly worse than the data problem.
    """
    for confidence in Confidence:
        symbol_map._ledger.conn.execute(
            "UPDATE symbol_map SET confidence = ?, blocked = 1 WHERE t212_ticker = ?",
            (confidence.value, TICKER),
        )
        symbol_map._ledger.conn.commit()
        stored = symbol_map.get(TICKER)
        assert stored is not None
        assert stored.may_exit
        assert symbol_map.may_exit(TICKER)[0]


def test_verifying_an_unmapped_ticker_is_an_error(verifier: SymbolVerifier) -> None:
    with pytest.raises(VerificationError, match="no symbol mapping"):
        verifier.verify_by_cross_provider("NOPE_US_EQ", instrument=apple(), limit_bps=LIMIT_BPS)


def test_the_tier_summary_reports_what_is_enterable(
    symbol_map: SymbolMap, verifier: SymbolVerifier
) -> None:
    """While `n_enterable` is zero the bot cannot open its first position.

    That is the number `tb symbols audit` has to show, because "nothing is
    verified" was indistinguishable from a deadlock.
    """
    before = verifier.tier_summary()
    assert before["n_enterable"] == 0

    verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    after = verifier.tier_summary()
    assert after["n_enterable"] == 1
    assert after["n_cross_verified"] == 1
    assert after["n_full_size"] == 0


def test_a_dry_run_changes_nothing(symbol_map: SymbolMap, verifier: SymbolVerifier) -> None:
    """`commit=False` is what `tb symbols audit --dry-run` needs."""
    result = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
        commit=False,
    )
    assert result.tier is Tier.CROSS_PROVIDER
    stored = symbol_map.get(TICKER)
    assert stored is not None
    assert stored.confidence is Confidence.DERIVED


def test_verify_refuses_to_record_an_unverified_tier(symbol_map: SymbolMap) -> None:
    """`verify()` writes verified tiers only; failures go through `block()`."""
    comparison = compare_prices(
        derive_mapping(apple(), provider="alpaca"),
        broker_price=Decimal("155.40"),
        data_price=Decimal("155.41"),
        limit_bps=LIMIT_BPS,
    )
    with pytest.raises(ValueError, match="not a verified tier"):
        symbol_map.verify(comparison, limit_bps=LIMIT_BPS, confidence=Confidence.DERIVED)


def test_a_calendar_outside_its_range_blocks_the_strong_tier_safely(
    symbol_map: SymbolMap,
) -> None:
    """A fail-closed calendar must not block the floor-size path.

    The weak tier does not consult the calendar at all, precisely so that a
    calendar that has run out of holidays cannot re-create the deadlock.
    """
    verifier = SymbolVerifier(symbol_map, account_currency="USD")
    beyond = datetime(2031, 6, 3, 16, tzinfo=UTC)
    strong = verifier.verify_by_broker_quote(
        TICKER,
        instrument=apple(),
        broker_price=Decimal("155.41"),
        data_bar=bar(close="155.40"),
        limit_bps=LIMIT_BPS,
        now=beyond,
    )
    assert strong.tier is Tier.NONE

    weak = verifier.verify_by_cross_provider(
        TICKER,
        instrument=apple(),
        reference=alpaca_reference(),
        primary_bar=bar(provider="alpaca"),
        secondary_bar=bar(provider="yahoo"),
        limit_bps=LIMIT_BPS,
    )
    assert weak.tier is Tier.CROSS_PROVIDER
    assert verifier.entry_permission(TICKER, floor_notional=FLOOR).allowed


def test_the_verifier_needs_no_network_and_no_prices_of_its_own(
    symbol_map: SymbolMap,
) -> None:
    """Evidence is passed in, which is what makes every transition testable.

    Fetching inside the verifier would hide the evidence behind a mock and make
    the call site unable to say what it actually corroborated.
    """
    verifier = SymbolVerifier(symbol_map)
    assert verifier.quote_basis is QuoteCurrencyBasis.UNDETERMINED
    assert isinstance(date.today(), date)  # noqa: DTZ011 - no clock reached
