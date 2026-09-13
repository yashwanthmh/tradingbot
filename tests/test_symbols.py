"""The symbol map and the cross-venue price gate.

Trading 212 serves no market data, so every signal is computed on one venue's
prices and filled on another's. A mismapped ticker produces a signal that looks
valid, passes every risk check, and buys the wrong company — which is why
mapping is tested as a risk control rather than as a string transform.

Three properties carry the weight:

* a derived mapping does **not** permit an entry;
* the gate is **asymmetric** — a doubtful mapping never blocks an exit;
* a 100x price ratio is reported as a **unit mismatch**, because London quotes
  in pence and calling that a 9,900bps divergence tells nobody anything.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tb.broker.port import Instrument
from tb.data.symbols import (
    Confidence,
    DisagreementKind,
    SymbolMap,
    compare_prices,
    derive_mapping,
    parse_ticker,
)
from tb.ledger.events import EventType
from tb.ledger.store import Ledger

LIMIT_BPS = 75.0


def instrument(ticker: str, currency: str | None = "USD", **kwargs: object) -> Instrument:
    return Instrument(ticker=ticker, currency_code=currency, **kwargs)  # type: ignore[arg-type]


class TestTickerGrammar:
    @pytest.mark.parametrize(
        "ticker,base,venue,market",
        [
            ("AAPL_US_EQ", "AAPL", None, "US"),
            ("TSLA_US_EQ", "TSLA", None, "US"),
            ("VODl_EQ", "VOD", "l", None),
            ("SAPd_EQ", "SAP", "d", None),
            ("VUSAl_EQ", "VUSA", "l", None),
            ("ASMLa_EQ", "ASML", "a", None),
        ],
    )
    def test_known_shapes_parse(
        self, ticker: str, base: str, venue: str | None, market: str | None
    ) -> None:
        parsed = parse_ticker(ticker)
        assert parsed is not None
        assert (parsed.base, parsed.venue_letter, parsed.market) == (base, venue, market)

    @pytest.mark.parametrize("ticker", ["NOTATICKER", "", "AAPL", "random text", "A_B_C_D_E"])
    def test_an_unrecognised_shape_returns_none_rather_than_a_guess(self, ticker: str) -> None:
        assert parse_ticker(ticker) is None


class TestAlpacaDerivation:
    def test_a_us_listing_maps_to_the_bare_symbol(self) -> None:
        mapping = derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca")
        assert mapping.data_symbol == "AAPL"
        assert mapping.confidence is Confidence.DERIVED

    def test_a_non_us_listing_is_unmapped_rather_than_mapped_to_a_us_namesake(self) -> None:
        """The mismapping that buys the wrong company.

        Alpaca's free feed is US-only. Vodafone on the LSE is not the same
        security as anything trading as `VOD` in New York, and pricing one
        while filling the other is the failure this whole module exists for.
        """
        mapping = derive_mapping(instrument("VODl_EQ", currency="GBX"), provider="alpaca")
        assert mapping.data_symbol == ""
        assert mapping.confidence is Confidence.UNMAPPED
        assert "US-only" in mapping.derivation

    def test_currency_outweighs_the_ticker_suffix(self) -> None:
        """Currency is data; the ticker suffix is convention."""
        mapping = derive_mapping(instrument("SOMETHING_EQ", currency="USD"), provider="alpaca")
        assert mapping.data_symbol == "SOMETHING"


class TestYfinanceDerivation:
    @pytest.mark.parametrize(
        "ticker,expected",
        [
            ("AAPL_US_EQ", "AAPL"),
            ("VODl_EQ", "VOD.L"),
            ("SAPd_EQ", "SAP.DE"),
            ("ASMLa_EQ", "ASML.AS"),
            ("MCp_EQ", "MC.PA"),
        ],
    )
    def test_venue_letters_become_suffixes(self, ticker: str, expected: str) -> None:
        mapping = derive_mapping(instrument(ticker, currency="EUR"), provider="yfinance")
        assert mapping.data_symbol == expected

    def test_an_unknown_venue_letter_is_unmapped(self) -> None:
        """Refusing to guess an exchange is the whole point.

        The suffix table is a reconstruction, so an unfamiliar letter must
        produce nothing rather than something plausible-looking.
        """
        mapping = derive_mapping(instrument("XYZz_EQ", currency="EUR"), provider="yfinance")
        assert mapping.confidence is Confidence.UNMAPPED
        assert "refusing to guess" in mapping.derivation

    def test_no_venue_letter_and_not_us_is_ambiguous(self) -> None:
        mapping = derive_mapping(instrument("ABC_EQ", currency="EUR"), provider="yfinance")
        assert mapping.confidence is Confidence.AMBIGUOUS
        assert "different company" in mapping.derivation


class TestUnknownProvider:
    def test_an_unknown_provider_maps_nothing(self) -> None:
        mapping = derive_mapping(instrument("AAPL_US_EQ"), provider="madeup")
        assert mapping.confidence is Confidence.UNMAPPED


class TestPriceComparison:
    def _mapping(self, **kwargs: object) -> object:
        return derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca")

    def test_agreeing_prices_agree(self) -> None:
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("155.42"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.AGREES
        assert not comparison.should_block
        assert comparison.disagreement_bps is not None
        assert comparison.disagreement_bps < 5

    def test_a_divergence_beyond_the_band_blocks(self) -> None:
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("170.00"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.DIVERGED
        assert comparison.should_block
        assert "stale feed" in comparison.detail

    def test_a_hundredfold_ratio_is_a_unit_mismatch_not_a_divergence(self) -> None:
        """London lists in pence; most feeds quote pounds.

        Reporting that as a 9,900bps divergence is technically a block and
        completely unhelpful — the fix is the mapping's unit, not the tolerance.
        """
        comparison = compare_prices(
            derive_mapping(instrument("VODl_EQ", currency="GBX"), provider="yfinance"),
            broker_price=Decimal("0.72"),
            data_price=Decimal("72.00"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.UNIT_MISMATCH
        assert "pence against pounds" in comparison.detail
        assert "widening the tolerance" in comparison.detail

    def test_the_inverse_ratio_is_also_a_unit_mismatch(self) -> None:
        comparison = compare_prices(
            derive_mapping(instrument("VODl_EQ", currency="GBX"), provider="yfinance"),
            broker_price=Decimal("72.00"),
            data_price=Decimal("0.72"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.UNIT_MISMATCH

    @pytest.mark.parametrize("broker,data", [(None, Decimal("1")), (Decimal("1"), None)])
    def test_a_missing_price_cannot_verify(
        self, broker: Decimal | None, data: Decimal | None
    ) -> None:
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=broker,
            data_price=data,
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.MISSING_PRICE
        assert comparison.should_block

    def test_a_non_positive_price_blocks(self) -> None:
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("0"),
            data_price=Decimal("155"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.should_block

    def test_the_band_is_respected_exactly(self) -> None:
        # 1% apart is 100bps, which exceeds a 75bps limit.
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("100.00"),
            data_price=Decimal("101.00"),
            limit_bps=LIMIT_BPS,
        )
        assert comparison.kind is DisagreementKind.DIVERGED
        assert comparison.disagreement_bps is not None
        assert 95 < comparison.disagreement_bps < 105


class TestTheGate:
    def test_an_unmapped_ticker_cannot_be_entered(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        allowed, why = symbol_map.may_enter("AAPL_US_EQ")
        assert not allowed
        assert "tb symbols audit" in why

    def test_a_derived_mapping_cannot_be_entered(self, ledger: Ledger) -> None:
        """Derivation is a proposal, not a confirmation.

        On a fresh install nothing is tradable, which is the correct default for
        a join this dangerous.
        """
        symbol_map = SymbolMap(ledger, provider="alpaca")
        symbol_map.derive_all((instrument("AAPL_US_EQ"),))
        allowed, why = symbol_map.may_enter("AAPL_US_EQ")
        assert not allowed
        assert "not verified" in why

    def test_a_verified_mapping_can_be_entered(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        symbol_map.derive_all((instrument("AAPL_US_EQ"),))
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("155.41"),
            limit_bps=LIMIT_BPS,
        )
        symbol_map.verify(comparison, limit_bps=LIMIT_BPS)
        allowed, _ = symbol_map.may_enter("AAPL_US_EQ")
        assert allowed

    def test_a_blocked_mapping_cannot_be_entered(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        symbol_map.derive_all((instrument("AAPL_US_EQ"),))
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("999.00"),
            limit_bps=LIMIT_BPS,
        )
        symbol_map.block(comparison, limit_bps=LIMIT_BPS)
        allowed, why = symbol_map.may_enter("AAPL_US_EQ")
        assert not allowed
        assert "blocked" in why

    @pytest.mark.parametrize("ticker", ["AAPL_US_EQ", "UNKNOWN_THING", "VODl_EQ"])
    def test_exits_are_never_gated(self, ledger: Ledger, ticker: str) -> None:
        """Refusing to sell would turn a data problem into an unhedged position.

        True for an unmapped ticker, a blocked one, and one nobody has heard of.
        """
        symbol_map = SymbolMap(ledger, provider="alpaca")
        allowed, why = symbol_map.may_exit(ticker)
        assert allowed
        assert "never gated" in why

    def test_blocking_records_an_event(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        symbol_map.derive_all((instrument("AAPL_US_EQ"),))
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("999.00"),
            limit_bps=LIMIT_BPS,
        )
        symbol_map.block(comparison, limit_bps=LIMIT_BPS)
        events = list(ledger.iter_events(event_type=EventType.SYMBOL_BLOCKED))
        assert len(events) == 1
        assert "AAPL_US_EQ" in events[0]["payload_json"]


class TestPersistenceAndVerificationLifecycle:
    def test_a_mapping_round_trips(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="yfinance")
        symbol_map.derive_all((instrument("VODl_EQ", currency="GBX"),))
        stored = symbol_map.get("VODl_EQ")
        assert stored is not None
        assert stored.data_symbol == "VOD.L"
        assert symbol_map.t212_ticker_for("VOD.L") == "VODl_EQ"

    def test_verification_survives_a_re_derivation(self, ledger: Ledger) -> None:
        """Re-running the audit must not silently discard a confirmation.

        Otherwise every audit would reset the whole universe to untradable and
        the verification step would never stick.
        """
        symbol_map = SymbolMap(ledger, provider="alpaca")
        instruments = (instrument("AAPL_US_EQ"),)
        symbol_map.derive_all(instruments)
        comparison = compare_prices(
            derive_mapping(instrument("AAPL_US_EQ"), provider="alpaca"),
            broker_price=Decimal("155.40"),
            data_price=Decimal("155.41"),
            limit_bps=LIMIT_BPS,
        )
        symbol_map.verify(comparison, limit_bps=LIMIT_BPS)

        symbol_map.derive_all(instruments)
        stored = symbol_map.get("AAPL_US_EQ")
        assert stored is not None
        assert stored.confidence is Confidence.VERIFIED
        assert symbol_map.may_enter("AAPL_US_EQ")[0]

    def test_a_changed_derived_symbol_invalidates_an_earlier_verification(
        self, ledger: Ledger
    ) -> None:
        """What was confirmed is not what we would now use.

        If the broker changes an instrument's currency such that the derivation
        produces a different data symbol, the old confirmation is about a
        different security.
        """
        symbol_map = SymbolMap(ledger, provider="yfinance")
        symbol_map.derive_all((instrument("VODl_EQ", currency="GBX"),))
        comparison = compare_prices(
            derive_mapping(instrument("VODl_EQ", currency="GBX"), provider="yfinance"),
            broker_price=Decimal("72.00"),
            data_price=Decimal("72.01"),
            limit_bps=LIMIT_BPS,
        )
        symbol_map.verify(comparison, limit_bps=LIMIT_BPS)
        assert symbol_map.may_enter("VODl_EQ")[0]

        # The same ticker now derives differently.
        symbol_map.derive_all((instrument("VODz_EQ", currency="GBX"),))
        symbol_map._ledger.conn.execute(
            "UPDATE symbol_map SET data_symbol = 'VOD.DE' WHERE t212_ticker = 'VODl_EQ'"
        )
        symbol_map._ledger.conn.commit()
        symbol_map.derive_all((instrument("VODl_EQ", currency="GBX"),))

        stored = symbol_map.get("VODl_EQ")
        assert stored is not None
        assert stored.blocked
        assert "no longer applies" in (stored.blocked_reason or "")
        assert not symbol_map.may_enter("VODl_EQ")[0]

    def test_the_audit_summary_counts_every_bucket(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        instruments = (
            instrument("AAPL_US_EQ"),
            instrument("MSFT_US_EQ"),
            instrument("VODl_EQ", currency="GBX"),
            instrument("NOTATICKER", currency=None),
        )
        symbol_map.derive_all(instruments)
        summary = symbol_map.audit_summary(n_instruments=len(instruments))
        assert summary["n_instruments"] == 4
        assert summary["n_mapped"] == 2
        assert summary["n_unmapped"] == 2
        assert summary["n_verified"] == 0

    def test_the_audit_is_recorded(self, ledger: Ledger) -> None:
        symbol_map = SymbolMap(ledger, provider="alpaca")
        symbol_map.derive_all((instrument("AAPL_US_EQ"), instrument("VODl_EQ", currency="GBX")))
        symbol_map.record_audit(symbol_map.audit_summary(n_instruments=2))
        events = list(ledger.iter_events(event_type=EventType.SYMBOLS_AUDITED))
        assert len(events) == 1
        assert '"n_unmapped":1' in events[0]["payload_json"]
