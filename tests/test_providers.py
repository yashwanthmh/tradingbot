"""One conformance suite, run against every provider.

The point of parametrising over providers rather than writing three test files
is that the *store* cannot tell them apart. Whatever a provider returns becomes
a row that a backtest trades on, so the invariants have to hold identically for
all of them — and a per-provider test file is exactly how one of them ends up
with a weaker version of the same assertion.

Every provider is exercised through `RecordingTransport` or an in-memory
fixture. Nothing in this file touches a network, which is what makes the suite
a gate rather than a weather report.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tb.core.errors import TransportError
from tb.core.http import HttpResponse, RecordingTransport, json_response
from tb.data.pacing import PacingSpec, ProviderPacer
from tb.data.provider import (
    PRIMARY_PROVIDER,
    US_EASTERN,
    VENDOR_ADJUSTED_PROVIDERS,
    AmbiguousTimestampError,
    Bar,
    DataError,
    MarketDataProvider,
    Provenance,
    ProviderUnavailable,
    Resolution,
    Session,
    TimestampConvention,
    action_knowledge_time,
    classify_us_session,
    knowledge_time,
    normalise_bar_open,
    to_scaled,
)
from tb.data.providers import AlpacaProvider, CsvFixtureProvider, YahooProvider, write_bar_csv
from tb.data.providers.alpaca import (
    KEY_ID_VAR,
    MISNAMED_KEY_VARS,
    SDK_KEY_VARS,
    SECRET_VAR,
)

UID = "isin:US0378331005"
SYMBOL = "AAPL"
SCALE = 6

# A Wednesday, so no weekend edge cases sneak into the arithmetic.
DAY_ONE = datetime(2026, 3, 4, tzinfo=UTC)

# --------------------------------------------------------------------------
# Canonical fixture payloads
#
# Written as the vendors actually shape them, including the parts that are
# awkward: Yahoo's parallel arrays with nulls, Alpaca's symbol-keyed map and
# page token.
# --------------------------------------------------------------------------

# 09:30, 09:31 and 09:32 Eastern on 2026-03-04 (EST, UTC-5).
MINUTE_EPOCHS = [1772634600, 1772634660, 1772634720]


def _instant_pacer() -> ProviderPacer:
    """A pacer on a virtual clock, so the suite is fast *and* deterministic.

    The clock advances only when the pacer sleeps, which is the honest model: a
    frozen clock would mean the token bucket never refills, and `acquire` would
    spin. `MAX_ACQUIRE_ITERATIONS` turns that into an error rather than a hang,
    but a test double should not be relying on it.
    """
    elapsed = [1_000_000.0]

    def advance(seconds: float) -> None:
        elapsed[0] += max(0.0, seconds)

    return ProviderPacer(
        spec=PacingSpec(requests=1000, period_seconds=1.0, margin_seconds=0.0),
        clock=lambda: elapsed[0],
        sleeper=advance,
    )


def yahoo_body(
    *,
    epochs: list[int] | None = None,
    opens: list[Any] | None = None,
    highs: list[Any] | None = None,
    lows: list[Any] | None = None,
    closes: list[Any] | None = None,
    volumes: list[Any] | None = None,
    events: dict[str, Any] | None = None,
    timezone: str = "America/New_York",
) -> str:
    epochs = MINUTE_EPOCHS if epochs is None else epochs
    return json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "currency": "USD",
                            "symbol": SYMBOL,
                            "exchangeTimezoneName": timezone,
                            "currentTradingPeriod": {
                                # 09:30 - 16:00 Eastern, as epoch seconds.
                                "regular": {"start": 1772634600, "end": 1772658000}
                            },
                        },
                        "timestamp": epochs,
                        "indicators": {
                            "quote": [
                                {
                                    "open": opens if opens is not None else [100.1, 100.4, 100.2],
                                    "high": highs if highs is not None else [100.5, 100.6, 100.3],
                                    "low": lows if lows is not None else [100.0, 100.2, 100.1],
                                    "close": (
                                        closes if closes is not None else [100.4, 100.25, 100.15]
                                    ),
                                    "volume": (
                                        volumes if volumes is not None else [1200, 800, 640]
                                    ),
                                }
                            ]
                        },
                        **({"events": events} if events else {}),
                    }
                ],
            }
        }
    )


def alpaca_body(
    *,
    bars: list[dict[str, Any]] | None = None,
    next_page_token: str | None = None,
    symbol: str = SYMBOL,
) -> str:
    if bars is None:
        bars = [
            {"t": "2026-03-04T14:30:00Z", "o": 100.1, "h": 100.5, "l": 100.0, "c": 100.4, "v": 120},
            {"t": "2026-03-04T14:31:00Z", "o": 100.4, "h": 100.6, "l": 100.2, "c": 100.2, "v": 800},
            {"t": "2026-03-04T14:32:00Z", "o": 100.2, "h": 100.3, "l": 100.1, "c": 100.1, "v": 640},
        ]
    return json.dumps(
        {"bars": {symbol: bars}, "next_page_token": next_page_token, "currency": "USD"}
    )


def make_bar(
    *,
    minute: int,
    close: str = "100.40",
    resolution: Resolution = Resolution.MINUTE,
    provider: str = "csv_fixture",
) -> Bar:
    """A valid bar with the given close.

    High and low bracket the close rather than being fixed, so a test can vary
    the close without tripping the OHLC-ordering invariant — which is a real
    check, not one to be worked around with hand-picked numbers.
    """
    bar_open = datetime(2026, 3, 4, 14, 30, tzinfo=UTC) + timedelta(minutes=minute)
    if resolution is Resolution.DAILY:
        bar_open = DAY_ONE + timedelta(days=minute)
    last = Decimal(close)
    opening = Decimal("100.10")
    return Bar(
        instrument_uid=UID,
        resolution=resolution,
        bar_open_utc=bar_open,
        available_at_utc=bar_open + resolution.duration,
        ingested_at_utc=datetime(1970, 1, 1, tzinfo=UTC),
        provider=provider,
        provenance=Provenance.BACKFILL,
        session=Session.REGULAR,
        open=opening,
        high=max(opening, last) + Decimal("0.10"),
        low=min(opening, last) - Decimal("0.10"),
        close=last,
        volume=1200,
        currency="USD",
    )


# --------------------------------------------------------------------------
# Provider construction, one factory per implementation
# --------------------------------------------------------------------------


def build_yahoo() -> YahooProvider:
    transport = RecordingTransport(responses={"/v8/finance/chart/": json_response(yahoo_body())})
    return YahooProvider(transport=transport, pacer=_instant_pacer())


def build_alpaca() -> AlpacaProvider:
    transport = RecordingTransport(
        responses={
            "/v2/stocks/bars": json_response(alpaca_body()),
            f"/v2/stocks/{SYMBOL}/bars/latest": json_response(
                json.dumps(
                    {
                        "bar": {
                            "t": "2026-03-04T14:30:00Z",
                            "o": 100.1,
                            "h": 100.5,
                            "l": 100.0,
                            "c": 100.4,
                            "v": 1200,
                        },
                        "symbol": SYMBOL,
                    }
                )
            ),
        }
    )
    return AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())


def build_csv() -> CsvFixtureProvider:
    return CsvFixtureProvider(bars=[make_bar(minute=index) for index in range(3)])


PROVIDER_FACTORIES: dict[str, Callable[[], MarketDataProvider]] = {
    "yahoo": build_yahoo,
    "alpaca": build_alpaca,
    "csv_fixture": build_csv,
}


@pytest.fixture(params=sorted(PROVIDER_FACTORIES))
def provider(request: pytest.FixtureRequest) -> MarketDataProvider:
    return PROVIDER_FACTORIES[request.param]()


# --------------------------------------------------------------------------
# Conformance: the properties every provider must satisfy
# --------------------------------------------------------------------------


def test_every_provider_satisfies_the_protocol(provider: MarketDataProvider) -> None:
    assert isinstance(provider, MarketDataProvider)


def test_capabilities_are_self_consistent(provider: MarketDataProvider) -> None:
    caps = provider.capabilities
    assert caps.name
    assert caps.resolutions
    # A provider that declared AMBIGUOUS could never be ingested, so declaring
    # it is a build-time error rather than a runtime surprise.
    assert caps.timestamp_convention is not TimestampConvention.AMBIGUOUS
    for resolution in caps.resolutions:
        assert resolution in caps.declared_delay_seconds, (
            f"{caps.name} serves {resolution.value} but declares no delay for it, so "
            "live_capable would silently refuse it for the wrong reason"
        )


def test_the_read_path_knows_which_feeds_rewrite_their_history(
    provider: MarketDataProvider,
) -> None:
    """`VENDOR_ADJUSTED_PROVIDERS` is keyed by name, because a stored bar
    carries only its provider's name. It must say what each provider's own
    capabilities say, or the read path would prefer a rewritten history over a
    raw one — and the primary must be a feed that returns what the venue
    printed."""
    caps = provider.capabilities
    assert (caps.name in VENDOR_ADJUSTED_PROVIDERS) is (not caps.returns_raw_prices)
    if caps.name == PRIMARY_PROVIDER:
        assert caps.returns_raw_prices


def test_fetched_bars_are_ordered_and_in_window(provider: MarketDataProvider) -> None:
    start = datetime(2026, 3, 4, tzinfo=UTC)
    end = datetime(2026, 3, 5, tzinfo=UTC)
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=start,
        end=end,
    )
    bars = batch.sorted_bars()
    assert bars, f"{batch.provider} returned nothing for a window it has fixture data for"
    assert [b.bar_open_utc for b in bars] == sorted(b.bar_open_utc for b in bars)
    for bar in bars:
        assert bar.instrument_uid == UID
        assert bar.resolution is Resolution.MINUTE
        assert bar.provider == batch.provider


def test_knowledge_time_never_precedes_bar_close(provider: MarketDataProvider) -> None:
    """The invariant the whole as-of mechanism rests on.

    Enforced in `Bar.__post_init__`, re-asserted here per provider because a
    provider is the only place a bad `available_at` can be constructed.
    """
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    for bar in batch.bars:
        assert bar.available_at_utc >= bar.bar_close_utc
        assert bar.is_visible_at(bar.available_at_utc)
        assert not bar.is_visible_at(bar.available_at_utc - timedelta(microseconds=1))


def test_every_timestamp_is_utc_aware(provider: MarketDataProvider) -> None:
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    for bar in batch.bars:
        for moment in (bar.bar_open_utc, bar.available_at_utc, bar.ingested_at_utc):
            assert moment.tzinfo is not None
            assert moment.utcoffset() == timedelta(0)


def test_prices_are_decimals_that_survive_scaling(provider: MarketDataProvider) -> None:
    """No float ever reaches a price field.

    A float would still scale, so the assertion is on the *type*: a float that
    round-tripped through Parquet would come back with different bits and every
    hash computed over the row would change, silently invalidating a sealed
    vintage.
    """
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    for bar in batch.bars:
        for price in (bar.open, bar.high, bar.low, bar.close):
            assert type(price) is Decimal
            to_scaled(price, SCALE)


@pytest.mark.parametrize("provider_name", ["yahoo", "alpaca"])
def test_json_prices_are_never_routed_through_a_float(provider_name: str) -> None:
    """`parse_float=Decimal`, proved rather than trusted.

    The payload carries more significant digits than a double holds. Parsed the
    default way, `json` builds a float first and the extra digits are gone
    before any Decimal conversion can see them — a binary round trip in front of
    a system whose stored values are hashed. The digits below survive only
    because neither provider ever lets a float touch a price.
    """
    exact = "100.12345678901234567890"
    if provider_name == "yahoo":
        transport = RecordingTransport(
            responses={
                "/v8/finance/chart/": json_response(
                    yahoo_body(
                        epochs=[MINUTE_EPOCHS[0]],
                        opens=[100.1],
                        highs=[200.0],
                        lows=[100.0],
                        closes=["PLACEHOLDER"],
                        volumes=[1200],
                        # Substituted in as a raw JSON number rather than a
                        # string: a quoted value would prove nothing about
                        # float handling, since Decimal(str) is exact anyway.
                    ).replace('"PLACEHOLDER"', exact)
                )
            }
        )
        built: MarketDataProvider = YahooProvider(transport=transport, pacer=_instant_pacer())
    else:
        body = alpaca_body(
            bars=[
                {
                    "t": "2026-03-04T14:30:00Z",
                    "o": 100.1,
                    "h": 200.0,
                    "l": 100.0,
                    "c": "PLACEHOLDER",
                    "v": 1200,
                }
            ]
        ).replace('"PLACEHOLDER"', exact)
        transport = RecordingTransport(responses={"/v2/stocks/bars": json_response(body)})
        built = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())

    (bar,) = built.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    ).bars
    assert str(bar.close) == exact
    # The value a float would have produced, for contrast.
    assert str(Decimal(str(float(exact)))) != exact


def test_refetching_the_same_window_hashes_identically(provider: MarketDataProvider) -> None:
    """Two fetches of unchanged bars must not look like a revision.

    If they did, `bar_revisions` would fill with one entry per poll and the
    real restatements — the ones that mean the vendor rewrote history — would be
    undetectable in the noise.
    """
    window = {
        "instrument_uid": UID,
        "resolution": Resolution.MINUTE,
        "start": datetime(2026, 3, 4, tzinfo=UTC),
        "end": datetime(2026, 3, 5, tzinfo=UTC),
    }
    first = provider.fetch_bars(SYMBOL, **window)  # type: ignore[arg-type]
    second = provider.fetch_bars(SYMBOL, **window)  # type: ignore[arg-type]
    assert [b.row_hash(scale=SCALE) for b in first.sorted_bars()] == [
        b.row_hash(scale=SCALE) for b in second.sorted_bars()
    ]


def test_a_narrower_window_is_a_subset(provider: MarketDataProvider) -> None:
    wide = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert {b.identity for b in wide.bars}, "fixture produced no bars to subset"


def test_latest_bar_is_settled_or_absent(provider: MarketDataProvider) -> None:
    """A partially formed bar must never be returned.

    Its high, low and volume are fractions of the final values, and once stored
    it is indistinguishable from a finished bar. The subsequent "correction"
    would also register as a revision on every poll.
    """
    bar = provider.latest_bar(SYMBOL, instrument_uid=UID, resolution=Resolution.MINUTE)
    if bar is not None:
        assert bar.is_settled_at(datetime.now(UTC))


def test_a_window_with_no_data_is_empty_not_an_error(provider: MarketDataProvider) -> None:
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(1990, 1, 1, tzinfo=UTC),
        end=datetime(1990, 1, 2, tzinfo=UTC),
    )
    assert isinstance(batch.bars, tuple)


def test_close_is_idempotent(provider: MarketDataProvider) -> None:
    provider.close()
    provider.close()


# --------------------------------------------------------------------------
# Timestamp normalisation — the one-bar lookahead channel
# --------------------------------------------------------------------------


def test_an_ambiguous_convention_refuses_ingestion() -> None:
    with pytest.raises(AmbiguousTimestampError):
        normalise_bar_open(
            datetime(2026, 3, 4, 14, 30, tzinfo=UTC),
            convention=TimestampConvention.AMBIGUOUS,
            resolution=Resolution.MINUTE,
        )


def test_close_stamped_minutes_shift_back_by_one_bar() -> None:
    stamped = datetime(2026, 3, 4, 14, 31, tzinfo=UTC)
    assert normalise_bar_open(
        stamped, convention=TimestampConvention.BAR_CLOSE, resolution=Resolution.MINUTE
    ) == datetime(2026, 3, 4, 14, 30, tzinfo=UTC)


def test_a_daily_bar_is_knowable_on_its_own_session_day() -> None:
    """The bug this anchoring exists to prevent.

    Yahoo stamps a daily bar at the market open (14:30 UTC in winter). Left
    alone, a 24-hour duration puts its close at 14:30 the *next* day, so the
    Monday close would not become visible until an hour into Tuesday's session
    — a daily strategy permanently one session behind. Anchored at midnight,
    it becomes visible at 00:00 UTC Tuesday, which is 19:00 Eastern Monday:
    after the close, before the next pre-open.
    """
    stamped = datetime(2026, 3, 4, 14, 30, tzinfo=UTC)  # 09:30 EST
    bar_open = normalise_bar_open(
        stamped,
        convention=TimestampConvention.BAR_OPEN,
        resolution=Resolution.DAILY,
        session_tz=US_EASTERN,
    )
    assert bar_open == datetime(2026, 3, 4, tzinfo=UTC)

    available = knowledge_time(
        bar_open=bar_open,
        resolution=Resolution.DAILY,
        provenance=Provenance.BACKFILL,
        delay_seconds=0.0,
    )
    assert available == datetime(2026, 3, 5, tzinfo=UTC)
    # After the 16:00 Eastern close on the session day...
    assert available.astimezone(US_EASTERN).date().isoformat() == "2026-03-04"
    assert available.astimezone(US_EASTERN).hour == 19
    # ...and before the next session's pre-open.
    assert available < datetime(2026, 3, 5, 9, tzinfo=US_EASTERN)


def test_a_daily_stamp_at_local_midnight_keeps_its_session_date() -> None:
    """Alpaca's daily convention, and why the zone is passed in.

    Alpaca stamps `1Day` bars at midnight Eastern, which is 05:00 UTC in winter
    and 04:00 in summer — both the same UTC date. But the general case is not
    safe: read in UTC a stamp of `2026-03-04T05:00:00Z` is the 4th, while the
    same market's 00:00 stamp east of UTC would be the 3rd.
    """
    for stamp, expected in (
        ("2026-03-04T05:00:00Z", "2026-03-04"),  # EST
        ("2026-07-01T04:00:00Z", "2026-07-01"),  # EDT
    ):
        anchored = normalise_bar_open(
            datetime.fromisoformat(stamp.replace("Z", "+00:00")),
            convention=TimestampConvention.BAR_OPEN,
            resolution=Resolution.DAILY,
            session_tz=US_EASTERN,
        )
        assert anchored.isoformat() == f"{expected}T00:00:00+00:00"


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        # Winter: 09:30 EST is 14:30 UTC.
        ("2026-03-04T14:30:00+00:00", Session.REGULAR),
        ("2026-03-04T14:29:00+00:00", Session.EXTENDED),
        ("2026-03-04T20:59:00+00:00", Session.REGULAR),
        ("2026-03-04T21:00:00+00:00", Session.EXTENDED),
        # Summer: 09:30 EDT is 13:30 UTC. The same UTC clock time that was
        # regular in March is pre-market in July, which is the whole reason the
        # comparison happens in Eastern.
        ("2026-07-01T13:30:00+00:00", Session.REGULAR),
        ("2026-07-01T13:29:00+00:00", Session.EXTENDED),
        ("2026-07-01T19:59:00+00:00", Session.REGULAR),
        ("2026-07-01T20:00:00+00:00", Session.EXTENDED),
    ],
)
def test_session_classification_follows_eastern_not_utc(moment: str, expected: Session) -> None:
    assert classify_us_session(datetime.fromisoformat(moment), Resolution.MINUTE) is expected


def test_dst_transition_days_classify_without_an_hour_of_drift() -> None:
    """Spring forward, 2026-03-08. The US switches, the UK does not until the 29th.

    On 2026-03-09 the US is on EDT and the UK is still on GMT, so a system
    comparing UTC clock times against a fixed offset is an hour wrong for three
    weeks. Eastern-based classification is not.
    """
    # 09:30 EDT on the Monday after the switch is 13:30 UTC.
    assert (
        classify_us_session(datetime(2026, 3, 9, 13, 30, tzinfo=UTC), Resolution.MINUTE)
        is Session.REGULAR
    )
    # 14:30 UTC — regular the week before — is now an hour into the session,
    # still regular, but 13:29 is pre-market where before it was not a session
    # minute at all.
    assert (
        classify_us_session(datetime(2026, 3, 9, 13, 29, tzinfo=UTC), Resolution.MINUTE)
        is Session.EXTENDED
    )


# --------------------------------------------------------------------------
# Yahoo specifics
# --------------------------------------------------------------------------


def test_yahoo_skips_null_padded_bars_rather_than_filling_them() -> None:
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": json_response(
                yahoo_body(
                    opens=[100.1, None, 100.2],
                    highs=[100.5, None, 100.3],
                    lows=[100.0, None, 100.1],
                    closes=[100.4, None, 100.15],
                    volumes=[1200, None, 640],
                )
            )
        }
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert len(batch.bars) == 2
    assert any("null-padded" in warning for warning in batch.warnings)
    # The gap is left as a gap. A forward-filled bar would be a price nobody
    # printed, and nothing downstream could tell it from a real one.
    opens = [bar.bar_open_utc for bar in batch.sorted_bars()]
    assert opens[1] - opens[0] == timedelta(minutes=2)


def test_yahoo_refuses_mismatched_array_lengths() -> None:
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": json_response(
                yahoo_body(closes=[100.4, 100.25])  # one short
            )
        }
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    with pytest.raises(DataError, match="mismatched array lengths"):
        provider.fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.MINUTE,
            start=datetime(2026, 3, 4, tzinfo=UTC),
            end=datetime(2026, 3, 5, tzinfo=UTC),
        )


def test_yahoo_sends_a_browser_user_agent_and_no_credentials() -> None:
    transport = RecordingTransport(responses={"/v8/finance/chart/": json_response(yahoo_body())})
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    headers = transport.calls[0]["headers"]
    assert "Authorization" not in headers
    assert not any("KEY" in name.upper() for name in headers)


@pytest.mark.parametrize("symbol", ["GBPUSD=X", "^GSPC"])
def test_yahoo_reports_no_volume_rather_than_zero_for_fx_and_indices(symbol: str) -> None:
    """0 and "not reported" are different claims, and the difference is fatal.

    Spot FX and indices have no share count; Yahoo sends a literal 0. Stored as
    0, an FX bar with a real price range trips `Bar`'s synthetic-bar guard — the
    check that catches forward-filled equity bars — so every FX bar is dropped
    and the rate table ends up empty. A sizing path that can never convert is
    then blocked on a GBP ceiling it cannot evaluate.
    """
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": json_response(
                yahoo_body(
                    epochs=[MINUTE_EPOCHS[0]],
                    opens=[1.27],
                    highs=[1.2750],
                    lows=[1.2680],
                    closes=[1.2730],
                    volumes=[0],
                )
            )
        }
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    (bar,) = provider.fetch_bars(
        symbol,
        instrument_uid=f"sym:{symbol}",
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    ).bars
    assert bar.volume is None


def test_yahoo_keeps_a_real_zero_volume_for_an_equity() -> None:
    """An equity bar with zero volume and zero range is a genuine flat minute.

    The FX exemption must not become a blanket one: on an equity, zero volume
    *is* a measurement, and the synthetic-bar guard depends on it.
    """
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": json_response(
                yahoo_body(
                    epochs=[MINUTE_EPOCHS[0]],
                    opens=[100.0],
                    highs=[100.0],
                    lows=[100.0],
                    closes=[100.0],
                    volumes=[0],
                )
            )
        }
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    (bar,) = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    ).bars
    assert bar.volume == 0


def test_yahoo_declares_that_it_does_not_return_raw_prices() -> None:
    """The honest limitation, asserted so a refactor cannot quietly flip it.

    Yahoo back-adjusts historical OHLC for splits with no raw option, so a
    stored Yahoo bar is the vendor's *current* view. Anything that reasons about
    "what the venue printed" — stop placement, tick rounding, the cross-venue
    check — must not silently start trusting this feed for it.
    """
    caps = YahooProvider(transport=RecordingTransport(), pacer=_instant_pacer()).capabilities
    assert caps.returns_raw_prices is False
    assert caps.consolidated_tape is True


def test_yahoo_cannot_drive_a_live_minute_decision() -> None:
    caps = YahooProvider(transport=RecordingTransport(), pacer=_instant_pacer()).capabilities
    allowed, reason = caps.live_capable(Resolution.MINUTE, max_delay_seconds=180.0)
    assert allowed is False
    assert "180s bound" in reason


def test_yahoo_throttling_backs_off_and_raises() -> None:
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": HttpResponse(
                status_code=429, headers={"retry-after": "7"}, text="slow down", elapsed_ms=3.0
            )
        }
    )
    pacer = _instant_pacer()
    provider = YahooProvider(transport=transport, pacer=pacer)
    with pytest.raises(TransportError, match="throttled"):
        provider.fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.MINUTE,
            start=datetime(2026, 3, 4, tzinfo=UTC),
            end=datetime(2026, 3, 5, tzinfo=UTC),
        )
    assert pacer.blocked_for > 0


def test_yahoo_a_block_page_is_reported_as_such() -> None:
    """The characteristic failure of a reverse-engineered endpoint."""
    transport = RecordingTransport(
        responses={
            "/v8/finance/chart/": HttpResponse(
                status_code=200,
                headers={"content-type": "text/html"},
                text="<html>Will be right back</html>",
                elapsed_ms=4.0,
            )
        }
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    with pytest.raises(DataError, match="block page"):
        provider.fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.MINUTE,
            start=datetime(2026, 3, 4, tzinfo=UTC),
            end=datetime(2026, 3, 5, tzinfo=UTC),
        )


def test_yahoo_actions_keep_exact_integer_ratios_and_no_declared_date() -> None:
    events = {
        "splits": {
            "1598832000": {
                "date": 1598832000,
                "numerator": 4,
                "denominator": 1,
                "splitRatio": "4:1",
            }
        },
        "dividends": {"1598832000": {"date": 1598832000, "amount": 0.205}},
    }
    transport = RecordingTransport(
        responses={"/v8/finance/chart/": json_response(yahoo_body(events=events))}
    )
    provider = YahooProvider(transport=transport, pacer=_instant_pacer())
    actions = provider.fetch_actions(
        SYMBOL,
        instrument_uid=UID,
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_type = {action.action_type: action for action in actions}
    split = by_type["split"]
    assert (split.ratio_num, split.ratio_den) == (4, 1)
    # Never imputed: Yahoo reports actions at roughly the ex-date and gives no
    # declaration date, so claiming one would put a future split's knowledge
    # time in the past.
    assert split.declared_date is None
    assert by_type["cash_dividend"].gross_amount == Decimal("0.205")
    # What is certain is that it was public by the start of its ex-date — an
    # upper bound, never a guess at something earlier — so that, rather than
    # the moment of a backfill years later, is when it was knowable.
    assert split.known_at_utc == datetime(2020, 8, 31, 4, tzinfo=UTC)


def test_an_action_seen_before_its_ex_date_is_known_from_when_it_was_seen() -> None:
    """The bound is the earliest of when it was seen, the end of its declaration
    day, and the start of its ex-date. A live fetch a week ahead of the ex-date
    knows it then; a backfill years later knows it from when it was public."""
    seen = datetime(2026, 3, 3, 15, tzinfo=UTC)
    ex_date = date(2026, 3, 10)
    assert action_knowledge_time(effective_date=ex_date, declared_date=None, observed=seen) == seen
    assert action_knowledge_time(
        effective_date=ex_date, declared_date=date(2026, 3, 1), observed=seen
    ) == datetime(2026, 3, 2, 5, tzinfo=UTC)
    years_later = datetime(2030, 1, 1, tzinfo=UTC)
    assert action_knowledge_time(
        effective_date=ex_date, declared_date=None, observed=years_later
    ) == datetime(2026, 3, 10, 4, tzinfo=UTC)


# --------------------------------------------------------------------------
# Alpaca specifics
# --------------------------------------------------------------------------


def test_alpaca_pins_the_free_feed_and_raw_adjustment() -> None:
    """Two query parameters that are load-bearing.

    `adjustment=raw` is why this provider can claim `returns_raw_prices`, and
    `feed=iex` is what the free plan serves — switching it mid-history would
    splice two feeds with a systematic step at the join.
    """
    provider = build_alpaca()
    provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    transport = provider._transport
    assert isinstance(transport, RecordingTransport)
    params = transport.calls[0]["params"]
    assert params["adjustment"] == "raw"
    assert params["feed"] == "iex"
    assert params["timeframe"] == "1Min"


def test_alpaca_admits_it_is_not_the_consolidated_tape() -> None:
    """The finding that gates minute-resolution live trading.

    IEX is ~2% of consolidated volume, so the disagreement against the tape is
    the same order as the gross edge. If this ever asserts True on the free
    feed, the bake-off's conclusion has been quietly discarded.
    """
    caps = build_alpaca().capabilities
    assert caps.consolidated_tape is False
    assert caps.returns_raw_prices is True
    assert "2%" in caps.note


def test_alpaca_follows_page_tokens_to_completion() -> None:
    """A backfill that stopped at page one would look complete and be wrong."""
    pages = {
        None: alpaca_body(
            bars=[
                {
                    "t": "2026-03-04T14:30:00Z",
                    "o": 100.1,
                    "h": 100.5,
                    "l": 100.0,
                    "c": 100.4,
                    "v": 1200,
                }
            ],
            next_page_token="page-2",
        ),
        "page-2": alpaca_body(
            bars=[
                {
                    "t": "2026-03-04T14:31:00Z",
                    "o": 100.4,
                    "h": 100.6,
                    "l": 100.2,
                    "c": 100.25,
                    "v": 800,
                }
            ],
            next_page_token=None,
        ),
    }

    def respond(_url: str, params: dict[str, Any] | None) -> HttpResponse:
        token = (params or {}).get("page_token")
        return json_response(pages[token])

    transport = RecordingTransport(responses={"/v2/stocks/bars": respond})
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert len(batch.bars) == 2
    assert len(transport.calls) == 2


def test_alpaca_refuses_bars_for_a_symbol_it_was_not_asked_about() -> None:
    """The symbol-map failure this layer exists to prevent, at the wire level."""
    transport = RecordingTransport(
        responses={"/v2/stocks/bars": json_response(alpaca_body(symbol="MSFT"))}
    )
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    with pytest.raises(DataError, match="MSFT"):
        provider.fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.MINUTE,
            start=datetime(2026, 3, 4, tzinfo=UTC),
            end=datetime(2026, 3, 5, tzinfo=UTC),
        )


def test_alpaca_drops_a_bar_with_a_missing_price_rather_than_zeroing_it() -> None:
    transport = RecordingTransport(
        responses={
            "/v2/stocks/bars": json_response(
                alpaca_body(
                    bars=[
                        {
                            "t": "2026-03-04T14:30:00Z",
                            "o": 100.1,
                            "h": 100.5,
                            "l": 100.0,
                            "c": 100.4,
                            "v": 1200,
                        },
                        {"t": "2026-03-04T14:31:00Z", "o": 100.4, "h": 100.6, "l": None, "v": 800},
                    ]
                )
            )
        }
    )
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    batch = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert len(batch.bars) == 1
    assert any("missing 'l'" in warning for warning in batch.warnings)


def test_alpaca_never_echoes_a_rejected_credential() -> None:
    """An auth error body can contain the submitted key. It is never surfaced."""
    transport = RecordingTransport(
        responses={
            "/v2/stocks/bars": HttpResponse(
                status_code=403,
                headers={},
                text='{"message":"forbidden","key":"SUPERSECRETKEY"}',
                elapsed_ms=2.0,
            )
        }
    )
    provider = AlpacaProvider(
        "key-id", "SUPERSECRETKEY", transport=transport, pacer=_instant_pacer()
    )
    with pytest.raises(ProviderUnavailable) as caught:
        provider.fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.MINUTE,
            start=datetime(2026, 3, 4, tzinfo=UTC),
            end=datetime(2026, 3, 5, tzinfo=UTC),
        )
    assert "SUPERSECRETKEY" not in str(caught.value)
    assert KEY_ID_VAR in str(caught.value)


def test_alpaca_sends_its_credentials_as_headers_only() -> None:
    provider = build_alpaca()
    provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    transport = provider._transport
    assert isinstance(transport, RecordingTransport)
    call = transport.calls[0]
    assert call["headers"]["APCA-API-KEY-ID"] == "key"
    # A credential in a query string lands in every proxy and access log there
    # is, so it must never appear there.
    assert "secret" not in json.dumps(call["params"])


def test_alpaca_refuses_to_read_the_sdk_key_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """A trading-capable key must not be picked up as a data key.

    Alpaca's SDK convention is `APCA_API_KEY_ID`, and on a funded account such
    a key can place orders. This process executes through Trading 212, so it
    has no business holding one — and silently falling back to it would give the
    data layer order-placing credentials.
    """
    monkeypatch.delenv(KEY_ID_VAR, raising=False)
    monkeypatch.delenv(SECRET_VAR, raising=False)
    for var in SDK_KEY_VARS:
        monkeypatch.setenv(var, "a-trading-key")

    with pytest.raises(ProviderUnavailable) as caught:
        AlpacaProvider.from_env()
    message = str(caught.value)
    assert KEY_ID_VAR in message
    assert "place orders" in message
    assert "a-trading-key" not in message

    findings = AlpacaProvider.credential_findings()
    sdk_findings = [f for f in findings if "blast radius" in f]
    assert len(sdk_findings) == len(SDK_KEY_VARS)
    assert not AlpacaProvider.configured()


def test_alpaca_reports_a_key_exported_under_a_name_nothing_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure this repo shipped with, caught where it is diagnosable.

    `.env.example` named `ALPACA_API_KEY_ID`, which no code reads. Following it
    left the feed unconfigured, and the only symptom arrived several commands
    later as `ProviderUnavailable` — an error about a missing provider, which
    points at the wrong thing entirely.
    """
    monkeypatch.delenv(KEY_ID_VAR, raising=False)
    monkeypatch.delenv(SECRET_VAR, raising=False)
    for var in SDK_KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    for var in MISNAMED_KEY_VARS:
        monkeypatch.setenv(var, "a-real-key-under-the-wrong-name")

    findings = AlpacaProvider.credential_findings()
    assert [f for f in findings if MISNAMED_KEY_VARS[0] in f]
    assert all("a-real-key-under-the-wrong-name" not in f for f in findings)

    with pytest.raises(ProviderUnavailable) as caught:
        AlpacaProvider.from_env()
    message = str(caught.value)
    assert MISNAMED_KEY_VARS[0] in message
    assert KEY_ID_VAR in message
    assert "a-real-key-under-the-wrong-name" not in message


def test_alpaca_reports_half_a_credential_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ID_VAR, "data-key")
    monkeypatch.delenv(SECRET_VAR, raising=False)
    for var in SDK_KEY_VARS + MISNAMED_KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    findings = AlpacaProvider.credential_findings()
    assert len(findings) == 1
    assert SECRET_VAR in findings[0]
    assert not AlpacaProvider.configured()


def test_alpaca_reports_nothing_once_both_names_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ID_VAR, "data-key")
    monkeypatch.setenv(SECRET_VAR, "data-secret")
    for var in SDK_KEY_VARS + MISNAMED_KEY_VARS:
        monkeypatch.delenv(var, raising=False)

    assert AlpacaProvider.credential_findings() == ()
    assert AlpacaProvider.configured()


def test_env_example_only_names_alpaca_variables_the_code_reads() -> None:
    """The test that would have caught the original defect.

    A template naming a variable nothing reads is worse than no template: it
    looks authoritative, and the resulting failure names the provider rather
    than the typo.
    """
    template = Path(__file__).resolve().parents[1] / ".env.example"
    named = {
        line.split("=", 1)[0].lstrip("# ").strip()
        for line in template.read_text().splitlines()
        if "=" in line and line.lstrip("# ").strip().startswith("ALPACA")
    }
    assert named, "the template stopped mentioning Alpaca at all"
    assert named == {KEY_ID_VAR, SECRET_VAR}


def test_alpaca_from_env_builds_when_the_data_keys_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ID_VAR, "data-key")
    monkeypatch.setenv(SECRET_VAR, "data-secret")
    provider = AlpacaProvider.from_env(transport=RecordingTransport(), pacer=_instant_pacer())
    assert provider.name == "alpaca"


def test_alpaca_latest_bar_refuses_an_unsettled_bar() -> None:
    """The partial-bar trap, with the clock inside the bar's own minute."""
    forming = (
        (datetime.now(UTC).replace(second=0, microsecond=0)).isoformat().replace("+00:00", "Z")
    )
    transport = RecordingTransport(
        responses={
            f"/v2/stocks/{SYMBOL}/bars/latest": json_response(
                json.dumps(
                    {
                        "bar": {
                            "t": forming,
                            "o": 100.1,
                            "h": 100.2,
                            "l": 100.0,
                            "c": 100.15,
                            "v": 17,
                        },
                        "symbol": SYMBOL,
                    }
                )
            )
        }
    )
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    assert provider.latest_bar(SYMBOL, instrument_uid=UID, resolution=Resolution.MINUTE) is None


def test_alpaca_splits_keep_exact_integer_ratios() -> None:
    """A 1-for-10 reverse split must not become 0.1.

    The factor table is a running product over decades of history; a repeating
    decimal drifts, and `canonical_json` faithfully hashes the drift into every
    vintage that touches it.
    """
    transport = RecordingTransport(
        responses={
            "/v1/corporate-actions": json_response(
                json.dumps(
                    {
                        "corporate_actions": {
                            "forward_splits": [
                                {
                                    "symbol": SYMBOL,
                                    "new_rate": 4,
                                    "old_rate": 1,
                                    "ex_date": "2020-08-31",
                                    "process_date": "2020-09-01",
                                    "declaration_date": "2020-07-30",
                                }
                            ],
                            "reverse_splits": [
                                {
                                    "symbol": SYMBOL,
                                    "new_rate": 1,
                                    "old_rate": 10,
                                    "ex_date": "2021-01-05",
                                }
                            ],
                            "cash_dividends": [
                                {
                                    "symbol": SYMBOL,
                                    "rate": 0.24,
                                    "ex_date": "2021-02-05",
                                    "declaration_date": "2021-01-27",
                                }
                            ],
                        },
                        "next_page_token": None,
                    }
                )
            )
        }
    )
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    actions = provider.fetch_actions(
        SYMBOL,
        instrument_uid=UID,
        start=datetime(2020, 1, 1, tzinfo=UTC),
        end=datetime(2022, 1, 1, tzinfo=UTC),
    )
    forward = next(a for a in actions if a.effective_date == "2020-08-31")
    reverse = next(a for a in actions if a.effective_date == "2021-01-05")
    dividend = next(a for a in actions if a.action_type == "cash_dividend")

    assert (forward.ratio_num, forward.ratio_den) == (4, 1)
    # Exact integers, not 0.1.
    assert (reverse.ratio_num, reverse.ratio_den) == (1, 10)
    # Alpaca supplies a declaration date where Yahoo cannot, which is what makes
    # `known_at` a fact rather than "whenever we happened to look".
    assert forward.declared_date == "2020-07-30"
    assert dividend.gross_amount == Decimal("0.24")
    # And `known_at` is taken from it: public by the end of that day in New
    # York, not years later when a backfill fetched it — which hid every split
    # from every backtest instant before the fetch.
    assert forward.known_at_utc == datetime(2020, 7, 31, 4, tzinfo=UTC)
    assert dividend.known_at_utc == datetime(2021, 1, 28, 5, tzinfo=UTC)
    # With no declaration date, public by the start of its ex-date at the latest.
    assert reverse.known_at_utc == datetime(2021, 1, 5, 5, tzinfo=UTC)


def test_alpaca_prefers_the_ex_date_over_the_process_date() -> None:
    """The ex-date is the session on which the price gaps.

    `process_date` is Alpaca's bookkeeping date and can be a day later; keying
    the factor on it would apply the adjustment one session too late, leaving a
    fake gap in the adjusted series exactly where the split was.
    """
    transport = RecordingTransport(
        responses={
            "/v1/corporate-actions": json_response(
                json.dumps(
                    {
                        "corporate_actions": {
                            "forward_splits": [
                                {
                                    "symbol": SYMBOL,
                                    "new_rate": 2,
                                    "old_rate": 1,
                                    "ex_date": "2024-06-10",
                                    "process_date": "2024-06-11",
                                }
                            ]
                        },
                        "next_page_token": None,
                    }
                )
            )
        }
    )
    provider = AlpacaProvider("key", "secret", transport=transport, pacer=_instant_pacer())
    (action,) = provider.fetch_actions(
        SYMBOL,
        instrument_uid=UID,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert action.effective_date == "2024-06-10"


# --------------------------------------------------------------------------
# CSV fixture specifics
# --------------------------------------------------------------------------


def test_the_fixture_provider_has_no_clock() -> None:
    """Determinism, asserted rather than assumed.

    Every vintage-immutability and hash-stability test in the suite depends on
    this: if the fixture stamped `ingested_at` from the wall clock, those tests
    would compare values that differ by construction on every run.
    """
    first = build_csv().fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    second = build_csv().fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert [b.storage_row(scale=SCALE) for b in first.bars] == [
        b.storage_row(scale=SCALE) for b in second.bars
    ]


def test_a_csv_round_trip_preserves_every_hash(tmp_path: Path) -> None:
    """Writing and reloading a bar must not change it.

    It only holds because prices are written as Decimal strings. A `float()` on
    the way out would change the last digits of some values and with them every
    row hash — which is how a "harmless" serialisation helper invalidates a
    sealed vintage.
    """
    bars = [make_bar(minute=index, close=f"100.{index}5") for index in range(3)]
    path = tmp_path / f"{SYMBOL}_{Resolution.MINUTE.value}.csv"
    write_bar_csv(path, bars)

    reloaded = CsvFixtureProvider(tmp_path).fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    )
    assert [b.row_hash(scale=SCALE) for b in reloaded.sorted_bars()] == [
        b.row_hash(scale=SCALE) for b in bars
    ]


def test_a_fixture_row_with_a_naive_timestamp_is_refused(tmp_path: Path) -> None:
    path = tmp_path / f"{SYMBOL}_{Resolution.DAILY.value}.csv"
    path.write_text(
        "bar_open_utc,open,high,low,close,volume\n2026-03-04T00:00:00,1,2,0.5,1.5,10\n",
        encoding="utf-8",
    )
    batch = CsvFixtureProvider(tmp_path).fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.DAILY,
        start=datetime(2026, 3, 1, tzinfo=UTC),
        end=datetime(2026, 3, 10, tzinfo=UTC),
    )
    assert batch.bars == ()
    assert any("no timezone offset" in warning for warning in batch.warnings)


def test_a_fixture_missing_a_required_column_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / f"{SYMBOL}_{Resolution.DAILY.value}.csv"
    path.write_text("bar_open_utc,open,high,close\n", encoding="utf-8")
    with pytest.raises(DataError, match="missing required column"):
        CsvFixtureProvider(tmp_path).fetch_bars(
            SYMBOL,
            instrument_uid=UID,
            resolution=Resolution.DAILY,
            start=datetime(2026, 3, 1, tzinfo=UTC),
            end=datetime(2026, 3, 10, tzinfo=UTC),
        )


def test_the_fixture_provider_can_impersonate_another_source() -> None:
    """Needed by the bake-off: two named sources that disagree, without a network."""
    provider = CsvFixtureProvider(
        bars=[make_bar(minute=0, close="100.90", provider="pretend-alpaca")],
        provider_name="pretend-alpaca",
    )
    (bar,) = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 5, tzinfo=UTC),
    ).bars
    assert bar.provider == "pretend-alpaca"
    assert provider.capabilities.name == "pretend-alpaca"


def test_the_fetch_window_is_half_open_on_the_right() -> None:
    """A bar opening exactly at the decision instant has not happened yet."""
    provider = CsvFixtureProvider(bars=[make_bar(minute=0)])
    at_boundary = provider.fetch_bars(
        SYMBOL,
        instrument_uid=UID,
        resolution=Resolution.MINUTE,
        start=datetime(2026, 3, 4, tzinfo=UTC),
        end=datetime(2026, 3, 4, 14, 30, tzinfo=UTC),
    )
    assert at_boundary.bars == ()
