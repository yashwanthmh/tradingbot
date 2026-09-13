"""The rate governor.

Driven by a fake clock rather than real sleeps, so the properties that matter —
never exceeding a limit, risk-reducing calls getting through under saturation,
a cold boot not bursting — are checked deterministically instead of by timing.

Trading 212's limits are per-account, so a 429 is not a transient annoyance: it
can arrive in the middle of reconciliation, which is the one moment the system
most needs to be able to read state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tb.broker.t212.endpoints import Endpoint, spec_for
from tb.broker.t212.ratelimit import (
    NullGovernor,
    RateGovernor,
    RateLimitHeaders,
    RateLimitTimeout,
)


class FakeClock:
    """A clock that only moves when something sleeps.

    Lets a test assert "this burst would have needed 58 seconds" without waiting
    58 seconds, and makes the arithmetic exact rather than approximately right.
    """

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    @property
    def total_slept(self) -> float:
        return sum(self.slept)

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _governor(clock: FakeClock, **kwargs: object) -> RateGovernor:
    return RateGovernor(clock=clock.time, sleep=clock.sleep, **kwargs)  # type: ignore[arg-type]


class TestColdBoot:
    def test_a_cold_boot_assumes_the_budget_is_spent(self) -> None:
        """The scenario this defends.

        A process spends its budget and dies. It restarts a second later. If it
        assumed a full bucket it would burst straight into a 429 — during
        reconciliation, before it knows what the account holds.
        """
        clock = FakeClock()
        governor = _governor(clock)
        spec = spec_for(Endpoint.PORTFOLIO)

        waited = governor.acquire(Endpoint.PORTFOLIO)
        assert waited >= spec.period_seconds / spec.capacity

    def test_opting_out_of_the_pessimistic_boot_starts_full(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        assert governor.acquire(Endpoint.PORTFOLIO) == 0.0

    def test_persisted_state_is_restored(self, tmp_path: Path) -> None:
        clock = FakeClock()
        state = tmp_path / "ratelimit.json"

        first = _governor(clock, state_path=state)
        first.acquire(Endpoint.ORDER_MARKET)
        first.acquire(Endpoint.ORDER_MARKET)
        assert state.exists()

        # A restart with no wall-clock time passing must not regain the tokens
        # the previous process spent.
        second = _governor(clock, state_path=state)
        spec = spec_for(Endpoint.ORDER_MARKET)
        remaining = 0
        while second.try_acquire(Endpoint.ORDER_MARKET):
            remaining += 1
        assert remaining <= spec.capacity - spec.reserve_for_risk_reducing - 2

    def test_a_corrupt_state_file_is_treated_as_no_state(self, tmp_path: Path) -> None:
        """Unreadable state means assume spent, not assume full."""
        state = tmp_path / "ratelimit.json"
        state.write_text("this is not json {{{", encoding="utf-8")
        clock = FakeClock()
        governor = _governor(clock, state_path=state)
        assert governor.acquire(Endpoint.PORTFOLIO) > 0.0

    def test_the_spend_is_persisted_immediately(self, tmp_path: Path) -> None:
        """Losing a refill is safe; losing a spend is not.

        So state is written when a token is consumed, not on a timer — a crash
        must never let the next boot think it has budget it already used.
        """
        clock = FakeClock()
        state = tmp_path / "ratelimit.json"
        governor = _governor(clock, state_path=state)
        governor.acquire(Endpoint.ORDER_MARKET)

        body = json.loads(state.read_text(encoding="utf-8"))
        tokens = body["buckets"][Endpoint.ORDER_MARKET.value]["tokens"]
        assert tokens < spec_for(Endpoint.ORDER_MARKET).capacity

    def test_a_write_failure_does_not_stop_trading(self, tmp_path: Path) -> None:
        """A full disk must not become an inability to place a protective stop."""
        clock = FakeClock()
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("i am a file", encoding="utf-8")
        governor = _governor(clock, state_path=blocked / "ratelimit.json")
        governor.acquire(Endpoint.ORDER_MARKET)  # must not raise


class TestLimitsAreRespected:
    def test_a_burst_never_exceeds_the_configured_rate(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        spec = spec_for(Endpoint.ORDER_STOP)

        calls = 10
        for _ in range(calls):
            governor.acquire(Endpoint.ORDER_STOP)

        elapsed = clock.now - 1_000_000.0
        # The first call is free; the rest each pay a period.
        expected = (calls - 1) * (spec.period_seconds / spec.capacity)
        assert elapsed >= expected

    def test_protective_stop_arithmetic_matches_the_documented_claim(self) -> None:
        """25 symbols really do need ~50 seconds of stop-order budget.

        This number drives `max_universe_symbols`, so it is asserted rather
        than left in a comment where it can rot.
        """
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        for _ in range(25):
            governor.acquire(Endpoint.ORDER_STOP, risk_reducing=True)
        assert 46.0 <= (clock.now - 1_000_000.0) <= 54.0

    def test_buckets_are_independent(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        governor.acquire(Endpoint.PORTFOLIO)
        # Spending the portfolio budget must not delay an order-list read.
        assert governor.wait_estimate(Endpoint.ORDERS_LIST) == 0.0

    def test_refill_is_proportional_to_elapsed_time(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        for _ in range(spec_for(Endpoint.ORDER_MARKET).capacity):
            governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True)
        assert not governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True)

        # 60s / 50 tokens = 1.2s per token.
        clock.advance(2.5)
        assert governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True)

    def test_a_backwards_clock_does_not_remove_tokens(self) -> None:
        """NTP steps and VM migrations move wall-clock time backwards.

        Crediting the negative elapsed time would *subtract* tokens, stalling
        the system for as long as the step.
        """
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        before = governor.wait_estimate(Endpoint.PORTFOLIO)
        clock.advance(-3600.0)
        assert governor.wait_estimate(Endpoint.PORTFOLIO) == before

    @settings(max_examples=40, deadline=None)
    @given(
        st.lists(
            st.sampled_from([Endpoint.PORTFOLIO, Endpoint.ORDERS_LIST, Endpoint.ORDER_STOP]),
            min_size=1,
            max_size=20,
        )
    )
    def test_no_endpoint_is_ever_over_budget(self, sequence: list[Endpoint]) -> None:
        """Property: for any interleaving, each bucket stays within its rate."""
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)

        taken: dict[Endpoint, list[float]] = {}
        for endpoint in sequence:
            governor.acquire(endpoint)
            taken.setdefault(endpoint, []).append(clock.now)

        for endpoint, times in taken.items():
            spec = spec_for(endpoint)
            window = spec.period_seconds
            for index, stamp in enumerate(times):
                in_window = [t for t in times[: index + 1] if t > stamp - window]
                assert len(in_window) <= spec.capacity, (
                    f"{endpoint.value}: {len(in_window)} calls inside a "
                    f"{window}s window, limit {spec.capacity}"
                )


class TestRiskReducingPriority:
    def test_the_reserve_keeps_a_flatten_possible(self) -> None:
        """A runaway entry loop must not be able to spend the exit budget.

        Market orders get 50/60s with 10 reserved. After the risk-increasing
        callers have taken everything available to them, a risk-reducing call
        must still go through immediately.
        """
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        spec = spec_for(Endpoint.ORDER_MARKET)

        entries = 0
        while governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=False):
            entries += 1

        assert entries == spec.capacity - spec.reserve_for_risk_reducing
        assert governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True)

    def test_the_reserve_is_exactly_as_large_as_configured(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        spec = spec_for(Endpoint.ORDER_MARKET)

        while governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=False):
            pass
        exits = 0
        while governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True):
            exits += 1
        assert exits == spec.reserve_for_risk_reducing

    def test_a_capacity_one_bucket_has_nothing_to_reserve(self) -> None:
        """Which is why priority ordering exists as well as the reserve.

        Most Trading 212 limits are one call per period. You cannot reserve a
        fraction of a single token, so on those endpoints the protection has to
        come from ordering among waiters instead.
        """
        assert spec_for(Endpoint.ORDER_STOP).capacity == 1
        assert spec_for(Endpoint.ORDER_STOP).reserve_for_risk_reducing == 0

    def test_a_risk_reducing_call_is_not_penalised_by_the_reserve(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        assert governor.wait_estimate(Endpoint.ORDER_MARKET, risk_reducing=True) == 0.0


class TestServerAuthority:
    def test_the_servers_remaining_count_overrides_a_more_optimistic_local_one(self) -> None:
        """Our count is an inference; theirs returns the 429."""
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        governor.observe_headers(
            Endpoint.ORDER_MARKET,
            RateLimitHeaders(limit=50, period_seconds=60, remaining=0),
        )
        assert not governor.try_acquire(Endpoint.ORDER_MARKET, risk_reducing=True)

    def test_a_more_optimistic_server_count_is_not_adopted(self) -> None:
        """The pessimistic view wins in both directions.

        A server saying we have more left than we think may simply not have
        counted a request already in flight.
        """
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=True)
        governor.observe_headers(
            Endpoint.PORTFOLIO, RateLimitHeaders(limit=1, period_seconds=1, remaining=1)
        )
        assert governor.wait_estimate(Endpoint.PORTFOLIO) > 0.0

    def test_observed_limits_are_reported_for_the_probe(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        governor.observe_headers(
            Endpoint.PORTFOLIO, RateLimitHeaders(limit=5, period_seconds=1, remaining=5)
        )
        rows = {r["endpoint"]: r for r in governor.observations()}
        row = rows[Endpoint.PORTFOLIO.value]
        assert row["observed_limit"] == 5
        assert row["agrees"] is False
        assert any("configured 1/1s" in note for note in governor.disagreements())

    def test_agreement_is_reported_when_the_table_is_right(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        spec = spec_for(Endpoint.PORTFOLIO)
        governor.observe_headers(
            Endpoint.PORTFOLIO,
            RateLimitHeaders(
                limit=spec.capacity, period_seconds=int(spec.period_seconds), remaining=1
            ),
        )
        assert governor.disagreements() == []

    def test_uninformative_headers_are_ignored(self) -> None:
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        governor.observe_headers(Endpoint.PORTFOLIO, RateLimitHeaders())
        assert governor.wait_estimate(Endpoint.PORTFOLIO) == 0.0


class TestHeaderParsing:
    def test_headers_are_parsed_case_insensitively(self) -> None:
        parsed = RateLimitHeaders.from_response_headers(
            {
                "X-RateLimit-Limit": "50",
                "x-ratelimit-period": "60",
                "X-RATELIMIT-REMAINING": "7",
                "x-ratelimit-reset": "1700000000.5",
            }
        )
        assert (parsed.limit, parsed.period_seconds, parsed.remaining) == (50, 60, 7)
        assert parsed.reset_epoch == 1700000000.5

    def test_absent_headers_are_not_an_error(self) -> None:
        parsed = RateLimitHeaders.from_response_headers({"content-type": "application/json"})
        assert not parsed.informative

    @pytest.mark.parametrize("value", ["", "not-a-number", "NaN-ish", "  "])
    def test_a_malformed_header_is_treated_as_absent(self, value: str) -> None:
        """A bad header must not be able to crash the client mid-session."""
        parsed = RateLimitHeaders.from_response_headers({"x-ratelimit-limit": value})
        assert parsed.limit is None

    def test_a_float_valued_limit_is_truncated_not_rejected(self) -> None:
        parsed = RateLimitHeaders.from_response_headers({"x-ratelimit-limit": "50.0"})
        assert parsed.limit == 50


class TestRateLimited:
    def test_a_429_empties_the_bucket_and_blocks_until_reset(self) -> None:
        """A 429 proves the local model was wrong.

        Resuming with a stale token count would immediately earn another, so the
        bucket is emptied as well as blocked.
        """
        clock = FakeClock()
        governor = _governor(clock, assume_spent_on_cold_boot=False)
        backoff = governor.note_rate_limited(
            Endpoint.PORTFOLIO, RateLimitHeaders(reset_epoch=clock.now + 30.0)
        )
        assert backoff == pytest.approx(30.0)
        assert not governor.try_acquire(Endpoint.PORTFOLIO, risk_reducing=True)
        assert governor.wait_estimate(Endpoint.PORTFOLIO) == pytest.approx(30.0, abs=0.2)

        clock.advance(31.0)
        assert governor.try_acquire(Endpoint.PORTFOLIO, risk_reducing=True)

    def test_a_429_without_a_reset_header_falls_back_to_the_period(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        backoff = governor.note_rate_limited(
            Endpoint.PORTFOLIO, RateLimitHeaders(period_seconds=45)
        )
        assert backoff == 45.0

    def test_a_429_with_nothing_usable_backs_off_generously(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        assert governor.note_rate_limited(Endpoint.PORTFOLIO, RateLimitHeaders()) == 60.0

    def test_a_past_reset_time_is_not_trusted(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        backoff = governor.note_rate_limited(
            Endpoint.PORTFOLIO, RateLimitHeaders(reset_epoch=clock.now - 100.0)
        )
        assert backoff > 0.0


class TestTimeout:
    def test_a_deadline_raises_rather_than_waiting(self) -> None:
        """Used by M4 to drop an entry whose bar has gone stale.

        Executing a minute-bar signal ninety seconds late is worse than not
        executing it.
        """
        clock = FakeClock()
        governor = _governor(clock)
        with pytest.raises(RateLimitTimeout) as caught:
            governor.acquire(Endpoint.INSTRUMENTS, timeout=1.0)
        assert caught.value.endpoint is Endpoint.INSTRUMENTS
        assert caught.value.needed > 1.0

    def test_a_generous_deadline_succeeds(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        governor.acquire(Endpoint.PORTFOLIO, timeout=600.0)

    def test_try_acquire_never_blocks(self) -> None:
        clock = FakeClock()
        governor = _governor(clock)
        assert governor.try_acquire(Endpoint.INSTRUMENTS) is False
        assert clock.total_slept == 0.0


class TestNullGovernor:
    def test_it_never_waits_and_records_what_was_asked(self) -> None:
        governor = NullGovernor()
        assert governor.acquire(Endpoint.ORDER_MARKET, risk_reducing=True) == 0.0
        assert governor.calls == [(Endpoint.ORDER_MARKET, True)]

    def test_it_is_not_reachable_from_config(self) -> None:
        """Deliberately not selectable by a flag.

        A governor that can be switched off by configuration will eventually be
        switched off in production, so the only way to get one is to construct
        it in code.
        """
        import inspect

        from tb.broker.t212 import client as client_module

        source = inspect.getsource(client_module)
        assert "NullGovernor" not in source
