"""Autonomous long-only equity trading bot for Trading 212.

Layout (milestones in the build plan):

    core/     canonical serialisation, time, errors — shared primitives
    config/   the control layer: hard limits, hash pinning                (M0)
    ledger/   hash-chained append-only event log                          (M0)
    ops/      kill switch, run state, watchdog, journal, dashboard     (M0/M4/M8)
    broker/   Trading 212 adapter, rate governor, reconciler             (M1)
    data/     market data providers, point-in-time bar store             (M2)
    features/ the single feature pipeline used by every mode             (M3)
    backtest/ non-cheating backtester, cost model                        (M3)
    strategy/ Strategy interface, declarative DSL, ML, RL stub        (M3/M6/M7)
    risk/     the risk engine every order must pass through              (M4)
    engine/   the live tick loop, intent write-ahead log                 (M4)
    registry/ strategy registry, promotion ladder                        (M5)
    research/ trial accounting, sealed holdout, spec search           (M5/M6)
    portfolio/capital allocator, correlation caps, decay detection       (M5)
"""

__version__ = "0.1.0"
