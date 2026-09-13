"""Concrete market-data providers.

Three implementations of `MarketDataProvider`, with deliberately different
weaknesses — which is the point. The cross-check between them is what catches
the failure mode neither one reports: a free feed returning plausible wrong
numbers rather than an error.

| provider | raw prices | whole market | live-capable | history |
|---|---|---|---|---|
| `alpaca` | yes (`adjustment=raw`) | **no** (IEX, ~2%) | after the bake-off | IEX from ~2016 |
| `yahoo` | **no** (back-adjusted) | yes | no (delayed) | daily decades, minute ~30d |
| `csv_fixture` | yes | n/a | only if named | whatever is in the file |

Neither free provider is good at everything, and the split is systematic:
Alpaca has the prices the venue printed but not the whole market, Yahoo has the
whole market but not the prices it printed. So the ten-year daily backfill
comes from Yahoo, live and recent history from Alpaca, and the disagreement
between them is measured rather than assumed — `tb data bakeoff` turns it into
the arithmetic behind the paid-data decision.

Constructing a provider is deliberately explicit; there is no registry that
maps a config string to a class. A typo in a config file should not be able to
silently swap the feed underneath a strategy.
"""

from __future__ import annotations

from tb.data.providers.alpaca import AlpacaProvider
from tb.data.providers.csv_fixture import CsvFixtureProvider, write_bar_csv
from tb.data.providers.yahoo import YahooProvider

__all__ = [
    "AlpacaProvider",
    "CsvFixtureProvider",
    "YahooProvider",
    "write_bar_csv",
]
