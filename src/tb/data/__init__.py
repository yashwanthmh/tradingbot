"""Market data: providers, the point-in-time bar store, and the symbol map.

Trading 212 serves no market data at all, so this package is what makes the
system able to see a price. The symbol map arrives first (M1) because the join
between the data venue and the execution venue is the most dangerous piece of
plumbing here: a mismapped ticker produces a signal that passes every risk
check and buys the wrong company.
"""

from tb.data.symbols import (
    Confidence,
    DisagreementKind,
    PriceComparison,
    SymbolMap,
    SymbolMapping,
    compare_prices,
    derive_mapping,
    parse_ticker,
)

__all__ = [
    "Confidence",
    "DisagreementKind",
    "PriceComparison",
    "SymbolMap",
    "SymbolMapping",
    "compare_prices",
    "derive_mapping",
    "parse_ticker",
]
