from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v127_broader_cross_venue_carry_bucket import (
    _config,
    build_v127_portfolio,
)


def test_top12_requires_full_positive_bucket() -> None:
    cfg = _config()
    rows = []
    for count, entry in (
        (11, pd.Timestamp("2025-08-04", tz="UTC")),
        (12, pd.Timestamp("2025-08-11", tz="UTC")),
    ):
        for index in range(count):
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": entry + pd.Timedelta(days=7),
                    "month_start": entry.replace(day=1),
                    "period": "development",
                    "symbol": f"S{index:02d}",
                    "score_30d": float(count - index),
                    "pair_gross_return": 0.01,
                    "bybit_return": 0.02,
                    "binance_return": 0.01,
                    "price_basis_return": 0.01,
                    "funding_spread_return": 0.0,
                }
            )
    portfolio = build_v127_portfolio(pd.DataFrame(rows), cfg)
    assert len(portfolio) == 1
    assert len(portfolio.loc[0, "selected_symbols"].split("|")) == 12
