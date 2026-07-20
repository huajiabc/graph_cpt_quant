from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v130_graph_diversified_cross_venue_carry import (
    V130Config,
    _portfolio_path,
)


def test_one_name_per_community_and_fixed_eighth_weight() -> None:
    entry = pd.Timestamp("2025-08-04", tz="UTC")
    rows = []
    for community in range(6):
        for rank in range(2):
            rows.append(
                {
                    "entry_time": entry,
                    "exit_time": entry + pd.Timedelta(days=7),
                    "month_start": entry.replace(day=1),
                    "period": "development",
                    "community_id": community,
                    "symbol": f"C{community}S{rank}",
                    "score_30d": 2.0 - rank,
                    "pair_gross_return": 0.01,
                    "price_basis_return": 0.0,
                    "funding_spread_return": 0.01,
                }
            )
    path = _portfolio_path(pd.DataFrame(rows), "community_id", V130Config())
    assert path.loc[0, "active_communities"] == 6
    assert path.loc[0, "invested_exposure"] == 0.75
    assert len(path.loc[0, "selected_symbols"].split("|")) == 6
