from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v128_tg1_p2_orthogonal_combination import (
    P2_ID,
    V128Config,
    build_v128_weekly_panel,
)


def test_p2_trade_is_slot_normalized_and_combined_half_weighted(tmp_path) -> None:
    entry = pd.Timestamp("2025-08-04", tz="UTC")
    tg1 = pd.DataFrame(
        {
            "entry_time": [entry],
            "exit_time": [entry + pd.Timedelta(days=7)],
            "month_start": [entry.replace(day=1)],
            "period": ["development"],
            "primary_net_return": [0.02],
        }
    )
    p2 = pd.DataFrame(
        {
            "portfolio_id": [P2_ID, "OTHER"],
            "selected": [True, True],
            "trade_id": ["A", "B"],
            "entry_time": [entry + pd.Timedelta(days=1)] * 2,
            "net_return_20bp": [0.08, 1.0],
        }
    )
    tg1_path = tmp_path / "tg1.parquet"
    p2_path = tmp_path / "p2.parquet"
    tg1.to_parquet(tg1_path, index=False)
    p2.to_parquet(p2_path, index=False)
    panel = build_v128_weekly_panel(
        V128Config(tg1_path=tg1_path, p2_path=p2_path)
    )
    assert np.isclose(panel.loc[0, "p2_return"], 0.01)
    assert np.isclose(panel.loc[0, "combined_return"], 0.015)
