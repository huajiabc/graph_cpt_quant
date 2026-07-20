from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v129_causal_risk_parity_combination import (
    V129Config,
    build_v129_panel,
)


def test_risk_parity_uses_only_prior_window_and_clips_weight(tmp_path) -> None:
    entry = pd.date_range("2025-08-04", periods=4, freq="7D", tz="UTC")
    source = pd.DataFrame(
        {
            "entry_time": entry,
            "exit_time": entry + pd.Timedelta(days=7),
            "month_start": [timestamp.replace(day=1) for timestamp in entry],
            "period": ["development"] * 4,
            "tg1_return": [0.001, -0.001, 0.001, 1.0],
            "p2_return": [0.1, -0.1, 0.1, 0.0],
            "p2_trades": [1] * 4,
            "combined_return": [0.0] * 4,
        }
    )
    path = tmp_path / "panel.parquet"
    source.to_parquet(path, index=False)
    panel = build_v129_panel(
        V129Config(panel_path=path, volatility_weeks=3)
    )
    assert panel.loc[0:2, "tg1_weight"].eq(0.5).all()
    assert panel.loc[3, "tg1_weight"] == 0.75
