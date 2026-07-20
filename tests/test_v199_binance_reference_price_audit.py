from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.reports.v199_binance_reference_price_audit import (
    load_reference_ohlc_panels,
    summarize_reference_relationships,
)


def test_reference_loader_keeps_exact_last_duplicate(tmp_path: Path) -> None:
    times = pd.to_datetime(
        ["2026-01-01 00:15Z", "2026-01-01 00:15Z", "2026-01-01 00:30Z"]
    )
    pd.DataFrame(
        {
            "feature_time": times,
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
        }
    ).to_parquet(tmp_path / "BTCUSDT.parquet", index=False)
    panels = load_reference_ohlc_panels(tmp_path)
    assert list(panels) == ["open", "high", "low", "close"]
    assert len(panels["close"]) == 2
    assert panels["close"].iloc[0, 0] == 11.5


def test_relationship_summary_recognizes_exact_implied_premium() -> None:
    times = pd.date_range("2026-01-01 00:15", periods=4, freq="15min", tz="UTC")
    columns = ["BTCUSDT", "ETHUSDT"]
    index = pd.DataFrame(
        [[100.0, 50.0], [101.0, 51.0], [99.0, 49.0], [102.0, 52.0]],
        index=times,
        columns=columns,
    )
    premium = pd.DataFrame(
        [[-0.001, 0.002], [0.001, -0.002], [0.003, 0.001], [-0.002, 0.003]],
        index=times,
        columns=columns,
    )
    mark = index.mul(1.0 + premium)
    aggregate, by_symbol = summarize_reference_relationships(
        mark, mark, index, premium
    )
    assert aggregate.loc[0, "aligned_points"] == 8
    assert np.isclose(aggregate.loc[0, "implied_premium_correlation"], 1.0)
    assert aggregate.loc[0, "median_abs_implied_premium_difference"] < 1e-12
    assert len(by_symbol) == 2
