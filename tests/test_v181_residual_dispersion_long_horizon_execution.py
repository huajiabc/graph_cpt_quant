from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v181_residual_dispersion_long_horizon_execution import (
    V181Config,
    build_continuous_v181_book,
)


def _signal(source: pd.Timestamp) -> pd.DataFrame:
    laggards = [f"L{i}USDT" for i in range(5)]
    leaders = [f"H{i}USDT" for i in range(5)]
    return pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "laggards": ["|".join(laggards)],
            "leaders": ["|".join(leaders)],
            "spread_beta": [0.0],
        }
    )


def test_continuous_single_sleeve_has_full_entry_and_exit_turnover() -> None:
    source = pd.Timestamp("2026-01-01", tz="UTC")
    index = pd.date_range(source, periods=4, freq="15min", tz="UTC")
    names = ["BTCUSDT", *[f"L{i}USDT" for i in range(5)], *[f"H{i}USDT" for i in range(5)]]
    close = pd.DataFrame(100.0, index=index, columns=names)
    cfg = V181Config(primary_holding_bars=2)
    book, sleeves = build_continuous_v181_book(_signal(source), close, cfg)
    assert len(sleeves) == 1
    assert math.isclose(book["turnover"].sum(), 2.0)
    assert math.isclose(book["primary_cost_return"].sum(), 0.003)
    assert book.loc[0, "active_sleeves"] == 1
    assert book.loc[1, "active_sleeves"] == 1
    assert book.loc[2, "active_sleeves"] == 0


def test_overlapping_identical_sleeves_net_without_extra_target_turnover() -> None:
    source = pd.Timestamp("2026-01-01", tz="UTC")
    index = pd.date_range(source, periods=5, freq="15min", tz="UTC")
    names = ["BTCUSDT", *[f"L{i}USDT" for i in range(5)], *[f"H{i}USDT" for i in range(5)]]
    close = pd.DataFrame(100.0, index=index, columns=names)
    signals = pd.concat(
        [_signal(source), _signal(source + pd.Timedelta(minutes=15))],
        ignore_index=True,
    )
    cfg = V181Config(primary_holding_bars=2)
    book, _ = build_continuous_v181_book(signals, close, cfg)
    assert book["active_sleeves"].max() == 2
    assert math.isclose(book["turnover"].sum(), 2.0)
