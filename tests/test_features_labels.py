from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.config import ExperimentConfig
from pressure_graph.features import (
    align_funding,
    align_open_interest,
    build_feature_table,
    ensure_bar_times,
    rolling_percentile_current_vs_prior,
)
from pressure_graph.labels import add_future_labels


def test_kline_ordering_and_bar_times() -> None:
    raw = pd.DataFrame(
        {
            "exchange": ["bybit", "bybit"],
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "bar_open_time": pd.to_datetime(
                ["2025-01-01 00:15:00Z", "2025-01-01 00:00:00Z"]
            ),
            "open": [101, 100],
            "high": [102, 101],
            "low": [100, 99],
            "close": [101, 100],
            "volume": [10, 10],
            "turnover": [1010, 1000],
        }
    )
    out = ensure_bar_times(raw)
    assert out["bar_open_time"].tolist() == sorted(out["bar_open_time"].tolist())
    assert (out["bar_close_time"] - out["bar_open_time"]).eq(pd.Timedelta(minutes=15)).all()
    assert out.iloc[0]["feature_time"] == out.iloc[0]["bar_close_time"]
    assert out.iloc[0]["entry_time"] == out.iloc[1]["bar_open_time"]


def test_rolling_percentile_is_current_vs_prior_window() -> None:
    values = pd.Series([1.0, 3.0, 2.0, 4.0])
    pct = rolling_percentile_current_vs_prior(values, window=2, min_periods=2)
    assert np.isnan(pct.iloc[0])
    assert np.isnan(pct.iloc[1])
    assert pct.iloc[2] == 50.0
    assert pct.iloc[3] == 100.0


def test_asof_join_uses_only_settled_funding_and_drops_stale() -> None:
    cfg = ExperimentConfig()
    bars = ensure_bar_times(
        pd.DataFrame(
            {
                "exchange": ["bybit", "bybit", "bybit"],
                "symbol": ["ETHUSDT", "ETHUSDT", "ETHUSDT"],
                "bar_open_time": pd.to_datetime(
                    ["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z", "2025-01-01 03:00:00Z"]
                ),
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [10, 10, 10],
                "turnover": [1000, 1010, 1020],
            }
        )
    )
    funding = pd.DataFrame(
        {
            "exchange": ["bybit", "bybit"],
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "funding_time": pd.to_datetime(["2025-01-01 00:10:00Z", "2025-01-01 04:00:00Z"]),
            "funding_rate_settled": [0.001, 0.002],
        }
    )
    instruments = pd.DataFrame(
        {"symbol": ["ETHUSDT"], "funding_interval_minutes": [60], "launch_time": [pd.NaT]}
    )
    out = align_funding(bars, funding, instruments, cfg)
    assert out.iloc[0]["funding_time"] <= out.iloc[0]["feature_time"]
    assert out.iloc[1]["funding_rate_settled"] == 0.001
    assert pd.isna(out.iloc[2]["funding_rate_settled"])


def test_open_interest_asof_never_uses_future_and_builds_value() -> None:
    bars = ensure_bar_times(
        pd.DataFrame(
            {
                "exchange": ["bybit", "bybit"],
                "symbol": ["ETHUSDT", "ETHUSDT"],
                "bar_open_time": pd.to_datetime(["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z"]),
                "open": [100, 110],
                "high": [101, 111],
                "low": [99, 109],
                "close": [100, 110],
                "volume": [10, 10],
                "turnover": [1000, 1100],
            }
        )
    )
    oi = pd.DataFrame(
        {
            "exchange": ["bybit", "bybit"],
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "oi_time": pd.to_datetime(["2025-01-01 00:14:00Z", "2025-01-01 01:00:00Z"]),
            "oi_base": [5.0, 999.0],
        }
    )
    out = align_open_interest(bars, oi)
    assert out.iloc[0]["oi_base"] == 5.0
    assert out.iloc[1]["oi_base"] == 5.0
    assert out.iloc[1]["oi_value_usdt"] == 5.0 * 110


def test_future_labels_start_from_next_bar() -> None:
    df = pd.DataFrame(
        {
            "exchange": ["bybit"] * 3,
            "symbol": ["ETHUSDT"] * 3,
            "bar_open_time": pd.to_datetime(
                ["2025-01-01 00:00:00Z", "2025-01-01 00:15:00Z", "2025-01-01 00:30:00Z"]
            ),
            "bar_close_time": pd.to_datetime(
                ["2025-01-01 00:15:00Z", "2025-01-01 00:30:00Z", "2025-01-01 00:45:00Z"]
            ),
            "open": [100, 100, 100],
            "high": [200, 102, 104],
            "low": [99, 98, 99],
            "close": [100, 100, 100],
            "volume": [1, 1, 1],
            "turnover": [100, 100, 100],
        }
    )
    out = add_future_labels(df, {"4h": 2})
    assert out.iloc[0]["future_max_up_4h"] == pytest.approx(0.04)
    assert bool(out.iloc[0]["hit_3pct_4h"])


def test_build_feature_table_keeps_no_future_funding_or_oi() -> None:
    cfg = ExperimentConfig()
    klines = pd.DataFrame(
        {
            "exchange": ["bybit"] * 20,
            "symbol": ["BTCUSDT"] * 20,
            "bar_open_time": pd.date_range("2025-01-01", periods=20, freq="15min", tz="UTC"),
            "open": np.linspace(100, 119, 20),
            "high": np.linspace(101, 120, 20),
            "low": np.linspace(99, 118, 20),
            "close": np.linspace(100, 119, 20),
            "volume": np.ones(20),
            "turnover": np.linspace(100, 119, 20),
        }
    )
    funding = pd.DataFrame(
        {
            "exchange": ["bybit"],
            "symbol": ["BTCUSDT"],
            "funding_time": [pd.Timestamp("2025-01-01 00:00:00Z")],
            "funding_rate_settled": [0.001],
        }
    )
    oi = pd.DataFrame(
        {
            "exchange": ["bybit"],
            "symbol": ["BTCUSDT"],
            "oi_time": [pd.Timestamp("2025-01-01 00:00:00Z")],
            "oi_base": [100.0],
        }
    )
    instruments = pd.DataFrame(
        {"symbol": ["BTCUSDT"], "funding_interval_minutes": [480], "launch_time": [pd.NaT]}
    )
    out = build_feature_table(klines, funding, oi, instruments, cfg)
    assert (out.loc[out["funding_time"].notna(), "funding_time"] <= out["feature_time"]).all()
    assert (out.loc[out["oi_time"].notna(), "oi_time"] <= out["feature_time"]).all()
