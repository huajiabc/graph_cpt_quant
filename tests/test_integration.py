from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.features import build_feature_table
from pressure_graph.labels import add_future_labels
from pressure_graph.paths import add_path_signals
from pressure_graph.reports import write_reports


def _synthetic_klines(symbol: str, days: int = 7) -> pd.DataFrame:
    periods = days * 96
    t = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    drift = 0.0001 if symbol == "BTCUSDT" else 0.0002
    close = 100 * (1 + drift) ** np.arange(periods)
    if symbol == "ETHUSDT":
        close[200:205] *= np.linspace(1.0, 1.05, 5)
    return pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": symbol,
            "bar_open_time": t,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(100, 200, periods),
            "turnover": close * np.linspace(100, 200, periods),
            "universe_static_current_top30": True,
            "universe_dynamic_monthly_top30": True,
        }
    )


def test_two_symbol_seven_day_pipeline(tmp_path) -> None:
    cfg = ExperimentConfig()
    cfg.paths.data_root = tmp_path / "data"
    cfg.paths.report_root = tmp_path / "reports" / "v0"
    cfg.features.min_history_days = 1
    cfg.features.rolling_window_days = 2

    klines = pd.concat([_synthetic_klines("BTCUSDT"), _synthetic_klines("ETHUSDT")])
    funding_times = pd.date_range("2025-01-01", periods=22, freq="8h", tz="UTC")
    funding = pd.DataFrame(
        {
            "exchange": "bybit",
            "symbol": ["BTCUSDT"] * len(funding_times) + ["ETHUSDT"] * len(funding_times),
            "funding_time": list(funding_times) * 2,
            "funding_rate_settled": [0.0001] * (2 * len(funding_times)),
        }
    )
    oi = []
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        t = pd.date_range("2025-01-01", periods=7 * 96, freq="15min", tz="UTC")
        oi.append(
            pd.DataFrame(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "oi_time": t,
                    "oi_base": np.linspace(1000, 1500, len(t)),
                }
            )
        )
    instruments = pd.DataFrame(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "funding_interval_minutes": [480, 480],
            "launch_time": [pd.Timestamp("2020-01-01", tz="UTC")] * 2,
        }
    )
    features = build_feature_table(klines, funding, pd.concat(oi), instruments, cfg)
    labeled = add_future_labels(features)
    signaled = add_path_signals(labeled, cfg)
    outputs = write_reports(signaled, cfg)
    assert not labeled.empty
    assert "future_max_up_4h" in labeled.columns
    assert outputs["path_stats"].exists()
    assert outputs["development_walk_forward"].exists()
    assert outputs["final_holdout"].exists()
    assert outputs["candidate_list"].exists()
