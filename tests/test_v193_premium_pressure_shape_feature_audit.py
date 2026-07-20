import numpy as np
import pandas as pd

from pressure_graph.reports.v193_premium_pressure_shape_feature_audit import (
    V193FeatureConfig,
    build_v193_pressure_features,
)


def test_v193_pressure_features_use_only_shifted_history() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    columns = ["BTCUSDT", "ALTUSDT"]
    price = pd.DataFrame(
        np.linspace(100.0, 104.0, len(index))[:, None] * np.ones((1, 2)),
        index=index,
        columns=columns,
    )
    premium_open = pd.DataFrame(
        np.arange(16, dtype=float).reshape(8, 2) / 10_000,
        index=index,
        columns=columns,
    )
    premium_range = pd.DataFrame(
        np.arange(1, 17, dtype=float).reshape(8, 2) / 100_000,
        index=index,
        columns=columns,
    )
    ohlc = {
        "open": premium_open,
        "high": premium_open + premium_range,
        "low": premium_open - premium_range,
        "close": premium_open + premium_range / 2,
    }
    cfg = V193FeatureConfig(source_lookback_bars=4, source_min_bars=3)
    original = build_v193_pressure_features(price, ohlc, cfg)
    mutated = {name: frame.copy() for name, frame in ohlc.items()}
    mutated["high"].loc[index[-1], "BTCUSDT"] += 1.0
    changed = build_v193_pressure_features(price, mutated, cfg)
    assert original["range_z"].loc[index[-2], "BTCUSDT"] == changed[
        "range_z"
    ].loc[index[-2], "BTCUSDT"]
    assert original["range_z"].loc[index[-1], "BTCUSDT"] != changed[
        "range_z"
    ].loc[index[-1], "BTCUSDT"]


def test_v193_close_location_is_bounded() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    price = pd.DataFrame({"BTCUSDT": np.arange(100.0, 105.0)}, index=index)
    ohlc = {
        "open": pd.DataFrame({"BTCUSDT": np.zeros(5)}, index=index),
        "high": pd.DataFrame({"BTCUSDT": np.ones(5)}, index=index),
        "low": pd.DataFrame({"BTCUSDT": -np.ones(5)}, index=index),
        "close": pd.DataFrame(
            {"BTCUSDT": [-1.0, -0.5, 0.0, 0.5, 1.0]}, index=index
        ),
    }
    cfg = V193FeatureConfig(source_lookback_bars=3, source_min_bars=2)
    features = build_v193_pressure_features(price, ohlc, cfg)
    assert features["close_location"]["BTCUSDT"].between(-1.0, 1.0).all()
    assert np.allclose(
        features["close_location"]["BTCUSDT"],
        np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0]),
    )
