from __future__ import annotations

from bisect import bisect_left, bisect_right, insort

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig


KEYS = ["exchange", "symbol"]


def _rolling_percentile(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(arr), np.nan)
    history: list[float] = []
    for idx, value in enumerate(arr):
        if idx > 0 and np.isfinite(arr[idx - 1]):
            insort(history, float(arr[idx - 1]))
        expired = idx - window - 1
        if expired >= 0 and np.isfinite(arr[expired]):
            pos = bisect_left(history, float(arr[expired]))
            if pos < len(history):
                history.pop(pos)
        if np.isfinite(value) and len(history) >= min_periods:
            left = bisect_left(history, float(value))
            right = bisect_right(history, float(value))
            result[idx] = 100.0 * ((left + right) / 2.0) / len(history)
    return pd.Series(result, index=values.index)


def _group_percentile(df: pd.DataFrame, col: str, window: int, min_periods: int) -> pd.Series:
    return (
        df.groupby(KEYS, group_keys=False, sort=False)[col]
        .apply(lambda series: _rolling_percentile(series, window, min_periods))
        .reindex(df.index)
    )


def add_v01_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.sort_values(KEYS + ["bar_open_time"]).copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    candle_range = (high - low).replace(0, np.nan)

    out["range_pct"] = high / low.replace(0, np.nan) - 1.0
    out["close_location_value"] = ((close - low) / candle_range).clip(0, 1)
    out["close_location_value"] = out["close_location_value"].fillna(0.5)
    out["upper_wick_ratio"] = ((high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range)
    out["upper_wick_ratio"] = out["upper_wick_ratio"].clip(0, 1).fillna(0)

    window = config.rolling_window_bars
    min_periods = config.min_history_bars
    percentile_sources = {
        "ret_15m": "ret_15m_percentile",
        "ret_1h": "ret_1h_percentile",
        "range_pct": "range_pct_percentile",
        "volatility_4h": "symbol_volatility_percentile",
    }
    for src, dst in percentile_sources.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = _group_percentile(out, src, window, min_periods)
    return out
