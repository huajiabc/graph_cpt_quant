from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["exchange", "symbol"]


def _forward_rolling_max(series: pd.Series, window: int) -> pd.Series:
    future = series.shift(-1)
    return future.iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]


def _forward_rolling_min(series: pd.Series, window: int) -> pd.Series:
    future = series.shift(-1)
    return future.iloc[::-1].rolling(window, min_periods=1).min().iloc[::-1]


def dd_before_hit(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    hit_threshold: float,
    dd_threshold: float,
) -> pd.Series:
    close_arr = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    result = np.zeros(len(close_arr), dtype=bool)
    for idx, price in enumerate(close_arr):
        if not np.isfinite(price):
            result[idx] = False
            continue
        hit_price = price * (1.0 + hit_threshold)
        dd_price = price * (1.0 - dd_threshold)
        saw_dd = False
        for lookahead in range(idx + 1, min(idx + window + 1, len(close_arr))):
            bar_hit = np.isfinite(high_arr[lookahead]) and high_arr[lookahead] >= hit_price
            bar_dd = np.isfinite(low_arr[lookahead]) and low_arr[lookahead] <= dd_price
            if bar_dd and not bar_hit:
                saw_dd = True
            elif bar_dd and bar_hit:
                result[idx] = True
                break
            elif bar_hit:
                result[idx] = saw_dd
                break
        else:
            result[idx] = saw_dd
    return pd.Series(result, index=close.index)


def _add_labels_for_group(group: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    out = group.sort_values("bar_open_time").copy()
    for suffix, bars in windows.items():
        future_high = _forward_rolling_max(out["high"], bars)
        future_low = _forward_rolling_min(out["low"], bars)
        out[f"future_max_up_{suffix}"] = future_high / out["close"] - 1.0
        out[f"future_max_down_{suffix}"] = future_low / out["close"] - 1.0
        out[f"future_ret_{suffix}"] = out["close"].shift(-bars) / out["close"] - 1.0
        out[f"hit_3pct_{suffix}"] = out[f"future_max_up_{suffix}"] >= 0.03
        out[f"hit_5pct_{suffix}"] = out[f"future_max_up_{suffix}"] >= 0.05
        out[f"dd_2pct_before_hit_3pct_{suffix}"] = dd_before_hit(
            out["close"], out["high"], out["low"], bars, 0.03, 0.02
        )
        out[f"dd_3pct_before_hit_5pct_{suffix}"] = dd_before_hit(
            out["close"], out["high"], out["low"], bars, 0.05, 0.03
        )
    return out


def add_future_labels(df: pd.DataFrame, windows: dict[str, int] | None = None) -> pd.DataFrame:
    if windows is None:
        windows = {"4h": 16, "12h": 48}
    frames = [_add_labels_for_group(group, windows) for _, group in df.groupby(KEYS, sort=False)]
    return pd.concat(frames, ignore_index=True) if frames else df.copy()
