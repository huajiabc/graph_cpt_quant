"""Short-side forward labels (mirror of ``labels/future.py``).

The short instruction doc (§4) is explicit: do not label short candidates with
``future_return`` alone. A short can have the direction right yet still be
stopped out by a squeeze first. So every short label pairs a downside-capture
measure with a squeeze/adverse measure, and the headline label is
``squeeze_before_hit`` — the short mirror of the long-side
``dd_before_hit``.

Sign convention (long-style price ratios, short P&L derived later):
    future_max_down_Nh = future_low / close  - 1   (<=0; short MFE magnitude)
    future_max_up_Nh   = future_high / close - 1   (>=0; short MAE = squeeze)

All labels read strictly forward (``shift(-1)`` start) so the current bar never
sees its own outcome; entries are taken at the next bar open downstream.
"""
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


def squeeze_before_hit(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    down_target: float,
    squeeze_threshold: float,
) -> pd.Series:
    """Short mirror of ``labels.future.dd_before_hit``.

    For each bar, walk forward up to ``window`` bars and decide whether a short
    taken now would be squeezed. Returns True when an adverse up-move of
    ``squeeze_threshold`` occurs *before* the downside target of ``down_target``
    is reached (i.e. the squeeze happens first or alongside the target). This is
    the event a short most wants to avoid: right direction, wrong sequencing.
    """
    close_arr = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    result = np.zeros(len(close_arr), dtype=bool)
    for idx, price in enumerate(close_arr):
        if not np.isfinite(price):
            continue
        target_price = price * (1.0 - down_target)
        squeeze_price = price * (1.0 + squeeze_threshold)
        saw_squeeze = False
        for lookahead in range(idx + 1, min(idx + window + 1, len(close_arr))):
            bar_target = np.isfinite(low_arr[lookahead]) and low_arr[lookahead] <= target_price
            bar_squeeze = np.isfinite(high_arr[lookahead]) and high_arr[lookahead] >= squeeze_price
            if bar_squeeze and not bar_target:
                saw_squeeze = True
            elif bar_squeeze and bar_target:
                result[idx] = True
                break
            elif bar_target:
                result[idx] = saw_squeeze
                break
        else:
            result[idx] = saw_squeeze
    return pd.Series(result, index=close.index)


def _add_short_labels_for_group(group: pd.DataFrame, windows: dict[str, int]) -> pd.DataFrame:
    out = group.sort_values("bar_open_time").copy()
    for suffix, bars in windows.items():
        future_high = _forward_rolling_max(out["high"], bars)
        future_low = _forward_rolling_min(out["low"], bars)
        out[f"future_max_up_{suffix}"] = future_high / out["close"] - 1.0
        out[f"future_max_down_{suffix}"] = future_low / out["close"] - 1.0
        out[f"future_ret_{suffix}"] = out["close"].shift(-bars) / out["close"] - 1.0
        # Short downside-capture hits: price fell at least N% at some point.
        out[f"hit_down_3pct_{suffix}"] = out[f"future_max_down_{suffix}"] <= -0.03
        out[f"hit_down_5pct_{suffix}"] = out[f"future_max_down_{suffix}"] <= -0.05
        # Short adverse / squeeze excursion (positive magnitude).
        out[f"max_adverse_excursion_{suffix}"] = out[f"future_max_up_{suffix}"]
        out[f"up_2pct_before_down_3pct_{suffix}"] = squeeze_before_hit(
            out["close"], out["high"], out["low"], bars, 0.03, 0.02
        )
        out[f"up_3pct_before_down_5pct_{suffix}"] = squeeze_before_hit(
            out["close"], out["high"], out["low"], bars, 0.05, 0.03
        )
    return out


def add_short_labels(df: pd.DataFrame, windows: dict[str, int] | None = None) -> pd.DataFrame:
    """Attach short-side forward labels per symbol.

    Default windows mirror the long-side 4h/12h horizons (16/48 15m bars).
    """
    if windows is None:
        windows = {"4h": 16, "12h": 48}
    frames = [_add_short_labels_for_group(group, windows) for _, group in df.groupby(KEYS, sort=False)]
    return pd.concat(frames, ignore_index=True) if frames else df.copy()


__all__ = ["add_short_labels", "squeeze_before_hit"]
