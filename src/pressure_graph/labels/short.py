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

v3.4 extensions add the docx-mandated label family:
    hit_down_3pct_4h / hit_down_5pct_12h / hit_down_8pct_24h
    up_before_down_2pct / up_before_down_3pct / up_before_down_5pct
    short_first_touch_<window>   (0=down_first, 1=up_first, 2=neither)
    time_to_down_hit_<window>    (bars to first downside hit; window+1 if never)
    time_to_up_stop_<window>     (bars to first upside stop; window+1 if never)
    clean_short_hit_<window>     (hit downside AND first touch is down)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

KEYS = ["exchange", "symbol"]


# v3.4 docx §1: window-scaled thresholds so 4h, 12h, 24h each match the docx's
# intended magnitudes. Existing v1.2s tests inherit the 4h defaults; 12h/24h
# are new in v3.4.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "4h": (0.03, 0.02),
    "12h": (0.05, 0.03),
    "24h": (0.08, 0.05),
}


@dataclass(frozen=True)
class FirstTouchCode:
    DOWN_FIRST: int = 0
    UP_FIRST: int = 1
    NEITHER: int = 2


FIRST_TOUCH = FirstTouchCode()


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


def short_first_touch_and_timing(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
    down_target: float,
    up_stop: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Walk forward and resolve sequencing.

    Returns three series aligned to ``close.index``:
      - first_touch: which barrier got hit first (DOWN_FIRST / UP_FIRST / NEITHER)
      - time_to_down: bars to first downside hit (``window + 1`` if never hit)
      - time_to_up:   bars to first upside hit   (``window + 1`` if never hit)

    Ambiguous bars (both barriers in one bar) resolve UP_FIRST — the conservative
    assumption for short books, matching ``simulate_short_exit``'s "stop_ambiguous".
    """
    close_arr = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    n = len(close_arr)
    sentinel = window + 1
    first = np.full(n, FIRST_TOUCH.NEITHER, dtype=np.int8)
    t_down = np.full(n, sentinel, dtype=np.int32)
    t_up = np.full(n, sentinel, dtype=np.int32)
    for idx in range(n):
        price = close_arr[idx]
        if not np.isfinite(price):
            continue
        target_price = price * (1.0 - down_target)
        stop_price = price * (1.0 + up_stop)
        down_hit_at = sentinel
        up_hit_at = sentinel
        last = min(idx + window, n - 1)
        for j in range(idx + 1, last + 1):
            bar_step = j - idx
            bar_down = np.isfinite(low_arr[j]) and low_arr[j] <= target_price
            bar_up = np.isfinite(high_arr[j]) and high_arr[j] >= stop_price
            if bar_down and down_hit_at == sentinel:
                down_hit_at = bar_step
            if bar_up and up_hit_at == sentinel:
                up_hit_at = bar_step
            if down_hit_at != sentinel and up_hit_at != sentinel:
                break
        t_down[idx] = down_hit_at
        t_up[idx] = up_hit_at
        if down_hit_at == sentinel and up_hit_at == sentinel:
            first[idx] = FIRST_TOUCH.NEITHER
        elif down_hit_at == sentinel:
            first[idx] = FIRST_TOUCH.UP_FIRST
        elif up_hit_at == sentinel:
            first[idx] = FIRST_TOUCH.DOWN_FIRST
        elif up_hit_at <= down_hit_at:
            # Ambiguous (==) resolves UP_FIRST — same convention as the short simulator.
            first[idx] = FIRST_TOUCH.UP_FIRST
        else:
            first[idx] = FIRST_TOUCH.DOWN_FIRST
    return (
        pd.Series(first, index=close.index),
        pd.Series(t_down, index=close.index),
        pd.Series(t_up, index=close.index),
    )


def _add_short_labels_for_group(
    group: pd.DataFrame,
    windows: dict[str, int],
    thresholds: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
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
        out[f"hit_down_8pct_{suffix}"] = out[f"future_max_down_{suffix}"] <= -0.08
        # Short adverse / squeeze excursion (positive magnitude).
        out[f"max_adverse_excursion_{suffix}"] = out[f"future_max_up_{suffix}"]
        out[f"up_2pct_before_down_3pct_{suffix}"] = squeeze_before_hit(
            out["close"], out["high"], out["low"], bars, 0.03, 0.02
        )
        out[f"up_3pct_before_down_5pct_{suffix}"] = squeeze_before_hit(
            out["close"], out["high"], out["low"], bars, 0.05, 0.03
        )
        # v3.4 docx labels: window-scaled thresholds and first-touch resolution.
        if suffix in thresholds:
            down_target, up_stop = thresholds[suffix]
            out[f"up_{int(up_stop * 100)}pct_before_down_{int(down_target * 100)}pct_{suffix}"] = (
                squeeze_before_hit(out["close"], out["high"], out["low"], bars, down_target, up_stop)
            )
            first_touch, t_down, t_up = short_first_touch_and_timing(
                out["close"], out["high"], out["low"], bars, down_target, up_stop
            )
            out[f"short_first_touch_{suffix}"] = first_touch
            out[f"time_to_down_hit_{suffix}"] = t_down
            out[f"time_to_up_stop_{suffix}"] = t_up
            hit_col = f"hit_down_{int(down_target * 100)}pct_{suffix}"
            squeeze_col = (
                f"up_{int(up_stop * 100)}pct_before_down_{int(down_target * 100)}pct_{suffix}"
            )
            # clean_short_hit: target hit AND was NOT squeezed first.
            out[f"clean_short_hit_{suffix}"] = out[hit_col].fillna(False) & (~out[squeeze_col].fillna(False))
    return out


def add_short_labels(
    df: pd.DataFrame,
    windows: dict[str, int] | None = None,
    thresholds: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Attach short-side forward labels per symbol.

    Default windows now mirror the v3.4 short instruction: 4h, 12h, 24h. Tests
    that pass a narrower ``windows`` mapping continue to work; v3.4-specific
    labels (first-touch, time-to-hit, clean_short_hit) only emit for suffixes
    in ``thresholds``.
    """
    if windows is None:
        windows = {"4h": 16, "12h": 48, "24h": 96}
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    frames = [
        _add_short_labels_for_group(group, windows, thresholds)
        for _, group in df.groupby(KEYS, sort=False)
    ]
    return pd.concat(frames, ignore_index=True) if frames else df.copy()


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FIRST_TOUCH",
    "add_short_labels",
    "short_first_touch_and_timing",
    "squeeze_before_hit",
]
