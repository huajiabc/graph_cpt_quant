from __future__ import annotations

from bisect import bisect_left, bisect_right, insort

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig


KEYS = ["exchange", "symbol"]


def ensure_bar_times(df: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    out = df.copy()
    out["bar_open_time"] = pd.to_datetime(out["bar_open_time"], utc=True)
    out = out.sort_values(["exchange", "symbol", "bar_open_time"]).drop_duplicates(
        ["exchange", "symbol", "bar_open_time"]
    )
    out["bar_close_time"] = out["bar_open_time"] + pd.Timedelta(interval)
    out["feature_time"] = out["bar_close_time"]
    out["entry_time"] = out.groupby(KEYS, sort=False)["bar_open_time"].shift(-1)
    return out


def rolling_percentile_current_vs_prior(
    values: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(arr), np.nan, dtype=float)
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


def prior_window_zscore(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    prior = numeric.shift(1)
    mean = prior.rolling(window, min_periods=min_periods).mean()
    std = prior.rolling(window, min_periods=min_periods).std(ddof=0)
    return (numeric - mean) / std.replace(0, np.nan)


def _asof_group(
    bars: pd.DataFrame,
    right: pd.DataFrame,
    right_time_col: str,
    suffix: str,
) -> pd.DataFrame:
    if right.empty:
        return bars.copy()
    pieces: list[pd.DataFrame] = []
    right = right.copy()
    right[right_time_col] = pd.to_datetime(right[right_time_col], utc=True)
    for key, left_group in bars.groupby(KEYS, sort=False):
        exchange, symbol = key
        right_group = right[(right["exchange"] == exchange) & (right["symbol"] == symbol)]
        left_sorted = left_group.sort_values("feature_time")
        if right_group.empty:
            pieces.append(left_sorted)
            continue
        merged = pd.merge_asof(
            left_sorted,
            right_group.sort_values(right_time_col),
            left_on="feature_time",
            right_on=right_time_col,
            direction="backward",
            suffixes=("", suffix),
        )
        for col in [f"exchange{suffix}", f"symbol{suffix}"]:
            if col in merged.columns:
                merged = merged.drop(columns=col)
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True).sort_values(KEYS + ["bar_open_time"])


def align_funding(
    bars: pd.DataFrame,
    funding: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    out = _asof_group(bars, funding, "funding_time", "_funding")
    if "funding_time" not in out.columns:
        out["funding_time"] = pd.NaT
        out["funding_rate_settled"] = np.nan
    intervals = {}
    if not instruments.empty and "funding_interval_minutes" in instruments.columns:
        for row in instruments.itertuples(index=False):
            intervals[getattr(row, "symbol")] = float(getattr(row, "funding_interval_minutes"))
    out["funding_interval_minutes"] = out["symbol"].map(intervals).fillna(480.0)
    out["funding_age_minutes"] = (
        (out["feature_time"] - pd.to_datetime(out["funding_time"], utc=True))
        / pd.Timedelta(minutes=1)
    )
    future_mask = out["funding_time"].notna() & (out["funding_time"] > out["feature_time"])
    if future_mask.any():
        raise ValueError("funding as-of join produced future funding rows")
    max_age = (
        config.features.funding_asof_max_age_intervals * out["funding_interval_minutes"]
    )
    stale = out["funding_age_minutes"] > max_age
    out.loc[stale, ["funding_rate_settled", "funding_time", "funding_age_minutes"]] = [
        np.nan,
        pd.NaT,
        np.nan,
    ]
    return out


def align_open_interest(bars: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    out = _asof_group(bars, oi, "oi_time", "_oi")
    if "oi_time" not in out.columns:
        out["oi_time"] = pd.NaT
        out["oi_base"] = np.nan
    future_mask = out["oi_time"].notna() & (out["oi_time"] > out["feature_time"])
    if future_mask.any():
        raise ValueError("OI as-of join produced future OI rows")
    if "oi_value_usdt" not in out.columns:
        out["oi_value_usdt"] = np.nan
    out["oi_value_usdt"] = pd.to_numeric(out["oi_value_usdt"], errors="coerce")
    missing_value = out["oi_value_usdt"].isna() & out["oi_base"].notna()
    out.loc[missing_value, "oi_value_usdt"] = (
        pd.to_numeric(out.loc[missing_value, "oi_base"], errors="coerce")
        * pd.to_numeric(out.loc[missing_value, "close"], errors="coerce")
    )
    return out


def _group_apply_series(
    df: pd.DataFrame,
    col: str,
    func,
    *args,
) -> pd.Series:
    return (
        df.groupby(KEYS, group_keys=False, sort=False)[col]
        .apply(lambda series: func(series, *args))
        .reindex(df.index)
    )


def add_core_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = ensure_bar_times(df, config.experiment.base_interval)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    grouped = out.groupby(KEYS, group_keys=False, sort=False)
    out["ret_15m"] = grouped["close"].pct_change(1)
    out["ret_1h"] = grouped["close"].pct_change(4)
    out["ret_4h"] = grouped["close"].pct_change(16)
    out["volatility_1h"] = grouped["ret_15m"].transform(
        lambda s: s.rolling(4, min_periods=2).std(ddof=0)
    )
    out["volatility_4h"] = grouped["ret_15m"].transform(
        lambda s: s.rolling(16, min_periods=8).std(ddof=0)
    )
    out["volume_1h"] = grouped["volume"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    out["volume_4h"] = grouped["volume"].transform(lambda s: s.rolling(16, min_periods=1).sum())

    window = config.rolling_window_bars
    min_periods = config.min_history_bars
    for src, dst in [
        ("volume_1h", "volume_z_1h"),
        ("volume_4h", "volume_z_4h"),
        ("funding_rate_settled", "funding_z"),
    ]:
        if src in out.columns:
            out[dst] = _group_apply_series(out, src, prior_window_zscore, window, min_periods)

    if "funding_rate_settled" in out.columns:
        out["funding_percentile"] = _group_apply_series(
            out, "funding_rate_settled", rolling_percentile_current_vs_prior, window, min_periods
        )

    if "oi_value_usdt" in out.columns:
        out["oi_delta_1h"] = (
            grouped["oi_base"].pct_change(4, fill_method=None)
            if "oi_base" in out.columns
            else np.nan
        )
        out["oi_delta_4h"] = (
            grouped["oi_base"].pct_change(16, fill_method=None)
            if "oi_base" in out.columns
            else np.nan
        )
        out["oi_value_delta_1h"] = grouped["oi_value_usdt"].pct_change(4, fill_method=None)
        out["oi_value_delta_4h"] = grouped["oi_value_usdt"].pct_change(16, fill_method=None)
        for src, dst in [
            ("oi_value_delta_1h", "oi_value_delta_z_1h"),
            ("oi_value_delta_4h", "oi_value_delta_z_4h"),
        ]:
            out[dst] = _group_apply_series(out, src, prior_window_zscore, window, min_periods)
        for src, dst in [
            ("oi_value_delta_1h", "oi_value_delta_1h_percentile"),
            ("oi_value_delta_4h", "oi_value_delta_4h_percentile"),
            ("oi_delta_1h", "oi_delta_1h_percentile"),
            ("oi_delta_4h", "oi_delta_4h_percentile"),
        ]:
            out[dst] = _group_apply_series(
                out, src, rolling_percentile_current_vs_prior, window, min_periods
            )

    for src, dst in [
        ("ret_4h", "ret_4h_percentile"),
        ("volume_1h", "volume_1h_percentile"),
        ("volume_4h", "volume_4h_percentile"),
    ]:
        out[dst] = _group_apply_series(
            out, src, rolling_percentile_current_vs_prior, window, min_periods
        )

    out["warmup_complete"] = grouped.cumcount() >= min_periods
    return out


BTC_VOL_WINDOW = 30 * 96
BTC_VOL_MIN_PERIODS = 14 * 96


def btc_market_state_labels(btc_ret_4h: pd.Series) -> pd.Series:
    state = pd.Series("BTC_chop", index=btc_ret_4h.index)
    state[pd.to_numeric(btc_ret_4h, errors="coerce") > 0.01] = "BTC_up"
    state[pd.to_numeric(btc_ret_4h, errors="coerce") < -0.015] = "BTC_down"
    return state


def btc_vol_regime_labels(percentile: pd.Series) -> pd.Series:
    regime = pd.Series("mid_vol", index=percentile.index)
    values = pd.to_numeric(percentile, errors="coerce")
    regime[values >= 70] = "high_vol"
    regime[values <= 30] = "low_vol"
    return regime


def add_btc_state(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    btc = out[out["symbol"].eq("BTCUSDT")][
        ["exchange", "bar_open_time", "ret_1h", "ret_4h", "volatility_4h"]
    ].rename(
        columns={
            "ret_1h": "btc_ret_1h",
            "ret_4h": "btc_ret_4h",
            "volatility_4h": "btc_volatility_4h",
        }
    )
    out = out.merge(btc, on=["exchange", "bar_open_time"], how="left")
    out["btc_market_state"] = btc_market_state_labels(out["btc_ret_4h"])
    vol = out.groupby("exchange")["btc_volatility_4h"].transform(
        lambda s: rolling_percentile_current_vs_prior(s, BTC_VOL_WINDOW, BTC_VOL_MIN_PERIODS)
    )
    out["btc_volatility_percentile"] = vol
    out["btc_vol_regime"] = btc_vol_regime_labels(out["btc_volatility_percentile"])
    return out


def build_feature_table(
    klines: pd.DataFrame,
    funding: pd.DataFrame,
    oi: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    bars = ensure_bar_times(klines, config.experiment.base_interval)
    bars = align_funding(bars, funding, instruments, config)
    bars = align_open_interest(bars, oi)
    bars = add_core_features(bars, config)
    return add_btc_state(bars)
