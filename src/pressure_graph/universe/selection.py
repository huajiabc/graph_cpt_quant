from __future__ import annotations

import re

import pandas as pd

from pressure_graph.config import ExperimentConfig


STABLECOIN_BASES = {
    "USDT",
    "USDC",
    "BUSD",
    "FDUSD",
    "DAI",
    "TUSD",
    "USDE",
    "USDP",
    "PYUSD",
}


def base_asset_from_symbol(symbol: str) -> str:
    for quote in ("USDT", "USDC", "USD"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return re.sub(r"(USDT|USDC|USD)$", "", symbol)


def filter_instruments(
    instruments: pd.DataFrame,
    asof_time: pd.Timestamp,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if instruments.empty:
        return instruments
    df = instruments.copy()
    if "symbol" not in df.columns:
        return df.iloc[0:0]
    if config.universe.exclude_stablecoins:
        df = df[~df["symbol"].map(base_asset_from_symbol).isin(STABLECOIN_BASES)].copy()
    if "launch_time" in df.columns:
        launch = pd.to_datetime(df["launch_time"], utc=True, errors="coerce")
        min_age = pd.Timedelta(days=config.universe.min_listing_age_days)
        df = df[(launch.isna()) | (launch <= asof_time - min_age)].copy()
    return df


def static_current_top_symbols(
    tickers: pd.DataFrame,
    instruments: pd.DataFrame,
    asof_time: pd.Timestamp,
    config: ExperimentConfig,
) -> list[str]:
    eligible = filter_instruments(instruments, asof_time, config)
    if eligible.empty or tickers.empty:
        return []
    merged = tickers.merge(eligible[["symbol"]], on="symbol", how="inner")
    turnover_col = "turnover24h" if "turnover24h" in merged.columns else "quoteVolume"
    merged[turnover_col] = pd.to_numeric(merged[turnover_col], errors="coerce")
    ranked = merged.dropna(subset=[turnover_col]).sort_values(turnover_col, ascending=False)
    return ranked["symbol"].drop_duplicates().head(config.universe.top_n).tolist()


def missing_ratio_by_symbol(klines: pd.DataFrame, bars_per_day: int) -> pd.Series:
    ratios: dict[str, float] = {}
    for symbol, group in klines.groupby("symbol", sort=False):
        times = pd.to_datetime(group["bar_open_time"], utc=True)
        if times.empty:
            ratios[symbol] = 1.0
            continue
        expected = int(((times.max() - times.min()) / pd.Timedelta(minutes=15))) + 1
        observed = times.nunique()
        ratios[symbol] = max(0.0, 1.0 - observed / max(expected, 1))
    return pd.Series(ratios)


def dynamic_monthly_top_symbols(
    klines: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[pd.Timestamp, list[str]]:
    if klines.empty:
        return {}
    df = klines.copy()
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"], utc=True)
    df["turnover"] = pd.to_numeric(df.get("turnover"), errors="coerce")
    if df["turnover"].isna().all():
        df["turnover"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(
            df["volume"], errors="coerce"
        )
    missing = missing_ratio_by_symbol(df, config.bars_per_day)
    valid_missing = missing[missing <= config.universe.max_missing_ratio].index
    df = df[df["symbol"].isin(valid_missing)].copy()
    if df.empty:
        return {}

    month_starts = pd.date_range(
        df["bar_open_time"].min().normalize().replace(day=1),
        df["bar_open_time"].max().normalize().replace(day=1),
        freq="MS",
        tz="UTC",
    )
    selections: dict[pd.Timestamp, list[str]] = {}
    lookback = pd.Timedelta(days=config.universe.dynamic_lookback_days)
    for month_start in month_starts:
        eligible = filter_instruments(instruments, month_start, config)
        eligible_symbols = set(eligible["symbol"]) if not eligible.empty else set(df["symbol"].unique())
        hist = df[
            (df["bar_open_time"] >= month_start - lookback)
            & (df["bar_open_time"] < month_start)
            & (df["symbol"].isin(eligible_symbols))
        ]
        if hist.empty:
            selections[month_start] = []
            continue
        ranked = (
            hist.groupby("symbol", as_index=False)["turnover"]
            .sum(min_count=1)
            .dropna(subset=["turnover"])
            .sort_values("turnover", ascending=False)
        )
        selections[month_start] = ranked["symbol"].head(config.universe.top_n).tolist()
    return selections


def apply_universe_flags(
    klines: pd.DataFrame,
    instruments: pd.DataFrame,
    tickers: pd.DataFrame,
    config: ExperimentConfig,
    asof_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = klines.copy()
    if df.empty:
        return df
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"], utc=True)
    asof = asof_time or df["bar_open_time"].max()
    static = set(static_current_top_symbols(tickers, instruments, asof, config))
    dynamic = dynamic_monthly_top_symbols(df, instruments, config)

    df["universe_static_current_top30"] = df["symbol"].isin(static)
    month = pd.to_datetime(
        {
            "year": df["bar_open_time"].dt.year,
            "month": df["bar_open_time"].dt.month,
            "day": 1,
        },
        utc=True,
    )
    dynamic_flags = []
    for row_month, symbol in zip(month, df["symbol"], strict=False):
        dynamic_flags.append(symbol in set(dynamic.get(row_month, [])))
    df["universe_dynamic_monthly_top30"] = dynamic_flags
    return df
