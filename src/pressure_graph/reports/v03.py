from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pressure_graph.backtest import ENTRY_POLICIES, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir
from pressure_graph.universe.selection import filter_instruments, missing_ratio_by_symbol


C2_CANDIDATE = "C2_short_squeeze_e4_pullback_swing"
C2_SIGNAL_COL = "short_squeeze_signal_event"
C2_STATE_COL = "short_squeeze_signal"
C2_ENTRY_POLICY = "E4_pullback_0.5pct_valid_4_bars"
C2_EXECUTION_RULE = "swing"
COST_BPS = [5, 10, 20, 30, 50]
TOP_NS = [30, 50, 100]
KEYS = ["exchange", "symbol"]
MAX_TOP_N = max(TOP_NS)
V03_REQUIRED_COLUMNS = [
    "exchange",
    "symbol",
    "bar_open_time",
    "bar_close_time",
    "feature_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "turnover",
    "universe_static_current_top30",
    "funding_time",
    "funding_rate_settled",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "volatility_4h",
    "volume_z_1h",
    "volume_z_4h",
    "funding_z",
    "funding_percentile",
    "oi_value_delta_1h_percentile",
    "oi_value_delta_4h_percentile",
    "ret_4h_percentile",
    "warmup_complete",
    "btc_ret_4h",
    "btc_market_state",
]
V03_TURNOVER_COLUMNS = ["symbol", "bar_open_time", "turnover", "close", "volume"]


def _month_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True)
    return pd.to_datetime({"year": ts.dt.year, "month": ts.dt.month, "day": 1}, utc=True)


def _parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _read_existing_columns(
    path: Path,
    columns: list[str],
    filters: list[tuple[str, str, object]] | None = None,
) -> pd.DataFrame:
    available = set(_parquet_columns(path))
    selected = [col for col in columns if col in available]
    return pd.read_parquet(path, columns=selected, filters=filters)


def _downcast_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "funding_rate_settled",
        "ret_15m",
        "ret_1h",
        "ret_4h",
        "volatility_4h",
        "volume_z_1h",
        "volume_z_4h",
        "funding_z",
        "funding_percentile",
        "oi_value_delta_1h_percentile",
        "oi_value_delta_4h_percentile",
        "ret_4h_percentile",
        "btc_ret_4h",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    for col in ["exchange", "symbol", "btc_market_state"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in ["bar_open_time", "bar_close_time", "feature_time", "funding_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "warmup_complete" in df.columns:
        df["warmup_complete"] = df["warmup_complete"].fillna(False).astype(bool)
    return df


def _event_flags(signals: pd.Series, cooldown_bars: int) -> pd.Series:
    signal_arr = signals.fillna(False).astype(bool).to_numpy()
    events = np.zeros(len(signal_arr), dtype=bool)
    last_event = -10**9
    for idx, is_signal in enumerate(signal_arr):
        if is_signal and idx - last_event >= cooldown_bars:
            events[idx] = True
            last_event = idx
    return pd.Series(events, index=signals.index)


def prepare_v03_c2_dataset(features: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = features.sort_values(KEYS + ["bar_open_time"]).copy()
    short_rule = config.path_rules.short_squeeze
    crowded_rule = config.path_rules.crowded_long_risk

    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)

    market_ok = pd.to_numeric(out.get("btc_ret_4h"), errors="coerce") > short_rule.btc_ret_4h_min
    oi_delta_1h_pct = pd.to_numeric(out.get("oi_value_delta_1h_percentile"), errors="coerce")
    oi_delta_4h_pct = pd.to_numeric(out.get("oi_value_delta_4h_percentile"), errors="coerce")
    short_oi_up = (oi_delta_1h_pct > 75) | (oi_delta_4h_pct > 75)
    funding_not_hot = (
        pd.to_numeric(out.get("funding_percentile"), errors="coerce") < 60
    ) | (pd.to_numeric(out.get("funding_z"), errors="coerce") < short_rule.funding_z_max)
    price_resilient = (
        pd.to_numeric(out.get("ret_1h"), errors="coerce") > 0
    ) | (pd.to_numeric(out.get("ret_4h"), errors="coerce") > 0)
    volume_confirm = pd.to_numeric(out.get("volume_z_1h"), errors="coerce") > 1.0
    out["short_squeeze_signal_raw"] = (
        warmup & market_ok & short_oi_up & funding_not_hot & price_resilient & volume_confirm
    )

    out["crowded_long_risk"] = (
        warmup
        & (pd.to_numeric(out.get("funding_percentile"), errors="coerce") > crowded_rule.funding_percentile_min)
        & (oi_delta_4h_pct > crowded_rule.oi_percentile_min)
        & (pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce") < crowded_rule.ret_4h_percentile_max)
    )
    out[C2_STATE_COL] = out["short_squeeze_signal_raw"] & ~out["crowded_long_risk"]
    out[C2_SIGNAL_COL] = (
        out.groupby(KEYS, group_keys=False, sort=False, observed=True)[C2_STATE_COL]
        .apply(lambda series: _event_flags(series, config.events.cooldown_bars_4h))
        .reindex(out.index)
    )
    return out


def build_dynamic_rank_table(
    turnover_df: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if turnover_df.empty:
        return pd.DataFrame(columns=["month_start", "symbol", "dynamic_all_rank", "dynamic_all_trailing_turnover"])
    working = turnover_df.copy()
    working["bar_open_time"] = pd.to_datetime(working["bar_open_time"], utc=True, errors="coerce")
    working["turnover"] = pd.to_numeric(working.get("turnover"), errors="coerce")
    if working["turnover"].isna().all() and {"close", "volume"}.issubset(working.columns):
        working["turnover"] = pd.to_numeric(working["close"], errors="coerce") * pd.to_numeric(
            working["volume"], errors="coerce"
        )

    missing = missing_ratio_by_symbol(working, config.bars_per_day)
    valid_symbols = set(missing[missing <= config.universe.max_missing_ratio].index)
    working = working[working["symbol"].isin(valid_symbols)].copy()
    working["month_start"] = _month_start(working["bar_open_time"])
    month_starts = sorted(working["month_start"].dropna().unique())
    lookback = pd.Timedelta(days=config.universe.dynamic_lookback_days)

    rank_rows = []
    for month_start in month_starts:
        month_start = pd.Timestamp(month_start)
        eligible = filter_instruments(instruments, month_start, config)
        eligible_symbols = set(eligible["symbol"]) if not eligible.empty else set(working["symbol"].unique())
        hist = working[
            (working["bar_open_time"] >= month_start - lookback)
            & (working["bar_open_time"] < month_start)
            & (working["symbol"].isin(eligible_symbols))
        ]
        if hist.empty:
            continue
        ranked = (
            hist.groupby("symbol", as_index=False)["turnover"]
            .sum(min_count=1)
            .dropna(subset=["turnover"])
            .sort_values("turnover", ascending=False)
            .reset_index(drop=True)
        )
        ranked["dynamic_all_rank"] = np.arange(1, len(ranked) + 1)
        ranked["month_start"] = month_start
        ranked = ranked.rename(columns={"turnover": "dynamic_all_trailing_turnover"})
        rank_rows.append(
            ranked[["month_start", "symbol", "dynamic_all_rank", "dynamic_all_trailing_turnover"]]
        )
    return pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()


def _read_v03_ranked_features(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    ranks = build_dynamic_rank_table(turnover, instruments, config)
    del turnover
    if ranks.empty:
        raise ValueError("No dynamic all-eligible ranks could be computed for v0.3.")
    top_symbols = sorted(ranks[ranks["dynamic_all_rank"] <= MAX_TOP_N]["symbol"].dropna().unique().tolist())
    if not top_symbols:
        raise ValueError("No symbols reached the dynamic all-eligible Top100 universe.")

    features = _read_existing_columns(
        feature_path,
        V03_REQUIRED_COLUMNS,
        filters=[("symbol", "in", top_symbols)],
    )
    features = _downcast_frame(features)
    features["month_start"] = _month_start(features["bar_open_time"])
    features = features.merge(ranks, on=["month_start", "symbol"], how="left")
    features = features[pd.to_numeric(features["dynamic_all_rank"], errors="coerce") <= MAX_TOP_N].copy()
    features["dynamic_all_rank"] = pd.to_numeric(
        features["dynamic_all_rank"], errors="coerce", downcast="integer"
    )
    features["dynamic_all_trailing_turnover"] = pd.to_numeric(
        features["dynamic_all_trailing_turnover"], errors="coerce", downcast="float"
    )
    return features


def _read_v03_symbol_features(
    feature_path: Path,
    ranks: pd.DataFrame,
    symbol: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    features = _read_existing_columns(
        feature_path,
        V03_REQUIRED_COLUMNS,
        filters=[("symbol", "==", symbol)],
    )
    if features.empty:
        return features
    features = _downcast_frame(features)
    features["month_start"] = _month_start(features["bar_open_time"])
    symbol_ranks = ranks[ranks["symbol"].eq(symbol)]
    features = features.merge(symbol_ranks, on=["month_start", "symbol"], how="left")
    features = features[pd.to_numeric(features["dynamic_all_rank"], errors="coerce") <= MAX_TOP_N].copy()
    if features.empty:
        return features
    features["dynamic_all_rank"] = pd.to_numeric(
        features["dynamic_all_rank"], errors="coerce", downcast="integer"
    )
    features["dynamic_all_trailing_turnover"] = pd.to_numeric(
        features["dynamic_all_trailing_turnover"], errors="coerce", downcast="float"
    )
    return _add_v03_report_columns(features, config)


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _baseline_signal_count(df: pd.DataFrame) -> int:
    if "baseline_event" not in df.columns:
        return 0
    return int(df["baseline_event"].fillna(False).sum())


def _add_v03_report_columns(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = prepare_v03_c2_dataset(df, config)
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    candle_range = (high - low).replace(0, np.nan)
    out["range_pct"] = (high / low.replace(0, np.nan) - 1.0).astype("float32")
    out["close_location_value"] = ((close - low) / candle_range).clip(0, 1).fillna(0.5).astype("float32")
    out["upper_wick_ratio"] = (
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / candle_range
    ).clip(0, 1).fillna(0).astype("float32")
    for top_n in TOP_NS:
        out[f"universe_dynamic_all_top{top_n}"] = (
            pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= top_n
        )
    out["liquidity_bucket"] = pd.cut(
        pd.to_numeric(out["dynamic_all_rank"], errors="coerce"),
        bins=[0, 10, 30, 50, 100, np.inf],
        labels=["rank_1_10", "rank_11_30", "rank_31_50", "rank_51_100", "rank_101_plus"],
    ).astype("string")
    out["month"] = out["month_start"].dt.strftime("%Y-%m")
    if "symbol_volatility_percentile" not in out.columns:
        out["symbol_volatility_percentile"] = (
            pd.to_numeric(out["volatility_4h"], errors="coerce")
            .groupby([out["symbol"], out["month"]], sort=False, observed=True)
            .rank(pct=True)
            .mul(100.0)
            .astype("float32")
        )
    return out


def add_dynamic_all_eligible_universe(
    df: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    top_ns: list[int] | None = None,
) -> pd.DataFrame:
    top_ns = top_ns or TOP_NS
    out = df.copy()
    out["bar_open_time"] = pd.to_datetime(out["bar_open_time"], utc=True)
    out["month_start"] = _month_start(out["bar_open_time"])
    out["turnover"] = pd.to_numeric(out.get("turnover"), errors="coerce")
    if out["turnover"].isna().all():
        out["turnover"] = pd.to_numeric(out["close"], errors="coerce") * pd.to_numeric(
            out["volume"], errors="coerce"
        )

    missing = missing_ratio_by_symbol(out, config.bars_per_day)
    valid_symbols = set(missing[missing <= config.universe.max_missing_ratio].index)
    working = out[out["symbol"].isin(valid_symbols)].copy()
    month_starts = sorted(working["month_start"].dropna().unique())
    lookback = pd.Timedelta(days=config.universe.dynamic_lookback_days)

    rank_rows = []
    for month_start in month_starts:
        month_start = pd.Timestamp(month_start)
        eligible = filter_instruments(instruments, month_start, config)
        eligible_symbols = set(eligible["symbol"]) if not eligible.empty else set(working["symbol"].unique())
        hist = working[
            (working["bar_open_time"] >= month_start - lookback)
            & (working["bar_open_time"] < month_start)
            & (working["symbol"].isin(eligible_symbols))
        ]
        if hist.empty:
            continue
        ranked = (
            hist.groupby("symbol", as_index=False)["turnover"]
            .sum(min_count=1)
            .dropna(subset=["turnover"])
            .sort_values("turnover", ascending=False)
            .reset_index(drop=True)
        )
        ranked["dynamic_all_rank"] = np.arange(1, len(ranked) + 1)
        ranked["month_start"] = month_start
        rank_rows.append(ranked[["month_start", "symbol", "dynamic_all_rank", "turnover"]])

    ranks = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    if ranks.empty:
        out["dynamic_all_rank"] = np.nan
        out["dynamic_all_trailing_turnover"] = np.nan
    else:
        ranks = ranks.rename(columns={"turnover": "dynamic_all_trailing_turnover"})
        out = out.merge(ranks, on=["month_start", "symbol"], how="left")

    for top_n in top_ns:
        out[f"universe_dynamic_all_top{top_n}"] = (
            pd.to_numeric(out["dynamic_all_rank"], errors="coerce") <= top_n
        )
    out["liquidity_bucket"] = pd.cut(
        pd.to_numeric(out["dynamic_all_rank"], errors="coerce"),
        bins=[0, 10, 30, 50, 100, np.inf],
        labels=["rank_1_10", "rank_11_30", "rank_31_50", "rank_51_100", "rank_101_plus"],
    ).astype("string")
    out["month"] = out["month_start"].dt.strftime("%Y-%m")
    return out


def _c2_policy():
    for policy in ENTRY_POLICIES:
        if policy.name == C2_ENTRY_POLICY:
            return policy
    raise KeyError(C2_ENTRY_POLICY)


def _simulate_base_trades(df: pd.DataFrame, signal_col: str = C2_SIGNAL_COL) -> pd.DataFrame:
    return simulate_entry_policy_trades(
        df,
        signal_col,
        "short_squeeze",
        _c2_policy(),
        ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48),
        0,
        "sl_first",
        True,
        C2_STATE_COL if signal_col == C2_SIGNAL_COL else signal_col,
    )

def _expand_costs(base_trades: pd.DataFrame, costs: list[int] | None = None) -> pd.DataFrame:
    costs = costs or COST_BPS
    if base_trades.empty:
        return base_trades.copy()
    frames = []
    for cost in costs:
        data = base_trades.copy()
        data["cost_single_side_bps"] = float(cost)
        data["net_return_ex_fee_slippage"] = pd.to_numeric(data["gross_return"], errors="coerce") - (
            2.0 * float(cost) / 10_000.0
        )
        if "funding_cost" in data.columns:
            data["net_return_ex_fee_slippage_funding"] = (
                data["net_return_ex_fee_slippage"] - pd.to_numeric(data["funding_cost"], errors="coerce").fillna(0)
            )
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _attach_trade_context(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "liquidity_bucket",
        "dynamic_all_trailing_turnover",
        "btc_market_state",
        "symbol_volatility_percentile",
        "liquidity_quality",
        "turnover_rank_30d",
        "turnover_rank_90d",
        "core_liquidity",
        "transient_hot",
    ]
    context = df[[col for col in context_cols if col in df.columns]].drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )
    out = trades.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    )
    return out.drop(columns=["feature_time"], errors="ignore")


def _summary_row(
    trades: pd.DataFrame,
    signal_n: int,
    **keys: object,
) -> dict[str, object]:
    filled_n = len(trades)
    net = pd.to_numeric(trades.get("net_return_ex_fee_slippage"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    exit_reason = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    return {
        **keys,
        "signals": int(signal_n),
        "trades": int(filled_n),
        "fill_rate": float(filled_n / signal_n) if signal_n else np.nan,
        "gross_expectancy": float(gross.mean()) if filled_n else np.nan,
        "net_expectancy": float(net.mean()) if filled_n else np.nan,
        "tp_rate": float(exit_reason.str.startswith("tp").mean()) if filled_n else np.nan,
        "sl_rate": float(exit_reason.str.startswith("sl").mean()) if filled_n else np.nan,
        "timeout_rate": float(exit_reason.eq("max_hold").mean()) if filled_n else np.nan,
        "p25_return": float(net.quantile(0.25)) if filled_n else np.nan,
        "p75_return": float(net.quantile(0.75)) if filled_n else np.nan,
        "max_loss": float(net.min()) if filled_n else np.nan,
    }


def summarize_trades(
    trades: pd.DataFrame,
    signal_counts: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for key, group in trades.groupby(group_cols, dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        filters = {col: value for col, value in zip(group_cols, key_tuple, strict=False)}
        counts = signal_counts.copy()
        for col, value in filters.items():
            if col in counts.columns:
                counts = counts[counts[col].eq(value)]
        signal_n = int(counts["signals"].sum()) if "signals" in counts.columns else 0
        rows.append(_summary_row(group, signal_n, **filters))
    return pd.DataFrame(rows)


def _signal_counts(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sample = df[df[C2_SIGNAL_COL].fillna(False)].copy()
    if sample.empty:
        return pd.DataFrame(columns=[*group_cols, "signals"])
    return sample.groupby(group_cols, dropna=False).size().reset_index(name="signals")


def _matched_random_rows(df: pd.DataFrame, signal_col: str = C2_SIGNAL_COL, seed: int = 42) -> pd.DataFrame:
    data = df.copy()
    data["vol_bucket"] = pd.cut(
        pd.to_numeric(data["symbol_volatility_percentile"], errors="coerce"),
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["v0_20", "v20_40", "v40_60", "v60_80", "v80_100"],
    ).astype(str)
    rng = pd.Series(
        index=data.index,
        data=pd.util.hash_pandas_object(data[["symbol", "bar_open_time"]], index=False),
    )
    data["_rand"] = ((rng + seed) % 1_000_000).astype(float)
    rows = []
    keys = ["symbol", "month", "vol_bucket", "btc_market_state", "liquidity_bucket"]
    for _, group in data.groupby(keys, sort=False, dropna=False, observed=True):
        target = int(group[signal_col].fillna(False).sum())
        if target <= 0:
            continue
        anchors = group[~group[signal_col].fillna(False)].sort_values("_rand").head(target)
        if not anchors.empty:
            rows.append(anchors)
    if not rows:
        return df.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True).drop(columns=["_rand"], errors="ignore")
    out["baseline_event"] = True
    return out


def _entry_only_rows(df: pd.DataFrame, signal_col: str = C2_SIGNAL_COL, cooldown_bars: int = 16) -> pd.DataFrame:
    data = df.copy()
    data["vol_bucket"] = pd.cut(
        pd.to_numeric(data["symbol_volatility_percentile"], errors="coerce"),
        bins=[-1, 20, 40, 60, 80, 101],
        labels=["v0_20", "v20_40", "v40_60", "v60_80", "v80_100"],
    ).astype(str)
    rows = []
    keys = ["symbol", "month", "vol_bucket", "btc_market_state", "liquidity_bucket"]
    for _, group in data.sort_values("bar_open_time").groupby(
        keys, sort=False, dropna=False, observed=True
    ):
        target = int(group[signal_col].fillna(False).sum())
        if target <= 0:
            continue
        anchors = group[~group[signal_col].fillna(False)].iloc[::cooldown_bars].head(target)
        if not anchors.empty:
            rows.append(anchors)
    if not rows:
        return df.iloc[0:0].copy()
    out = pd.concat(rows, ignore_index=True)
    out["baseline_event"] = True
    return out


def _simulate_for_universe(df: pd.DataFrame, universe_col: str, signal_col: str = C2_SIGNAL_COL) -> pd.DataFrame:
    data = df[df[universe_col].fillna(False)].copy()
    base = _simulate_base_trades(data, signal_col)
    return _attach_trade_context(_expand_costs(base), data)


def _partition_filter(trades: pd.DataFrame, partition: str) -> pd.Series:
    month = trades["month"].astype(str)
    if partition == "full_12m":
        return pd.Series(True, index=trades.index)
    if partition == "ex_2026_05":
        return ~month.eq("2026-05")
    if partition == "pre_2026_05":
        return month.lt("2026-05")
    if partition == "only_2026_05":
        return month.eq("2026-05")
    raise KeyError(partition)


def _signals_for_partition(df: pd.DataFrame, universe_col: str, partition: str) -> int:
    data = df[df[universe_col].fillna(False) & df[C2_SIGNAL_COL].fillna(False)].copy()
    month = data["month"].astype(str)
    if partition == "full_12m":
        return len(data)
    if partition == "ex_2026_05":
        return int((~month.eq("2026-05")).sum())
    if partition == "pre_2026_05":
        return int(month.lt("2026-05").sum())
    if partition == "only_2026_05":
        return int(month.eq("2026-05").sum())
    return 0


def c2_partition_summary(trades_by_topn: dict[int, pd.DataFrame], prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_n, trades in trades_by_topn.items():
        universe_col = f"universe_dynamic_all_top{top_n}"
        for partition in ["full_12m", "ex_2026_05", "pre_2026_05", "only_2026_05"]:
            signal_n = _signals_for_partition(prepared, universe_col, partition)
            for cost, group in trades[_partition_filter(trades, partition)].groupby("cost_single_side_bps"):
                rows.append(
                    _summary_row(
                        group,
                        signal_n,
                        universe=f"dynamic_all_top{top_n}",
                        partition=partition,
                        cost_single_side_bps=cost,
                    )
                )
    return pd.DataFrame(rows)


def c2_leave_one_month_out(trades_by_topn: dict[int, pd.DataFrame], prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    months = sorted(prepared["month"].dropna().unique())
    for top_n, trades in trades_by_topn.items():
        universe_col = f"universe_dynamic_all_top{top_n}"
        for excluded in months:
            data = trades[~trades["month"].astype(str).eq(str(excluded))]
            signal_n = int(
                (
                    prepared[universe_col].fillna(False)
                    & prepared[C2_SIGNAL_COL].fillna(False)
                    & ~prepared["month"].astype(str).eq(str(excluded))
                ).sum()
            )
            for cost, group in data.groupby("cost_single_side_bps"):
                rows.append(
                    _summary_row(
                        group,
                        signal_n,
                        universe=f"dynamic_all_top{top_n}",
                        excluded_month=excluded,
                        cost_single_side_bps=cost,
                    )
                )
    return pd.DataFrame(rows)


def c2_liquidity_bucket(trades_by_topn: dict[int, pd.DataFrame], prepared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for top_n, trades in trades_by_topn.items():
        universe_col = f"universe_dynamic_all_top{top_n}"
        counts = _signal_counts(prepared[prepared[universe_col].fillna(False)], ["liquidity_bucket"])
        if trades.empty:
            continue
        for (bucket, cost), group in trades.groupby(["liquidity_bucket", "cost_single_side_bps"], dropna=False):
            signal_n = int(counts[counts["liquidity_bucket"].eq(bucket)]["signals"].sum())
            rows.append(
                _summary_row(
                    group,
                    signal_n,
                    universe=f"dynamic_all_top{top_n}",
                    liquidity_bucket=bucket,
                    cost_single_side_bps=cost,
                )
            )
    return pd.DataFrame(rows)


def c2_contribution(trades_by_topn: dict[int, pd.DataFrame], dimension: str) -> pd.DataFrame:
    rows = []
    for top_n, trades in trades_by_topn.items():
        for cost, cost_group in trades.groupby("cost_single_side_bps"):
            total_net = pd.to_numeric(cost_group["net_return_ex_fee_slippage"], errors="coerce").sum()
            for value, group in cost_group.groupby(dimension, dropna=False):
                net_sum = pd.to_numeric(group["net_return_ex_fee_slippage"], errors="coerce").sum()
                rows.append(
                    {
                        "universe": f"dynamic_all_top{top_n}",
                        "cost_single_side_bps": cost,
                        "dimension": dimension,
                        "value": value,
                        "trades": len(group),
                        "trade_share": len(group) / len(cost_group) if len(cost_group) else np.nan,
                        "net_sum": net_sum,
                        "net_contribution": net_sum / total_net if total_net else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _compare_baselines(
    signal_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
    entry_summary: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["universe", "cost_single_side_bps"]
    out = signal_summary.merge(
        matched_summary[keys + ["net_expectancy"]].rename(columns={"net_expectancy": "matched_random_net"}),
        on=keys,
        how="left",
    ).merge(
        entry_summary[keys + ["net_expectancy"]].rename(columns={"net_expectancy": "entry_only_net"}),
        on=keys,
        how="left",
    )
    out["matched_random_lift"] = out["net_expectancy"] - out["matched_random_net"]
    out["entry_only_lift"] = out["net_expectancy"] - out["entry_only_net"]
    return out


def c2_portfolio_concurrency(trades_by_topn: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for top_n, trades in trades_by_topn.items():
        filled = trades.copy()
        if filled.empty:
            continue
        filled["entry_time"] = pd.to_datetime(filled["entry_time"], utc=True)
        filled["exit_time"] = pd.to_datetime(filled["exit_time"], utc=True)
        for cost, group in filled.groupby("cost_single_side_bps"):
            points = sorted(set(group["entry_time"]).union(set(group["exit_time"])))
            active_counts = []
            for ts in points:
                active_counts.append(len(group[(group["entry_time"] <= ts) & (group["exit_time"] > ts)]))
            for max_pos in [1, 3, 5, 10]:
                rows.append(
                    {
                        "universe": f"dynamic_all_top{top_n}",
                        "cost_single_side_bps": cost,
                        "max_positions": max_pos,
                        "total_trades": len(group),
                        "max_concurrent_positions": max(active_counts) if active_counts else 0,
                        "avg_concurrent_positions": np.mean(active_counts) if active_counts else np.nan,
                        "capital_utilization_proxy": (
                            min(max(active_counts), max_pos) / max_pos if active_counts else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def write_candidate_list(report_root: Path, summary: pd.DataFrame, ex_may: pd.DataFrame) -> None:
    lines = ["# Crypto Pressure Graph v0.3 Candidate List", ""]
    lines.append("C2 is frozen. This report only changes universe/evaluation slices, not rules.")
    lines.append("")
    lines.append("## 5bp Normal 15m Readout")
    top = summary[summary["cost_single_side_bps"].eq(5)].sort_values("net_expectancy", ascending=False)
    for row in top.head(12).itertuples(index=False):
        lines.append(
            f"- {row.universe}: trades={row.trades}, net5={row.net_expectancy:.4%}, "
            f"matched_lift={row.matched_random_lift:.4%}, entry_lift={row.entry_only_lift:.4%}"
        )
    lines.append("")
    lines.append("## Ex-May Check")
    may = ex_may[(ex_may["partition"].eq("ex_2026_05")) & (ex_may["cost_single_side_bps"].eq(5))]
    for row in may.itertuples(index=False):
        lines.append(f"- {row.universe}: ex-May net5={row.net_expectancy:.4%}, trades={row.trades}")
    (report_root / "candidate_list.md").write_text("\n".join(lines), encoding="utf-8")


def write_v03_reports_from_prepared(
    prepared: pd.DataFrame,
    report_root: Path = Path("reports/v0_3"),
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    trades_by_topn: dict[int, pd.DataFrame] = {}
    matched_by_topn: dict[int, pd.DataFrame] = {}
    entry_by_topn: dict[int, pd.DataFrame] = {}
    signal_summaries = []
    matched_summaries = []
    entry_summaries = []

    for top_n in TOP_NS:
        universe_col = f"universe_dynamic_all_top{top_n}"
        data = prepared[prepared[universe_col].fillna(False)].copy()
        signal_trades = _attach_trade_context(_expand_costs(_simulate_base_trades(data)), data)
        matched_rows = _matched_random_rows(data)
        matched_trades = _attach_trade_context(
            _expand_costs(_simulate_base_trades(matched_rows, "baseline_event")), matched_rows
        )
        entry_rows = _entry_only_rows(data)
        entry_trades = _attach_trade_context(
            _expand_costs(_simulate_base_trades(entry_rows, "baseline_event")), entry_rows
        )

        trades_by_topn[top_n] = signal_trades
        matched_by_topn[top_n] = matched_trades
        entry_by_topn[top_n] = entry_trades

        signal_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(data[C2_SIGNAL_COL].fillna(False).sum())]}
        )
        matched_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(matched_rows["baseline_event"].sum())]}
        )
        entry_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(entry_rows["baseline_event"].sum())]}
        )
        for trades, counts, sink in [
            (signal_trades, signal_counts, signal_summaries),
            (matched_trades, matched_counts, matched_summaries),
            (entry_trades, entry_counts, entry_summaries),
        ]:
            for cost, group in trades.groupby("cost_single_side_bps"):
                sink.append(
                    _summary_row(
                        group,
                        int(counts["signals"].sum()),
                        universe=f"dynamic_all_top{top_n}",
                        cost_single_side_bps=cost,
                    )
                )

    signal_summary = pd.DataFrame(signal_summaries)
    matched_summary = pd.DataFrame(matched_summaries)
    entry_summary = pd.DataFrame(entry_summaries)
    all_summary = _compare_baselines(signal_summary, matched_summary, entry_summary)
    ex_may = c2_partition_summary(trades_by_topn, prepared)

    outputs = {
        "features": report_root / "perp_pressure_features_all_eligible.marker",
        "c2_all_eligible_summary": report_root / "c2_all_eligible_summary.csv",
        "c2_ex_may_summary": report_root / "c2_ex_may_summary.csv",
        "c2_monthly_contribution": report_root / "c2_monthly_contribution.csv",
        "c2_symbol_contribution": report_root / "c2_symbol_contribution.csv",
        "c2_liquidity_bucket": report_root / "c2_liquidity_bucket.csv",
        "c2_universe_topn_compare": report_root / "c2_universe_topn_compare.csv",
        "c2_matched_baseline": report_root / "c2_matched_baseline.csv",
        "c2_entry_only_baseline": report_root / "c2_entry_only_baseline.csv",
        "c2_leave_one_month_out": report_root / "c2_leave_one_month_out.csv",
        "c2_portfolio_concurrency": report_root / "c2_portfolio_concurrency.csv",
        "candidate_list": report_root / "candidate_list.md",
    }

    all_summary.to_csv(outputs["c2_all_eligible_summary"], index=False)
    ex_may.to_csv(outputs["c2_ex_may_summary"], index=False)
    c2_contribution(trades_by_topn, "month").to_csv(outputs["c2_monthly_contribution"], index=False)
    c2_contribution(trades_by_topn, "symbol").to_csv(outputs["c2_symbol_contribution"], index=False)
    c2_liquidity_bucket(trades_by_topn, prepared).to_csv(outputs["c2_liquidity_bucket"], index=False)
    all_summary.to_csv(outputs["c2_universe_topn_compare"], index=False)
    matched_summary.to_csv(outputs["c2_matched_baseline"], index=False)
    entry_summary.to_csv(outputs["c2_entry_only_baseline"], index=False)
    c2_leave_one_month_out(trades_by_topn, prepared).to_csv(outputs["c2_leave_one_month_out"], index=False)
    c2_portfolio_concurrency(trades_by_topn).to_csv(outputs["c2_portfolio_concurrency"], index=False)
    write_candidate_list(report_root, all_summary, ex_may)
    outputs["features"].write_text("features are stored under data/processed/v0_3", encoding="utf-8")
    return outputs


def write_v03_reports_from_trades(
    trades_by_topn: dict[int, pd.DataFrame],
    matched_by_topn: dict[int, pd.DataFrame],
    entry_by_topn: dict[int, pd.DataFrame],
    signal_context: pd.DataFrame,
    matched_signal_counts: dict[int, int] | None = None,
    entry_signal_counts: dict[int, int] | None = None,
    report_root: Path = Path("reports/v0_3"),
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    matched_signal_counts = matched_signal_counts or {}
    entry_signal_counts = entry_signal_counts or {}
    signal_summaries = []
    matched_summaries = []
    entry_summaries = []

    for top_n in TOP_NS:
        universe = f"dynamic_all_top{top_n}"
        universe_col = f"universe_dynamic_all_top{top_n}"
        signal_n = int(signal_context[universe_col].fillna(False).sum()) if universe_col in signal_context else 0

        signal_trades = trades_by_topn.get(top_n, pd.DataFrame())
        matched_trades = matched_by_topn.get(top_n, pd.DataFrame())
        entry_trades = entry_by_topn.get(top_n, pd.DataFrame())
        for trades, count, sink in [
            (signal_trades, signal_n, signal_summaries),
            (matched_trades, matched_signal_counts.get(top_n, 0), matched_summaries),
            (entry_trades, entry_signal_counts.get(top_n, 0), entry_summaries),
        ]:
            if trades.empty:
                continue
            for cost, group in trades.groupby("cost_single_side_bps"):
                sink.append(
                    _summary_row(
                        group,
                        count,
                        universe=universe,
                        cost_single_side_bps=cost,
                    )
                )

    signal_summary = pd.DataFrame(signal_summaries)
    matched_summary = pd.DataFrame(matched_summaries)
    entry_summary = pd.DataFrame(entry_summaries)
    all_summary = _compare_baselines(signal_summary, matched_summary, entry_summary)
    ex_may = c2_partition_summary(trades_by_topn, signal_context)

    outputs = {
        "features": report_root / "perp_pressure_features_all_eligible.marker",
        "c2_all_eligible_summary": report_root / "c2_all_eligible_summary.csv",
        "c2_ex_may_summary": report_root / "c2_ex_may_summary.csv",
        "c2_monthly_contribution": report_root / "c2_monthly_contribution.csv",
        "c2_symbol_contribution": report_root / "c2_symbol_contribution.csv",
        "c2_liquidity_bucket": report_root / "c2_liquidity_bucket.csv",
        "c2_universe_topn_compare": report_root / "c2_universe_topn_compare.csv",
        "c2_matched_baseline": report_root / "c2_matched_baseline.csv",
        "c2_entry_only_baseline": report_root / "c2_entry_only_baseline.csv",
        "c2_leave_one_month_out": report_root / "c2_leave_one_month_out.csv",
        "c2_portfolio_concurrency": report_root / "c2_portfolio_concurrency.csv",
        "candidate_list": report_root / "candidate_list.md",
    }

    all_summary.to_csv(outputs["c2_all_eligible_summary"], index=False)
    ex_may.to_csv(outputs["c2_ex_may_summary"], index=False)
    c2_contribution(trades_by_topn, "month").to_csv(outputs["c2_monthly_contribution"], index=False)
    c2_contribution(trades_by_topn, "symbol").to_csv(outputs["c2_symbol_contribution"], index=False)
    c2_liquidity_bucket(trades_by_topn, signal_context).to_csv(outputs["c2_liquidity_bucket"], index=False)
    all_summary.to_csv(outputs["c2_universe_topn_compare"], index=False)
    matched_summary.to_csv(outputs["c2_matched_baseline"], index=False)
    entry_summary.to_csv(outputs["c2_entry_only_baseline"], index=False)
    c2_leave_one_month_out(trades_by_topn, signal_context).to_csv(
        outputs["c2_leave_one_month_out"], index=False
    )
    c2_portfolio_concurrency(trades_by_topn).to_csv(outputs["c2_portfolio_concurrency"], index=False)
    write_candidate_list(report_root, all_summary, ex_may)
    outputs["features"].write_text("features are stored under data/processed/v0_3", encoding="utf-8")
    return outputs


def write_v03_reports(
    features: pd.DataFrame,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = Path("reports/v0_3"),
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    prepared = add_dynamic_all_eligible_universe(features, instruments, config, TOP_NS)
    prepared = prepared[
        pd.to_numeric(prepared["dynamic_all_rank"], errors="coerce") <= MAX_TOP_N
    ].copy()
    prepared = _add_v03_report_columns(prepared, config)

    trades_by_topn: dict[int, pd.DataFrame] = {}
    matched_by_topn: dict[int, pd.DataFrame] = {}
    entry_by_topn: dict[int, pd.DataFrame] = {}
    signal_summaries = []
    matched_summaries = []
    entry_summaries = []

    for top_n in TOP_NS:
        universe_col = f"universe_dynamic_all_top{top_n}"
        data = prepared[prepared[universe_col].fillna(False)].copy()
        signal_trades = _attach_trade_context(_expand_costs(_simulate_base_trades(data)), data)
        matched_rows = _matched_random_rows(data)
        matched_trades = _attach_trade_context(
            _expand_costs(_simulate_base_trades(matched_rows, "baseline_event")), matched_rows
        )
        entry_rows = _entry_only_rows(data)
        entry_trades = _attach_trade_context(
            _expand_costs(_simulate_base_trades(entry_rows, "baseline_event")), entry_rows
        )

        trades_by_topn[top_n] = signal_trades
        matched_by_topn[top_n] = matched_trades
        entry_by_topn[top_n] = entry_trades

        signal_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(data[C2_SIGNAL_COL].fillna(False).sum())]}
        )
        matched_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(matched_rows["baseline_event"].sum())]}
        )
        entry_counts = pd.DataFrame(
            {"universe": [f"dynamic_all_top{top_n}"], "signals": [int(entry_rows["baseline_event"].sum())]}
        )
        for trades, counts, sink in [
            (signal_trades, signal_counts, signal_summaries),
            (matched_trades, matched_counts, matched_summaries),
            (entry_trades, entry_counts, entry_summaries),
        ]:
            for cost, group in trades.groupby("cost_single_side_bps"):
                sink.append(
                    _summary_row(
                        group,
                        int(counts["signals"].sum()),
                        universe=f"dynamic_all_top{top_n}",
                        cost_single_side_bps=cost,
                    )
                )

    signal_summary = pd.DataFrame(signal_summaries)
    matched_summary = pd.DataFrame(matched_summaries)
    entry_summary = pd.DataFrame(entry_summaries)
    all_summary = _compare_baselines(signal_summary, matched_summary, entry_summary)
    ex_may = c2_partition_summary(trades_by_topn, prepared)

    outputs = {
        "features": report_root / "perp_pressure_features_all_eligible.marker",
        "c2_all_eligible_summary": report_root / "c2_all_eligible_summary.csv",
        "c2_ex_may_summary": report_root / "c2_ex_may_summary.csv",
        "c2_monthly_contribution": report_root / "c2_monthly_contribution.csv",
        "c2_symbol_contribution": report_root / "c2_symbol_contribution.csv",
        "c2_liquidity_bucket": report_root / "c2_liquidity_bucket.csv",
        "c2_universe_topn_compare": report_root / "c2_universe_topn_compare.csv",
        "c2_matched_baseline": report_root / "c2_matched_baseline.csv",
        "c2_entry_only_baseline": report_root / "c2_entry_only_baseline.csv",
        "c2_leave_one_month_out": report_root / "c2_leave_one_month_out.csv",
        "c2_portfolio_concurrency": report_root / "c2_portfolio_concurrency.csv",
        "candidate_list": report_root / "candidate_list.md",
    }

    all_summary.to_csv(outputs["c2_all_eligible_summary"], index=False)
    ex_may.to_csv(outputs["c2_ex_may_summary"], index=False)
    c2_contribution(trades_by_topn, "month").to_csv(outputs["c2_monthly_contribution"], index=False)
    c2_contribution(trades_by_topn, "symbol").to_csv(outputs["c2_symbol_contribution"], index=False)
    c2_liquidity_bucket(trades_by_topn, prepared).to_csv(outputs["c2_liquidity_bucket"], index=False)
    all_summary.to_csv(outputs["c2_universe_topn_compare"], index=False)
    matched_summary.to_csv(outputs["c2_matched_baseline"], index=False)
    entry_summary.to_csv(outputs["c2_entry_only_baseline"], index=False)
    c2_leave_one_month_out(trades_by_topn, prepared).to_csv(outputs["c2_leave_one_month_out"], index=False)
    c2_portfolio_concurrency(trades_by_topn).to_csv(outputs["c2_portfolio_concurrency"], index=False)
    write_candidate_list(report_root, all_summary, ex_may)
    outputs["features"].write_text("features are stored under data/processed/v0_3", encoding="utf-8")
    return outputs


def write_v03_reports_from_feature_path(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = Path("reports/v0_3"),
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    ranks = build_dynamic_rank_table(turnover, instruments, config)
    del turnover
    if ranks.empty:
        raise ValueError("No dynamic all-eligible ranks could be computed for v0.3.")

    top_symbols = sorted(ranks[ranks["dynamic_all_rank"] <= MAX_TOP_N]["symbol"].dropna().unique().tolist())
    trades_frames: dict[int, list[pd.DataFrame]] = {top_n: [] for top_n in TOP_NS}
    matched_frames: dict[int, list[pd.DataFrame]] = {top_n: [] for top_n in TOP_NS}
    entry_frames: dict[int, list[pd.DataFrame]] = {top_n: [] for top_n in TOP_NS}
    matched_signal_counts: dict[int, int] = {top_n: 0 for top_n in TOP_NS}
    entry_signal_counts: dict[int, int] = {top_n: 0 for top_n in TOP_NS}
    signal_context_frames: list[pd.DataFrame] = []

    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "liquidity_bucket",
        "dynamic_all_trailing_turnover",
        "btc_market_state",
        "symbol_volatility_percentile",
        C2_SIGNAL_COL,
        *[f"universe_dynamic_all_top{top_n}" for top_n in TOP_NS],
    ]

    for symbol in top_symbols:
        data = _read_v03_symbol_features(feature_path, ranks, symbol, config)
        if data.empty:
            continue
        signal_rows = data[data[C2_SIGNAL_COL].fillna(False)]
        if not signal_rows.empty:
            signal_context_frames.append(signal_rows[[col for col in context_cols if col in signal_rows.columns]])

        for top_n in TOP_NS:
            universe_col = f"universe_dynamic_all_top{top_n}"
            symbol_data = data[data[universe_col].fillna(False)].copy()
            if symbol_data.empty:
                continue
            signal_trades = _attach_trade_context(
                _expand_costs(_simulate_base_trades(symbol_data)),
                symbol_data,
            )
            if not signal_trades.empty:
                trades_frames[top_n].append(signal_trades)

            matched_rows = _matched_random_rows(symbol_data)
            matched_signal_counts[top_n] += _baseline_signal_count(matched_rows)
            matched_trades = _attach_trade_context(
                _expand_costs(_simulate_base_trades(matched_rows, "baseline_event")),
                matched_rows,
            )
            if not matched_trades.empty:
                matched_frames[top_n].append(matched_trades)

            entry_rows = _entry_only_rows(symbol_data)
            entry_signal_counts[top_n] += _baseline_signal_count(entry_rows)
            entry_trades = _attach_trade_context(
                _expand_costs(_simulate_base_trades(entry_rows, "baseline_event")),
                entry_rows,
            )
            if not entry_trades.empty:
                entry_frames[top_n].append(entry_trades)

    signal_context = _concat_or_empty(signal_context_frames)
    trades_by_topn = {top_n: _concat_or_empty(frames) for top_n, frames in trades_frames.items()}
    matched_by_topn = {top_n: _concat_or_empty(frames) for top_n, frames in matched_frames.items()}
    entry_by_topn = {top_n: _concat_or_empty(frames) for top_n, frames in entry_frames.items()}
    return write_v03_reports_from_trades(
        trades_by_topn,
        matched_by_topn,
        entry_by_topn,
        signal_context,
        matched_signal_counts,
        entry_signal_counts,
        report_root,
    )
