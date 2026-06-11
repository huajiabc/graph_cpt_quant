from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest import EntryPolicy, simulate_entry_policy_trades
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir, raw_path, read_parquet
from pressure_graph.reports.v03 import (
    V03_REQUIRED_COLUMNS,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _concat_or_empty,
    _downcast_frame,
    _event_flags,
    _month_start,
    _read_existing_columns,
    build_dynamic_rank_table,
)
from pressure_graph.reports.v04 import _rank_table_with_lookback
from pressure_graph.reports.v06a31 import _entry_state_asof


REPORT_ROOT = Path("reports/v0_6b_flush_reversal")
TOP_NS = [30, 50, 100]
COST_BPS = [5, 10, 20, 30, 50]
R0_RECLAIM = EntryPolicy("R0_reclaim_signal_close_valid_8_bars", "pullback_reclaim", valid_bars=8, pullback_pct=0.0)
R1_RECLAIM = EntryPolicy("R1_pullback_1.0pct_reclaim_signal_close_valid_8_bars", "pullback_reclaim", valid_bars=8, pullback_pct=0.010)
NEXT_OPEN = EntryPolicy("B0_sharp_drop_next_open", "next_open")
P0_PULLBACK = EntryPolicy("B1_touch_signal_close_only_valid_8_bars", "pullback", valid_bars=8, pullback_pct=0.0)
P1_PULLBACK = EntryPolicy("B1_pullback_1.0pct_only_valid_8_bars", "pullback", valid_bars=8, pullback_pct=0.010)
ROLLING_BARS_30D = 30 * 24 * 4
MIN_BARS_14D = 14 * 24 * 4


@dataclass(frozen=True)
class FlushPath:
    path: str
    signal_col: str
    entry_policy: EntryPolicy
    pullback_only_policy: EntryPolicy
    execution_rule: str
    allowed_entry_states: tuple[str, ...]
    rationale: str


PATHS = [
    FlushPath(
        path="FR1_1h_flush_reclaim",
        signal_col="fr1_signal_event",
        entry_policy=R0_RECLAIM,
        pullback_only_policy=P0_PULLBACK,
        execution_rule="vol_regime_fast",
        allowed_entry_states=("BTC_up", "BTC_chop"),
        rationale="1h sharp drop plus volume capitulation and OI down, then reclaim signal close.",
    ),
    FlushPath(
        path="FR2_4h_flush_deeper_reclaim",
        signal_col="fr2_signal_event",
        entry_policy=R1_RECLAIM,
        pullback_only_policy=P1_PULLBACK,
        execution_rule="swing",
        allowed_entry_states=("BTC_up", "BTC_chop"),
        rationale="4h sharp drop while BTC is not down hard, then 1pct pullback/reclaim.",
    ),
    FlushPath(
        path="FR3_funding_flush_reclaim",
        signal_col="fr3_signal_event",
        entry_policy=R0_RECLAIM,
        pullback_only_policy=P0_PULLBACK,
        execution_rule="vol_regime_fast",
        allowed_entry_states=("BTC_up", "BTC_chop"),
        rationale="high funding plus symbol flush and OI down, then reclaim signal close.",
    ),
]


def _required_columns() -> list[str]:
    return list(
        dict.fromkeys(
            [
                *V03_REQUIRED_COLUMNS,
                "btc_ret_1h",
                "btc_volatility_4h",
                "volume_1h_percentile",
                "oi_value_delta_1h_percentile",
                "oi_delta_1h_percentile",
                "oi_delta_4h_percentile",
            ]
        )
    )


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _rule(name: str, config: ExperimentConfig) -> tuple[ExecutionRule, Callable[[pd.Series], ExecutionRule] | None]:
    if name == "swing":
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48), None
    if name == "vol_regime_fast":
        return config.execution.rules["fast"], _vol_regime_rule
    raise KeyError(name)


def _read_symbol_features(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    symbol: str,
    config: ExperimentConfig,
) -> pd.DataFrame:
    data = _read_existing_columns(feature_path, _required_columns(), filters=[("symbol", "==", symbol)])
    if data.empty:
        return data
    data = _downcast_frame(data)
    data["month_start"] = _month_start(data["bar_open_time"])
    data = data.merge(rank30, on=["month_start", "symbol"], how="left")
    data = data.merge(rank90, on=["month_start", "symbol"], how="left")
    data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= max(TOP_NS)].copy()
    if data.empty:
        return data
    data["dynamic_all_rank"] = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    data["turnover_rank_90d"] = pd.to_numeric(data["turnover_rank_90d"], errors="coerce")
    data = _add_v03_report_columns(data, config)
    return _add_flush_columns(data, config)


def _rolling_q(series: pd.Series, q: float) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .shift(1)
        .rolling(ROLLING_BARS_30D, min_periods=MIN_BARS_14D)
        .quantile(q)
    )


def _add_flush_columns(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)
    ret_1h = pd.to_numeric(out.get("ret_1h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")
    ret_4h_pct = pd.to_numeric(out.get("ret_4h_percentile"), errors="coerce")
    volume_z_1h = pd.to_numeric(out.get("volume_z_1h"), errors="coerce")
    volume_z_4h = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    oi_1h = pd.to_numeric(out.get("oi_value_delta_1h_percentile"), errors="coerce")
    oi_4h = pd.to_numeric(out.get("oi_value_delta_4h_percentile"), errors="coerce")
    funding_pct = pd.to_numeric(out.get("funding_percentile"), errors="coerce")
    btc_ret_4h = pd.to_numeric(out.get("btc_ret_4h"), errors="coerce")
    btc_state = out["btc_market_state"].astype("string")

    out["ret_1h_q10_prior_30d"] = (
        out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)["ret_1h"]
        .apply(lambda series: _rolling_q(series, 0.10))
        .reindex(out.index)
    )
    sharp_1h = ret_1h < pd.to_numeric(out["ret_1h_q10_prior_30d"], errors="coerce")
    btc_up_or_chop = btc_state.isin(["BTC_up", "BTC_chop"])
    btc_not_down_hard = (btc_ret_4h > -0.015) & ~btc_state.eq("BTC_down")
    out["fr1_signal_state"] = warmup & btc_up_or_chop & sharp_1h & (volume_z_1h > 2.0) & (oi_1h < 30)
    out["fr2_signal_state"] = warmup & btc_not_down_hard & (ret_4h_pct < 15) & (volume_z_4h > 2.0) & (oi_4h < 30)
    out["fr3_signal_state"] = (
        warmup & btc_up_or_chop & (funding_pct > 80) & sharp_1h & (ret_4h < 0) & (volume_z_1h > 1.5) & (oi_1h < 30)
    )
    out["neutral_reclaim_state"] = warmup & btc_up_or_chop & (volume_z_4h.abs() <= 0.5)
    out["matched_random_state"] = warmup & btc_up_or_chop & ~(out["fr1_signal_state"] | out["fr2_signal_state"] | out["fr3_signal_state"])
    out["bearish_continuation_state"] = warmup & btc_state.eq("BTC_down") & sharp_1h & (volume_z_1h > 1.5)
    for state_col in [
        "fr1_signal_state",
        "fr2_signal_state",
        "fr3_signal_state",
        "neutral_reclaim_state",
        "matched_random_state",
        "bearish_continuation_state",
    ]:
        event_col = state_col.replace("_state", "_event")
        out[event_col] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[state_col]
            .apply(lambda series: _event_flags(series, config.events.cooldown_bars_4h))
            .reindex(out.index)
        )
    hashed = pd.util.hash_pandas_object(out[["symbol", "bar_open_time"]], index=False)
    out["matched_random_event"] = out["matched_random_event"] & ((hashed % 37) == 0)
    out["liquidity_bucket"] = pd.cut(
        pd.to_numeric(out["dynamic_all_rank"], errors="coerce"),
        bins=[0, 10, 30, 50, 100, np.inf],
        labels=["rank_1_10", "rank_11_30", "rank_31_50", "rank_51_100", "rank_101_plus"],
    ).astype("string")
    out["core_liquidity"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] <= 50)
    out["transient_hot"] = (out["dynamic_all_rank"] <= 30) & (out["turnover_rank_90d"] > 50)
    return out


def _top_symbols(ranks: pd.DataFrame) -> list[str]:
    return sorted(
        ranks[pd.to_numeric(ranks["dynamic_all_rank"], errors="coerce") <= max(TOP_NS)]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )


def _attach_context(trades: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    context_cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "dynamic_all_trailing_turnover",
        "liquidity_bucket",
        "btc_market_state",
        "symbol_volatility_percentile",
        "core_liquidity",
        "transient_hot",
        "turnover_rank_90d",
    ]
    context = data[[col for col in context_cols if col in data.columns]].drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )
    out = trades.merge(
        context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    out = out.rename(columns={"btc_market_state": "btc_state_at_signal"})
    entry_context = data[["exchange", "symbol", "feature_time", "btc_market_state"]].copy()
    return _entry_state_asof(out, entry_context)


def _expand_costs(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frames = []
    for cost in COST_BPS:
        data = trades.copy()
        data["cost_single_side_bps"] = float(cost)
        data["net_return"] = pd.to_numeric(data["gross_return"], errors="coerce") - 2.0 * float(cost) / 10_000.0
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _simulate(
    data: pd.DataFrame,
    path: FlushPath,
    signal_col: str,
    baseline: str,
    policy: EntryPolicy,
    config: ExperimentConfig,
    allowed_entry_states: tuple[str, ...],
) -> pd.DataFrame:
    if not data[signal_col].fillna(False).any():
        return pd.DataFrame()
    rule, resolver = _rule(path.execution_rule, config)
    sim = data.copy()
    sim["__v06b_signal"] = sim[signal_col].fillna(False).astype(bool)
    trades = simulate_entry_policy_trades(
        sim,
        "__v06b_signal",
        path.path,
        policy,
        rule,
        0,
        "sl_first",
        True,
        "__v06b_signal",
        resolver,
    )
    if trades.empty:
        return trades
    out = _attach_context(trades, data)
    out = out[out["btc_state_at_entry"].astype(str).isin(allowed_entry_states)].copy()
    if out.empty:
        return out
    out["path"] = path.path
    out["baseline"] = baseline
    out["entry_policy"] = policy.name
    out["execution_rule"] = path.execution_rule
    return _expand_costs(out)


def _summary_row(trades: pd.DataFrame, signal_n: int, **keys: object) -> dict[str, object]:
    net = pd.to_numeric(trades.get("net_return"), errors="coerce")
    gross = pd.to_numeric(trades.get("gross_return"), errors="coerce")
    exit_reason = trades.get("exit_reason", pd.Series(dtype=str)).astype(str)
    return {
        **keys,
        "signals": int(signal_n),
        "trades": int(len(trades)),
        "fill_rate": float(len(trades) / signal_n) if signal_n else np.nan,
        "gross_expectancy": float(gross.mean()) if len(trades) else np.nan,
        "net_expectancy": float(net.mean()) if len(trades) else np.nan,
        "net_sum": float(net.sum()) if len(trades) else 0.0,
        "tp_rate": float(exit_reason.str.startswith("tp").mean()) if len(trades) else np.nan,
        "sl_rate": float(exit_reason.str.startswith("sl").mean()) if len(trades) else np.nan,
        "timeout_rate": float(exit_reason.eq("max_hold").mean()) if len(trades) else np.nan,
        "max_loss": float(net.min()) if len(trades) else np.nan,
    }


def _path_stats(trades: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    count_lookup = counts.groupby(["top_n", "path", "baseline"], sort=False)["signals"].sum().to_dict()
    rows = []
    for top_n in TOP_NS:
        universe_mask = pd.to_numeric(trades["dynamic_all_rank"], errors="coerce") <= top_n
        for key, group in trades[universe_mask].groupby(
            ["path", "baseline", "entry_policy", "execution_rule", "cost_single_side_bps"],
            sort=False,
            dropna=False,
        ):
            path, baseline, entry_policy, execution_rule, cost = key
            rows.append(
                _summary_row(
                    group,
                    int(count_lookup.get((top_n, path, baseline), 0)),
                    top_n=top_n,
                    path=path,
                    baseline=baseline,
                    entry_policy=entry_policy,
                    execution_rule=execution_rule,
                    cost_single_side_bps=cost,
                )
            )
    return pd.DataFrame(rows)


def _monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])]
    for key, group in sample.groupby(["path", "baseline", "cost_single_side_bps", "month"], sort=False, dropna=False):
        path, baseline, cost, month = key
        rows.append(_summary_row(group, len(group), path=path, baseline=baseline, cost_single_side_bps=cost, month=month))
    return pd.DataFrame(rows)


def _compare_baselines(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return stats
    keys = ["top_n", "path", "cost_single_side_bps"]
    candidate = stats[stats["baseline"].eq("candidate_reclaim")][keys + ["net_expectancy", "sl_rate"]].rename(
        columns={"net_expectancy": "candidate_net", "sl_rate": "candidate_sl_rate"}
    )
    out = stats.merge(candidate, on=keys, how="left")
    out["net_lift_vs_candidate"] = out["net_expectancy"] - out["candidate_net"]
    out["sl_delta_vs_candidate"] = out["sl_rate"] - out["candidate_sl_rate"]
    return out


def _write_notes(report_root: Path, stats: pd.DataFrame) -> None:
    lines = ["# v0.6B Long Flush Reversal", ""]
    lines.append("First pass only: existing price/volume/OI/funding fields, no liquidation/orderbook/on-chain data.")
    lines.append("Entry gate is checked at signal and entry time using no-future as-of BTC state.")
    lines.append("")
    lines.append("Decision: Proxy-only Flush Reversal failed first pass.")
    lines.append("Status: C-/D research result. No paper-live, no parameter tuning for now.")
    lines.append(
        "Interpretation: price/volume/OI/funding proxies are not enough to distinguish leverage-release reversals from bearish continuation."
    )
    lines.append("")
    focus = stats[
        stats["baseline"].eq("candidate_reclaim")
        & stats["cost_single_side_bps"].eq(10)
        & stats["top_n"].isin([30, 50, 100])
    ].copy()
    if focus.empty:
        lines.append("- No candidate reclaim trades produced.")
    else:
        for row in focus.sort_values("net_expectancy", ascending=False).head(12).itertuples(index=False):
            lines.append(
                f"- Top{row.top_n} {row.path}: trades={row.trades}, net10={row.net_expectancy:.4%}, "
                f"fill={row.fill_rate:.2%}, sl={row.sl_rate:.2%}, timeout={row.timeout_rate:.2%}"
            )
    lines.append("")
    lines.append("Decision rule: promote only if candidate_reclaim beats next_open, pullback_only, entry_only, and matched_random within the same TopN slice.")
    lines.append(
        "Next FR attempt should wait for stronger evidence such as liquidation spike, taker-sell exhaustion, orderbook bid rebuild, low sweep + reclaim, or shorter-interval OI flush."
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v06b_flush_reversal(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = build_dynamic_rank_table(turnover, instruments, config)
    rank90 = _rank_table_with_lookback(turnover, instruments, config, 90, "turnover_rank_90d", "trailing_90d_turnover")
    symbols = _top_symbols(rank30)
    del turnover
    trade_frames: list[pd.DataFrame] = []
    count_rows: list[dict[str, object]] = []
    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, config)
        if data.empty:
            continue
        print(f"v0.6B processing {idx}/{len(symbols)} {symbol}", flush=True)
        for path in PATHS:
            variants = [
                ("candidate_reclaim", path.signal_col, path.entry_policy, path.allowed_entry_states),
                ("sharp_drop_next_open", path.signal_col, NEXT_OPEN, path.allowed_entry_states),
                ("sharp_drop_pullback_only", path.signal_col, path.pullback_only_policy, path.allowed_entry_states),
                ("entry_only_reclaim", "neutral_reclaim_event", path.entry_policy, path.allowed_entry_states),
                ("matched_random_reclaim", "matched_random_event", path.entry_policy, path.allowed_entry_states),
                ("bearish_continuation_control", "bearish_continuation_event", path.entry_policy, ("BTC_down",)),
            ]
            for baseline, signal_col, policy, allowed_states in variants:
                for top_n in TOP_NS:
                    count_rows.append(
                        {
                            "top_n": top_n,
                            "path": path.path,
                            "baseline": baseline,
                            "signals": int(
                                (
                                    (pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= top_n)
                                    & data[signal_col].fillna(False)
                                ).sum()
                            ),
                        }
                    )
                trades = _simulate(data, path, signal_col, baseline, policy, config, allowed_states)
                if not trades.empty:
                    trade_frames.append(trades)
    trades = _concat_or_empty(trade_frames)
    counts = pd.DataFrame(count_rows)
    stats = _path_stats(trades, counts)
    baseline = _compare_baselines(stats)
    monthly = _monthly(trades)
    controls = baseline[baseline["baseline"].isin(["bearish_continuation_control", "matched_random_reclaim"])].copy()
    outputs = {
        "frozen_paths": report_root / "00_frozen_paths.csv",
        "path_stats": report_root / "01_path_stats.csv",
        "monthly_attribution": report_root / "02_monthly_attribution.csv",
        "baseline_decomposition": report_root / "03_baseline_decomposition.csv",
        "negative_controls": report_root / "04_negative_controls.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    pd.DataFrame([path.__dict__ for path in PATHS]).to_csv(outputs["frozen_paths"], index=False)
    stats.to_csv(outputs["path_stats"], index=False)
    monthly.to_csv(outputs["monthly_attribution"], index=False)
    baseline.to_csv(outputs["baseline_decomposition"], index=False)
    controls.to_csv(outputs["negative_controls"], index=False)
    _write_notes(report_root, stats)
    return outputs


def run_v06b_flush_reversal_from_features(config: ExperimentConfig) -> dict[str, Path]:
    features_path = config.paths.data_root / "processed" / "v0_3" / "perp_pressure_features_all_eligible.parquet"
    instruments = read_parquet(raw_path(config.paths.data_root, "bybit", "instruments"))
    return write_v06b_flush_reversal(features_path, instruments, config)
