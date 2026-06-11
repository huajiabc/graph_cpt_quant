from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v03 import (
    C2_SIGNAL_COL,
    MAX_TOP_N,
    V03_TURNOVER_COLUMNS,
    _add_v03_report_columns,
    _attach_trade_context,
    _baseline_signal_count,
    _concat_or_empty,
    _downcast_frame,
    _entry_only_rows,
    _expand_costs,
    _matched_random_rows,
    _month_start,
    _read_existing_columns,
    _simulate_base_trades,
    _summary_row,
    build_dynamic_rank_table,
)


C2_CANDIDATE = "C2_short_squeeze_e4_pullback_swing"
REPORT_ROOT = Path("reports/v0_3a_attribution")
BRIDGE_UNIVERSES = {
    "U1_current_static_symbols": "universe_u1_current_static",
    "U2_current_pool_dynamic_top30": "universe_u2_current_pool_dynamic_top30",
    "U3_all_dynamic_top30_current_overlap": "universe_u3_all_top30_current_overlap",
    "U4_all_dynamic_top30_non_current": "universe_u4_all_top30_non_current",
    "U5_all_dynamic_top30": "universe_dynamic_all_top30",
    "U5_all_dynamic_top50": "universe_dynamic_all_top50",
    "U5_all_dynamic_top100": "universe_dynamic_all_top100",
}
QUALITY_COLUMNS = [
    "oi_value_delta_1h_percentile",
    "oi_value_delta_4h_percentile",
    "funding_percentile",
    "funding_z",
    "ret_15m",
    "ret_1h",
    "ret_4h",
    "volume_z_1h",
    "volume_z_4h",
    "btc_ret_4h",
    "volatility_4h",
    "symbol_volatility_percentile",
    "range_pct",
    "upper_wick_ratio",
    "close_location_value",
    "dynamic_all_rank",
    "current_pool_rank",
]


def _v03a_columns() -> list[str]:
    from pressure_graph.reports.v03 import V03_REQUIRED_COLUMNS

    return list(dict.fromkeys([*V03_REQUIRED_COLUMNS, "universe_static_current_top30"]))


def _read_current_symbols(feature_path: Path) -> set[str]:
    cols = ["symbol", "universe_static_current_top30"]
    data = _read_existing_columns(feature_path, cols)
    if "universe_static_current_top30" not in data.columns:
        return set()
    mask = data["universe_static_current_top30"].fillna(False).astype(bool)
    return set(data.loc[mask, "symbol"].dropna().astype(str).unique())


def _read_v02_targets(report_root: Path) -> tuple[pd.DataFrame, set[tuple[str, pd.Timestamp]]]:
    tick_summary_path = report_root / "tick_execution_comparison.csv"
    trades_path = report_root / "tick_execution_trades.csv"
    rows = []
    signal_keys: set[tuple[str, pd.Timestamp]] = set()
    if tick_summary_path.exists():
        summary = pd.read_csv(tick_summary_path)
        target = summary[
            summary["candidate"].eq(C2_CANDIDATE)
            & summary["fill_policy"].eq("normal_2bp")
            & summary["cost_single_side_bps"].eq(5)
        ]
        for row in target.itertuples(index=False):
            rows.append(
                {
                    "check": "U0_v02_exact_tick_target",
                    "execution": "trade_sequence",
                    "fill_policy": "normal_2bp",
                    "cost_single_side_bps": 5,
                    "target_net_expectancy": row.net_expectancy,
                    "target_trades": row.trades,
                    "target_signals": row.signals,
                    "target_fill_rate": row.fill_rate,
                }
            )
    if trades_path.exists():
        trades = pd.read_csv(trades_path, usecols=["candidate", "fill_policy", "cost_single_side_bps", "symbol", "signal_time"])
        signals = trades[
            trades["candidate"].eq(C2_CANDIDATE)
            & trades["fill_policy"].eq("normal_2bp")
            & trades["cost_single_side_bps"].eq(5)
        ][["symbol", "signal_time"]].drop_duplicates()
        signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True, errors="coerce")
        signal_keys = {
            (str(row.symbol), pd.Timestamp(row.signal_time))
            for row in signals.itertuples(index=False)
            if pd.notna(row.signal_time)
        }
    return pd.DataFrame(rows), signal_keys


def _build_current_pool_rank_table(
    turnover: pd.DataFrame,
    all_ranks: pd.DataFrame,
    current_symbols: set[str],
    instruments: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    if not current_symbols:
        return pd.DataFrame(columns=["month_start", "symbol", "current_pool_rank", "current_pool_trailing_turnover"])
    current_turnover = turnover[turnover["symbol"].isin(current_symbols)].copy()
    ranks = build_dynamic_rank_table(current_turnover, instruments, config)
    if ranks.empty:
        return pd.DataFrame(columns=["month_start", "symbol", "current_pool_rank", "current_pool_trailing_turnover"])
    ranks = ranks.rename(
        columns={
            "dynamic_all_rank": "current_pool_rank",
            "dynamic_all_trailing_turnover": "current_pool_trailing_turnover",
        }
    )
    return ranks[["month_start", "symbol", "current_pool_rank", "current_pool_trailing_turnover"]]


def _read_symbol_data(
    feature_path: Path,
    all_ranks: pd.DataFrame,
    current_ranks: pd.DataFrame,
    symbol: str,
    current_symbols: set[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    data = _read_existing_columns(feature_path, _v03a_columns(), filters=[("symbol", "==", symbol)])
    if data.empty:
        return data
    data = _downcast_frame(data)
    data["month_start"] = _month_start(data["bar_open_time"])
    data = data.merge(all_ranks, on=["month_start", "symbol"], how="left")
    if not current_ranks.empty:
        data = data.merge(current_ranks, on=["month_start", "symbol"], how="left")
    else:
        data["current_pool_rank"] = np.nan
        data["current_pool_trailing_turnover"] = np.nan

    is_current = symbol in current_symbols
    if not is_current:
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= MAX_TOP_N].copy()
    if data.empty:
        return data
    data["dynamic_all_rank"] = pd.to_numeric(data["dynamic_all_rank"], errors="coerce", downcast="integer")
    data["current_pool_rank"] = pd.to_numeric(data["current_pool_rank"], errors="coerce", downcast="integer")
    data = _add_v03_report_columns(data, config)
    data["universe_u1_current_static"] = is_current
    data["universe_u2_current_pool_dynamic_top30"] = is_current & (
        pd.to_numeric(data["current_pool_rank"], errors="coerce") <= 30
    )
    data["universe_u3_all_top30_current_overlap"] = is_current & data["universe_dynamic_all_top30"].fillna(False)
    data["universe_u4_all_top30_non_current"] = (~is_current) & data["universe_dynamic_all_top30"].fillna(False)
    return data


def _summarize_trades_for_universes(
    trades_by_universe: dict[str, pd.DataFrame],
    signal_counts: dict[str, int],
    matched_by_universe: dict[str, pd.DataFrame] | None = None,
    entry_by_universe: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    matched_by_universe = matched_by_universe or {}
    entry_by_universe = entry_by_universe or {}
    for universe, trades in trades_by_universe.items():
        if trades.empty:
            continue
        for cost, group in trades.groupby("cost_single_side_bps", sort=False):
            row = _summary_row(group, signal_counts.get(universe, 0), universe=universe, cost_single_side_bps=cost)
            matched = matched_by_universe.get(universe, pd.DataFrame())
            entry = entry_by_universe.get(universe, pd.DataFrame())
            if not matched.empty:
                m = matched[matched["cost_single_side_bps"].eq(cost)]
                row["matched_random_net"] = pd.to_numeric(m.get("net_return_ex_fee_slippage"), errors="coerce").mean()
            if not entry.empty:
                e = entry[entry["cost_single_side_bps"].eq(cost)]
                row["entry_only_net"] = pd.to_numeric(e.get("net_return_ex_fee_slippage"), errors="coerce").mean()
            row["matched_random_lift"] = row.get("net_expectancy", np.nan) - row.get("matched_random_net", np.nan)
            row["entry_only_lift"] = row.get("net_expectancy", np.nan) - row.get("entry_only_net", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_grouped(
    trades_by_universe: dict[str, pd.DataFrame],
    signal_context: pd.DataFrame,
    group_col: str,
    matched_by_universe: dict[str, pd.DataFrame] | None = None,
    entry_by_universe: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    rows = []
    matched_by_universe = matched_by_universe or {}
    entry_by_universe = entry_by_universe or {}
    for universe, trades in trades_by_universe.items():
        if trades.empty or group_col not in trades.columns:
            continue
        universe_col = BRIDGE_UNIVERSES[universe]
        counts = (
            signal_context[signal_context[universe_col].fillna(False)]
            .groupby(group_col, dropna=False, observed=True)
            .size()
            .to_dict()
            if group_col in signal_context.columns and universe_col in signal_context.columns
            else {}
        )
        for (value, cost), group in trades.groupby(
            [group_col, "cost_single_side_bps"], dropna=False, sort=False, observed=True
        ):
            row = _summary_row(
                group,
                int(counts.get(value, 0)),
                universe=universe,
                **{group_col: value},
                cost_single_side_bps=cost,
            )
            matched = matched_by_universe.get(universe, pd.DataFrame())
            entry = entry_by_universe.get(universe, pd.DataFrame())
            if not matched.empty and group_col in matched.columns:
                m = matched[matched["cost_single_side_bps"].eq(cost) & matched[group_col].eq(value)]
                row["matched_random_net"] = pd.to_numeric(m.get("net_return_ex_fee_slippage"), errors="coerce").mean()
            if not entry.empty and group_col in entry.columns:
                e = entry[entry["cost_single_side_bps"].eq(cost) & entry[group_col].eq(value)]
                row["entry_only_net"] = pd.to_numeric(e.get("net_return_ex_fee_slippage"), errors="coerce").mean()
            row["matched_random_lift"] = row.get("net_expectancy", np.nan) - row.get("matched_random_net", np.nan)
            row["entry_only_lift"] = row.get("net_expectancy", np.nan) - row.get("entry_only_net", np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _leave_one_month_out(trades_by_universe: dict[str, pd.DataFrame], signal_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    months = sorted(signal_context["month"].dropna().unique()) if "month" in signal_context else []
    for universe, trades in trades_by_universe.items():
        if trades.empty:
            continue
        universe_col = BRIDGE_UNIVERSES[universe]
        for excluded in months:
            filtered = trades[~trades["month"].astype(str).eq(str(excluded))]
            signal_n = int(
                (
                    signal_context[universe_col].fillna(False)
                    & ~signal_context["month"].astype(str).eq(str(excluded))
                ).sum()
            )
            for cost, group in filtered.groupby("cost_single_side_bps", sort=False):
                rows.append(
                    _summary_row(
                        group,
                        signal_n,
                        universe=universe,
                        excluded_month=excluded,
                        cost_single_side_bps=cost,
                    )
                )
    return pd.DataFrame(rows)


def _quality_distribution(signal_context: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for universe, universe_col in BRIDGE_UNIVERSES.items():
        if universe_col not in signal_context.columns:
            continue
        sample = signal_context[signal_context[universe_col].fillna(False)]
        if sample.empty:
            continue
        for col in QUALITY_COLUMNS:
            if col not in sample.columns:
                continue
            column = sample.loc[:, col]
            if isinstance(column, pd.DataFrame):
                column = column.iloc[:, 0]
            values = pd.to_numeric(column, errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "universe": universe,
                    "metric": col,
                    "n": len(values),
                    "mean": values.mean(),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                }
            )
    return pd.DataFrame(rows)


def _reproduction_check(
    v02_targets: pd.DataFrame,
    v02_signal_keys: set[tuple[str, pd.Timestamp]],
    signal_context: pd.DataFrame,
    universe_summary: pd.DataFrame,
    v02_bridge_trades: pd.DataFrame,
) -> pd.DataFrame:
    rows = v02_targets.to_dict("records") if not v02_targets.empty else []
    u1 = signal_context[signal_context["universe_u1_current_static"].fillna(False)].copy()
    v03_keys = {
        (str(row.symbol), pd.Timestamp(row.feature_time))
        for row in u1[["symbol", "feature_time"]].drop_duplicates().itertuples(index=False)
    }
    overlap = len(v02_signal_keys & v03_keys)
    rows.append(
        {
            "check": "signal_overlap_v02_normal_2bp_vs_v03_u1",
            "v02_signal_count": len(v02_signal_keys),
            "v03_u1_signal_count": len(v03_keys),
            "overlap_count": overlap,
            "overlap_rate_vs_v02": overlap / len(v02_signal_keys) if v02_signal_keys else np.nan,
        }
    )
    bridge_5 = v02_bridge_trades[v02_bridge_trades["cost_single_side_bps"].eq(5)] if not v02_bridge_trades.empty else pd.DataFrame()
    rows.append(
        {
            "check": "v03_15m_execution_on_v02_signal_set",
            "execution": "15m_bar",
            "cost_single_side_bps": 5,
            "trades": len(bridge_5),
            "net_expectancy": pd.to_numeric(bridge_5.get("net_return_ex_fee_slippage"), errors="coerce").mean()
            if not bridge_5.empty
            else np.nan,
            "gross_expectancy": pd.to_numeric(bridge_5.get("gross_return"), errors="coerce").mean()
            if not bridge_5.empty
            else np.nan,
        }
    )
    u1_5 = universe_summary[
        universe_summary["universe"].eq("U1_current_static_symbols")
        & universe_summary["cost_single_side_bps"].eq(5)
    ]
    if not u1_5.empty:
        row = u1_5.iloc[0].to_dict()
        rows.append(
            {
                "check": "v03_15m_u1_current_static_all_events",
                "execution": "15m_bar",
                "cost_single_side_bps": 5,
                "trades": row.get("trades"),
                "signals": row.get("signals"),
                "net_expectancy": row.get("net_expectancy"),
                "gross_expectancy": row.get("gross_expectancy"),
            }
        )
    return pd.DataFrame(rows)


def _candidate_reclassification(report_root: Path, bridge: pd.DataFrame, ex_may: pd.DataFrame) -> None:
    lines = ["# Crypto Pressure Graph v0.3a Difference Attribution", ""]
    lines.append("C2 remains frozen. This report attributes the gap between v0.2 current-pool strength and v0.3 all-eligible weakness.")
    lines.append("")
    lines.append("## Bridge Readout")
    top = bridge[bridge["cost_single_side_bps"].eq(5)].sort_values("net_expectancy", ascending=False)
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.universe}: trades={row.trades}, signals={row.signals}, "
            f"net5={row.net_expectancy:.4%}, matched_lift={row.matched_random_lift:.4%}, "
            f"entry_lift={row.entry_only_lift:.4%}"
        )
    lines.append("")
    lines.append("## Current Classification")
    lines.append("C2 is a C+ regime-specific attack / pressure overlay until attribution shows a stable universe or regime where absolute net expectancy survives costs.")
    (report_root / "10_candidate_reclassification.md").write_text("\n".join(lines), encoding="utf-8")


def write_v03a_attribution(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    v02_targets, v02_signal_keys = _read_v02_targets(Path("reports/v0_2"))
    current_symbols = _read_current_symbols(feature_path)
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    all_ranks = build_dynamic_rank_table(turnover, instruments, config)
    current_ranks = _build_current_pool_rank_table(turnover, all_ranks, current_symbols, instruments, config)
    del turnover

    ranked_symbols = set(all_ranks[all_ranks["dynamic_all_rank"] <= MAX_TOP_N]["symbol"].dropna().astype(str))
    v02_symbols = {key[0] for key in v02_signal_keys}
    symbols = sorted(ranked_symbols | current_symbols | v02_symbols)

    trade_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    matched_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    entry_frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    signal_counts: dict[str, int] = defaultdict(int)
    matched_counts: dict[str, int] = defaultdict(int)
    entry_counts: dict[str, int] = defaultdict(int)
    signal_context_frames: list[pd.DataFrame] = []
    v02_bridge_frames: list[pd.DataFrame] = []

    context_cols = list(dict.fromkeys([
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "current_pool_rank",
        "liquidity_bucket",
        "dynamic_all_trailing_turnover",
        "btc_market_state",
        "symbol_volatility_percentile",
        *QUALITY_COLUMNS,
        *BRIDGE_UNIVERSES.values(),
        C2_SIGNAL_COL,
    ]))

    for idx, symbol in enumerate(symbols, start=1):
        data = _read_symbol_data(feature_path, all_ranks, current_ranks, symbol, current_symbols, config)
        if data.empty:
            continue
        if idx % 25 == 0:
            print(f"v0.3a processed {idx}/{len(symbols)} symbols", flush=True)

        if v02_signal_keys:
            keys = {(sym, ts) for sym, ts in v02_signal_keys if sym == symbol}
            if keys:
                data["v02_bridge_event"] = [
                    (str(row.symbol), pd.Timestamp(row.feature_time)) in keys
                    for row in data[["symbol", "feature_time"]].itertuples(index=False)
                ]
                v02_trades = _attach_trade_context(
                    _expand_costs(_simulate_base_trades(data, "v02_bridge_event")),
                    data,
                )
                if not v02_trades.empty:
                    v02_bridge_frames.append(v02_trades)

        signal_rows = data[data[C2_SIGNAL_COL].fillna(False)]
        if not signal_rows.empty:
            signal_context_frames.append(signal_rows[[col for col in context_cols if col in signal_rows.columns]])

        for universe, universe_col in BRIDGE_UNIVERSES.items():
            sample = data[data[universe_col].fillna(False)].copy()
            if sample.empty:
                continue
            signal_counts[universe] += int(sample[C2_SIGNAL_COL].fillna(False).sum())
            trades = _attach_trade_context(_expand_costs(_simulate_base_trades(sample)), sample)
            if not trades.empty:
                trade_frames[universe].append(trades)
            matched_rows = _matched_random_rows(sample)
            matched_counts[universe] += _baseline_signal_count(matched_rows)
            matched = _attach_trade_context(
                _expand_costs(_simulate_base_trades(matched_rows, "baseline_event")),
                matched_rows,
            )
            if not matched.empty:
                matched_frames[universe].append(matched)
            entry_rows = _entry_only_rows(sample)
            entry_counts[universe] += _baseline_signal_count(entry_rows)
            entry = _attach_trade_context(
                _expand_costs(_simulate_base_trades(entry_rows, "baseline_event")),
                entry_rows,
            )
            if not entry.empty:
                entry_frames[universe].append(entry)

    trades_by_universe = {key: _concat_or_empty(frames) for key, frames in trade_frames.items()}
    matched_by_universe = {key: _concat_or_empty(frames) for key, frames in matched_frames.items()}
    entry_by_universe = {key: _concat_or_empty(frames) for key, frames in entry_frames.items()}
    signal_context = _concat_or_empty(signal_context_frames)
    v02_bridge_trades = _concat_or_empty(v02_bridge_frames)

    bridge = _summarize_trades_for_universes(
        trades_by_universe,
        signal_counts,
        matched_by_universe,
        entry_by_universe,
    )
    monthly = _summarize_grouped(trades_by_universe, signal_context, "month", matched_by_universe, entry_by_universe)
    leave_one = _leave_one_month_out(trades_by_universe, signal_context)
    symbol_attr = _summarize_grouped(trades_by_universe, signal_context, "symbol", matched_by_universe, entry_by_universe)
    liquidity = _summarize_grouped(
        trades_by_universe,
        signal_context,
        "liquidity_bucket",
        matched_by_universe,
        entry_by_universe,
    )
    regime = _summarize_grouped(
        trades_by_universe,
        signal_context,
        "btc_market_state",
        matched_by_universe,
        entry_by_universe,
    )
    quality = _quality_distribution(signal_context)
    reproduction = _reproduction_check(v02_targets, v02_signal_keys, signal_context, bridge, v02_bridge_trades)

    outputs = {
        "00_reproduction_check": report_root / "00_reproduction_check.csv",
        "01_universe_bridge": report_root / "01_universe_bridge.csv",
        "02_monthly_attribution": report_root / "02_monthly_attribution.csv",
        "03_leave_one_month_out": report_root / "03_leave_one_month_out.csv",
        "04_symbol_attribution": report_root / "04_symbol_attribution.csv",
        "05_liquidity_bucket": report_root / "05_liquidity_bucket.csv",
        "06_cost_by_liquidity": report_root / "06_cost_by_liquidity.csv",
        "07_event_quality_distribution": report_root / "07_event_quality_distribution.csv",
        "08_baseline_lift_by_bucket": report_root / "08_baseline_lift_by_bucket.csv",
        "09_regime_split": report_root / "09_regime_split.csv",
        "10_candidate_reclassification": report_root / "10_candidate_reclassification.md",
    }
    reproduction.to_csv(outputs["00_reproduction_check"], index=False)
    bridge.to_csv(outputs["01_universe_bridge"], index=False)
    monthly.to_csv(outputs["02_monthly_attribution"], index=False)
    leave_one.to_csv(outputs["03_leave_one_month_out"], index=False)
    symbol_attr.to_csv(outputs["04_symbol_attribution"], index=False)
    liquidity.to_csv(outputs["05_liquidity_bucket"], index=False)
    liquidity.to_csv(outputs["06_cost_by_liquidity"], index=False)
    quality.to_csv(outputs["07_event_quality_distribution"], index=False)
    liquidity.to_csv(outputs["08_baseline_lift_by_bucket"], index=False)
    regime.to_csv(outputs["09_regime_split"], index=False)
    _candidate_reclassification(report_root, bridge, pd.DataFrame())
    return outputs
