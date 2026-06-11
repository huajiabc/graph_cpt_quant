from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.minute_execution import simulate_1m_execution
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import RECLAIM_1, _rule
from pressure_graph.reports.v07a1 import _entry_context_asof, _load_1m_cache
from pressure_graph.reports.v07b import TOP_N
from pressure_graph.reports.v07b2 import _numeric_context
from pressure_graph.reports.v07c2 import _month_setup
from pressure_graph.reports.v07d import CicCandidate, EXIT_MODELS, _simulate_candidate_exit


REPORT_ROOT = Path("reports/v0_7d1_cic_execution_integration")
ONE_MIN_COSTS = [10.0, 20.0, 50.0]
FOCAL_MONTH = "2025-08"
E0_VOL_REGIME = next(model for model in EXIT_MODELS if model.exit_type == "E0_vol_regime_fast")

FROZEN_CANDIDATES = [
    CicCandidate("MIR1_reference", "c2_mir1_raw", "bullish_volume_shock_event", RECLAIM_1, "reference"),
    CicCandidate("CIC1_beta_extreme", "c2_bucket_beta_extreme_overextended", "bullish_volume_shock_event", RECLAIM_1, "primary_extreme"),
    CicCandidate("CIC2_beta_broad", "c2_beta_continuation", "bullish_volume_shock_event", RECLAIM_1, "broad"),
]

OVERLAP_CANDIDATES = [
    CicCandidate("MIR1_only_ex_CIC1", "d1_mir1_only_ex_cic1", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("MIR1_intersect_CIC1", "d1_mir1_intersect_cic1", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("CIC1_only_ex_MIR1", "d1_cic1_only_ex_mir1", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("MIR1_or_CIC1", "d1_mir1_or_cic1", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("MIR1_only_ex_CIC2", "d1_mir1_only_ex_cic2", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("MIR1_intersect_CIC2", "d1_mir1_intersect_cic2", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("CIC2_only_ex_MIR1", "d1_cic2_only_ex_mir1", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
    CicCandidate("MIR1_or_CIC2", "d1_mir1_or_cic2", "bullish_volume_shock_event", RECLAIM_1, "overlap"),
]


def _add_overlap_gates(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    mir1 = out["c2_mir1_raw"].fillna(False).astype(bool)
    cic1 = out["c2_bucket_beta_extreme_overextended"].fillna(False).astype(bool)
    cic2 = out["c2_beta_continuation"].fillna(False).astype(bool)
    out["d1_mir1_only_ex_cic1"] = mir1 & ~cic1
    out["d1_mir1_intersect_cic1"] = mir1 & cic1
    out["d1_cic1_only_ex_mir1"] = cic1 & ~mir1
    out["d1_mir1_or_cic1"] = mir1 | cic1
    out["d1_mir1_only_ex_cic2"] = mir1 & ~cic2
    out["d1_mir1_intersect_cic2"] = mir1 & cic2
    out["d1_cic2_only_ex_mir1"] = cic2 & ~mir1
    out["d1_mir1_or_cic2"] = mir1 | cic2
    return out


def _signal_id(frame: pd.DataFrame) -> pd.Series:
    signal_time = pd.to_datetime(frame["signal_time"], utc=True, errors="coerce").astype(str)
    return frame["exchange"].astype(str) + "|" + frame["symbol"].astype(str) + "|" + signal_time + "|" + frame["candidate"].astype(str)


def _normalize_1m(trades: pd.DataFrame, candidate: CicCandidate, context: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["candidate"] = candidate.candidate
    out["candidate_role"] = candidate.role
    out["gate_col"] = candidate.gate_col
    out["baseline"] = "candidate_reclaim"
    out["execution_granularity"] = "1m_bar"
    out["net_return"] = pd.to_numeric(out["net_expectancy"], errors="coerce")
    out["month"] = pd.to_datetime(out["signal_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    out = _entry_context_asof(out, context, candidate.gate_col)
    out = out[out["market_gate_at_entry"].fillna(False).astype(bool)].copy()
    if out.empty:
        return out
    signal_context_cols = [
        col
        for col in [
            "exchange",
            "symbol",
            "feature_time",
            "volume_impulse_density",
            "volume_z_1h",
            "ret_4h_percentile",
            "c2_beta_extension_bucket",
            "c2_beta_extension_score",
            "leader_beta_cluster_id",
        ]
        if col in context.columns
    ]
    signal_context = context[signal_context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    out = out.merge(
        signal_context,
        left_on=["exchange", "symbol", "signal_time"],
        right_on=["exchange", "symbol", "feature_time"],
        how="left",
    ).drop(columns=["feature_time"], errors="ignore")
    out["signal_id"] = _signal_id(out)
    return out


def _prepare_1m_signal_rows(sim_windows: list[pd.DataFrame]) -> pd.DataFrame:
    if not sim_windows:
        return pd.DataFrame()
    cols = sorted(set().union(*(set(frame.columns) for frame in sim_windows)))
    keep = [
        col
        for col in cols
        if col
        in {
            "exchange",
            "symbol",
            "bar_open_time",
            "bar_close_time",
            "feature_time",
            "open",
            "high",
            "low",
            "close",
            "dynamic_all_rank",
            "btc_market_state",
            "bullish_volume_shock_event",
            "c2_mir1_raw",
            "c2_bucket_beta_extreme_overextended",
            "c2_beta_continuation",
            "volume_impulse_density",
            "volume_z_1h",
            "ret_4h_percentile",
            "c2_beta_extension_bucket",
            "c2_beta_extension_score",
            "leader_beta_cluster_id",
        }
    ]
    return pd.concat((frame[keep].copy() for frame in sim_windows), ignore_index=True).drop_duplicates(
        ["exchange", "symbol", "feature_time"]
    )


def _one_min_execution(
    one_min_rows: pd.DataFrame,
    trades_15m: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if one_min_rows.empty:
        return pd.DataFrame([{"status": "no_signals"}]), pd.DataFrame()
    symbols = sorted(one_min_rows["symbol"].dropna().astype(str).unique())
    minute_bars = _load_1m_cache(config, symbols)
    if minute_bars.empty:
        return pd.DataFrame([{"status": "pending_1m_data"}]), pd.DataFrame()
    minute_bars["bar_open_time"] = pd.to_datetime(minute_bars["bar_open_time"], utc=True, errors="coerce")
    start = pd.to_datetime(one_min_rows["feature_time"], utc=True, errors="coerce").min()
    end = pd.to_datetime(one_min_rows["feature_time"], utc=True, errors="coerce").max() + pd.Timedelta(hours=14)
    minute_bars = minute_bars[(minute_bars["bar_open_time"] >= start) & (minute_bars["bar_open_time"] <= end)].copy()
    rule, resolver = _rule("vol_regime_fast", config)
    context_cols = [
        col
        for col in [
            "exchange",
            "symbol",
            "feature_time",
            "btc_market_state",
            "c2_mir1_raw",
            "c2_bucket_beta_extreme_overextended",
            "c2_beta_continuation",
            "volume_impulse_density",
            "volume_z_1h",
            "ret_4h_percentile",
            "c2_beta_extension_bucket",
            "c2_beta_extension_score",
            "leader_beta_cluster_id",
        ]
        if col in one_min_rows.columns
    ]
    context = one_min_rows[context_cols].drop_duplicates(["exchange", "symbol", "feature_time"])
    frames = []
    signal_counts: dict[str, int] = {}
    for candidate in FROZEN_CANDIDATES:
        rows = one_min_rows[
            (pd.to_numeric(one_min_rows["dynamic_all_rank"], errors="coerce") <= TOP_N)
            & one_min_rows[candidate.gate_col].fillna(False).astype(bool)
            & one_min_rows[candidate.event_col].fillna(False).astype(bool)
        ].copy()
        signal_counts[candidate.candidate] = int(len(rows))
        if rows.empty:
            continue
        rows["__v07d1_1m_signal"] = True
        for cost in ONE_MIN_COSTS:
            trades = simulate_1m_execution(
                rows,
                minute_bars,
                "__v07d1_1m_signal",
                candidate.candidate,
                candidate.entry_policy,
                "vol_regime_fast",
                rule,
                cost,
                resolver,
            )
            trades = _normalize_1m(trades, candidate, context)
            if not trades.empty:
                trades["cost_single_side_bps"] = cost
                frames.append(trades)
    one_min_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if one_min_trades.empty:
        return pd.DataFrame([{"status": "no_1m_fills"}]), pd.DataFrame()
    rows = []
    for (candidate, cost), group in one_min_trades.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        exits = group["exit_reason"].astype(str)
        net_1m = float(pd.to_numeric(group["net_return"], errors="coerce").mean())
        comp15 = trades_15m[
            trades_15m["candidate"].astype(str).eq(str(candidate))
            & pd.to_numeric(trades_15m["cost_single_side_bps"], errors="coerce").eq(float(cost))
        ]
        net_15m = float(pd.to_numeric(comp15["net_return"], errors="coerce").mean()) if len(comp15) else np.nan
        rows.append(
            {
                "status": "ok",
                "candidate": candidate,
                "cost_single_side_bps": cost,
                "trades_15m": int(len(comp15)),
                "trades_1m": int(len(group)),
                "net_15m": net_15m,
                "net_1m": net_1m,
                "retention_vs_15m": float(net_1m / net_15m) if net_15m else np.nan,
                "same_bar_ambiguity": float(group["unresolved_1m_same_bar"].fillna(False).mean()),
                "fill_rate": float(len(group) / signal_counts.get(str(candidate), 0))
                if signal_counts.get(str(candidate), 0)
                else np.nan,
                "tp_rate": float(exits.str.startswith("tp").mean()),
                "sl_rate": float(exits.str.startswith("sl").mean()),
                "timeout_rate": float(exits.eq("max_hold").mean()),
                "median_holding_minutes": float(pd.to_numeric(group["holding_minutes"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows), one_min_trades


def _summary_for_trades(trades: pd.DataFrame, candidate_col: str = "candidate") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for (candidate, cost), group in trades.groupby([candidate_col, "cost_single_side_bps"], sort=False, dropna=False):
        net = pd.to_numeric(group["net_return"], errors="coerce")
        exits = group["exit_reason"].astype(str)
        rows.append(
            {
                "candidate": candidate,
                "cost_single_side_bps": cost,
                "trades": int(len(group)),
                "net_expectancy": float(net.mean()),
                "net_sum": float(net.sum()),
                "tp_rate": float(exits.str.startswith("tp").mean()),
                "sl_rate": float(exits.str.startswith("sl").mean()),
                "timeout_rate": float(exits.str.contains("max_hold").mean()),
                "hit10_12h": float(group.get("hit_10pct_12h", pd.Series(False, index=group.index)).fillna(False).mean()),
                "max_symbol_contribution": _max_contribution(group, "symbol", float(cost)),
            }
        )
    return pd.DataFrame(rows)


def _max_contribution(trades: pd.DataFrame, group_col: str, cost: float = 10.0) -> float:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if sample.empty or group_col not in sample.columns:
        return np.nan
    grouped = sample.groupby(group_col, sort=False, dropna=False)["net_return"].sum()
    total = grouped.sum()
    return float((grouped / total).abs().max()) if total else np.nan


def _month_cap_expectancy(trades: pd.DataFrame, cost: float = 20.0, cap: float = 0.35) -> float:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(float(cost))]
    if sample.empty:
        return np.nan
    total = pd.to_numeric(sample["net_return"], errors="coerce").sum()
    cap_value = total * cap if total > 0 else 0.0
    capped = []
    for _, group in sample.groupby("month", sort=False, dropna=False):
        value = pd.to_numeric(group["net_return"], errors="coerce").sum()
        capped.append(min(value, cap_value) if value > 0 and cap_value > 0 else value)
    return float(np.sum(capped) / len(sample)) if len(sample) else np.nan


def _ex_month_net(trades: pd.DataFrame, month: str, cost: float) -> float:
    sample = trades[
        pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(float(cost))
        & ~trades["month"].astype(str).eq(month)
    ]
    if sample.empty:
        return np.nan
    return float(pd.to_numeric(sample["net_return"], errors="coerce").mean())


def _month_concentration(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in trades.groupby("candidate", sort=False, dropna=False):
        cost10 = group[pd.to_numeric(group["cost_single_side_bps"], errors="coerce").eq(10.0)]
        if cost10.empty:
            continue
        rows.append(
            {
                "candidate": candidate,
                "trades": int(len(cost10)),
                "net10": float(pd.to_numeric(cost10["net_return"], errors="coerce").mean()),
                "net20": float(
                    pd.to_numeric(
                        group[pd.to_numeric(group["cost_single_side_bps"], errors="coerce").eq(20.0)]["net_return"],
                        errors="coerce",
                    ).mean()
                ),
                f"ex_{FOCAL_MONTH}_net10": _ex_month_net(group, FOCAL_MONTH, 10),
                f"ex_{FOCAL_MONTH}_net20": _ex_month_net(group, FOCAL_MONTH, 20),
                "month_cap35_net20": _month_cap_expectancy(group, 20, 0.35),
                "max_month_contribution": _max_contribution(group, "month", 10),
                "max_symbol_contribution": _max_contribution(group, "symbol", 10),
            }
        )
    return pd.DataFrame(rows)


def _leave_one_month_out(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, cost), group in trades.groupby(["candidate", "cost_single_side_bps"], sort=False, dropna=False):
        for month in sorted(group["month"].dropna().astype(str).unique()):
            sample = group[~group["month"].astype(str).eq(month)]
            rows.append(
                {
                    "candidate": candidate,
                    "cost_single_side_bps": cost,
                    "excluded_month": month,
                    "remaining_trades": int(len(sample)),
                    "remaining_net_expectancy": float(pd.to_numeric(sample["net_return"], errors="coerce").mean()) if len(sample) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _reconciliation(trades_15m: pd.DataFrame, trades_1m: pd.DataFrame) -> pd.DataFrame:
    def prep(data: pd.DataFrame, suffix: str) -> pd.DataFrame:
        sample = data[pd.to_numeric(data["cost_single_side_bps"], errors="coerce").eq(10.0)].copy()
        if sample.empty:
            return pd.DataFrame()
        sample["signal_time"] = pd.to_datetime(sample["signal_time"], utc=True, errors="coerce")
        sample["signal_id"] = _signal_id(sample)
        keep = ["signal_id", "candidate", "symbol", "signal_time", "entry_time", "exit_time", "exit_reason", "net_return"]
        sample = sample[[col for col in keep if col in sample.columns]]
        return sample.rename(columns={col: f"{col}_{suffix}" for col in ["entry_time", "exit_time", "exit_reason", "net_return"]})

    left = prep(trades_15m, "15m")
    right = prep(trades_1m, "1m")
    if left.empty and right.empty:
        return pd.DataFrame()
    merged = left.merge(right, on=["signal_id", "candidate", "symbol", "signal_time"], how="outer")
    merged["in_15m"] = merged["net_return_15m"].notna()
    merged["in_1m"] = merged["net_return_1m"].notna()
    merged["delta_net10"] = pd.to_numeric(merged["net_return_1m"], errors="coerce") - pd.to_numeric(merged["net_return_15m"], errors="coerce")
    return merged


def _included_vs_excluded(recon: pd.DataFrame) -> pd.DataFrame:
    if recon.empty:
        return pd.DataFrame()
    data = recon.copy()
    data["bucket"] = np.select(
        [
            data["in_15m"].fillna(False) & data["in_1m"].fillna(False),
            data["in_15m"].fillna(False) & ~data["in_1m"].fillna(False),
            ~data["in_15m"].fillna(False) & data["in_1m"].fillna(False),
        ],
        ["in_both", "15m_only", "1m_only"],
        default="neither",
    )
    rows = []
    for (candidate, bucket), group in data.groupby(["candidate", "bucket"], sort=False, dropna=False):
        rows.append(
            {
                "candidate": candidate,
                "bucket": bucket,
                "signals": int(len(group)),
                "net10_15m_avg": float(pd.to_numeric(group.get("net_return_15m"), errors="coerce").mean()),
                "net10_1m_avg": float(pd.to_numeric(group.get("net_return_1m"), errors="coerce").mean()),
                "delta_net10_avg": float(pd.to_numeric(group.get("delta_net10"), errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_integration(trades: pd.DataFrame) -> pd.DataFrame:
    base = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])].copy()
    if base.empty:
        return pd.DataFrame()
    base["signal_id_base"] = base["exchange"].astype(str) + "|" + base["symbol"].astype(str) + "|" + pd.to_datetime(base["signal_time"], utc=True, errors="coerce").astype(str)
    pools = []
    mir1 = base[base["candidate"].eq("MIR1_reference")].copy()
    cic1 = base[base["candidate"].eq("CIC1_beta_extreme")].copy()
    cic2 = base[base["candidate"].eq("CIC2_beta_broad")].copy()
    for name, frame in [("MIR1_primary_only", mir1), ("CIC1_only", cic1), ("CIC2_only", cic2)]:
        local = frame.copy()
        local["portfolio_pool"] = name
        local["priority_score"] = 1.0
        pools.append(local)
    for name, cic in [("MIR1_plus_CIC1_priority", cic1), ("MIR1_plus_CIC2_priority", cic2)]:
        cic_ids = set(cic["signal_id_base"].astype(str))
        local = pd.concat([cic.assign(priority_score=2.0), mir1[~mir1["signal_id_base"].astype(str).isin(cic_ids)].assign(priority_score=1.0)], ignore_index=True)
        local["portfolio_pool"] = name
        pools.append(local)
    data = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    data["entry_time"] = pd.to_datetime(data["entry_time"], utc=True, errors="coerce")
    data["exit_time"] = pd.to_datetime(data["exit_time"], utc=True, errors="coerce")
    data["rank_beta_extension_strength"] = _numeric_context(data, "c2_beta_extension_score", -np.inf)
    data["rank_local_volume_shock_strength"] = _numeric_context(data, "volume_z_1h", -np.inf)
    data["rank_market_impulse_density"] = _numeric_context(data, "volume_impulse_density", -np.inf)
    data["rank_liquidity"] = -_numeric_context(data, "dynamic_all_rank", 999.0)
    data["rank_first_come_first_served"] = -data["entry_time"].astype("int64")
    rows = []
    rank_cols = [
        "priority_score",
        "rank_beta_extension_strength",
        "rank_local_volume_shock_strength",
        "rank_market_impulse_density",
        "rank_liquidity",
        "rank_first_come_first_served",
    ]
    for (pool, cost), group in data.groupby(["portfolio_pool", "cost_single_side_bps"], sort=False, dropna=False):
        for rank_col in rank_cols:
            ranked = group.sort_values(["entry_time", rank_col, "symbol"], ascending=[True, False, True])
            for max_positions in [1, 3, 5, 10, 10_000]:
                active: list[pd.Timestamp] = []
                selected = []
                skipped = 0
                for row in ranked.itertuples(index=False):
                    entry = pd.Timestamp(row.entry_time)
                    active = [exit for exit in active if exit > entry]
                    if len(active) >= max_positions:
                        skipped += 1
                        continue
                    selected.append(row)
                    active.append(pd.Timestamp(row.exit_time))
                net = pd.Series([float(getattr(row, "net_return", np.nan)) for row in selected], dtype="float64")
                equity = net.cumsum()
                dd = equity - equity.cummax()
                rows.append(
                    {
                        "portfolio_pool": pool,
                        "ranking": rank_col.removeprefix("rank_"),
                        "cost_single_side_bps": cost,
                        "max_positions": "unlimited" if max_positions >= 10_000 else max_positions,
                        "selected_trades": int(len(selected)),
                        "skipped_trades": int(skipped),
                        "net_expectancy": float(net.mean()) if len(net) else np.nan,
                        "total_net": float(net.sum()) if len(net) else 0.0,
                        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _stream_15m(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07d1_15m_trades_tmp.csv"
    signal_path = report_root / "_v07d1_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    one_min_windows = []
    months = sorted(pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist())
    candidates = [*FROZEN_CANDIDATES, *OVERLAP_CANDIDATES]
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, _, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
        if sim_window.empty:
            continue
        sim_window = _add_overlap_gates(sim_window)
        one_min_windows.append(sim_window.copy())
        trade_frames = []
        signal_rows = []
        for candidate in candidates:
            trades, signals = _simulate_candidate_exit(sim_window, candidate, E0_VOL_REGIME, config)
            signal_rows.append({"candidate": candidate.candidate, "signals": signals})
            if not trades.empty:
                trades["candidate"] = candidate.candidate
                trade_frames.append(trades)
        month_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        if not month_trades.empty:
            month_trades["signal_id"] = _signal_id(month_trades)
            month_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
            wrote_trades = True
        month_signals = pd.DataFrame(signal_rows)
        if not month_signals.empty:
            month_signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
            wrote_signals = True
        print(f"v0.7D.1 month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, trade_frames, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    if not signals.empty:
        signals = signals.groupby("candidate", as_index=False, sort=False, dropna=False)["signals"].sum()
    return trades, signals, _prepare_1m_signal_rows(one_min_windows)


def _mir1_cic_overlap(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    summary = _summary_for_trades(trades)
    if summary.empty:
        return summary
    signal_lookup = signals.set_index("candidate")["signals"].to_dict() if not signals.empty else {}
    summary["signals"] = summary["candidate"].map(signal_lookup).fillna(0).astype(int)
    summary["month_cap35_net20"] = summary["candidate"].map(
        {
            candidate: _month_cap_expectancy(trades[trades["candidate"].astype(str).eq(str(candidate))], 20, 0.35)
            for candidate in summary["candidate"].astype(str).unique()
        }
    )
    return summary


def _write_notes(report_root: Path, one_min: pd.DataFrame, overlap: pd.DataFrame) -> None:
    lines = [
        "# v0.7D.1 CIC Execution + MIR1 Integration",
        "",
        "Frozen validation for CIC1/CIC2. No beta threshold tuning and no exit tuning; E0 vol_regime_fast only.",
        "",
    ]
    if not one_min.empty and "status" in one_min.columns:
        lines.append(f"- 1m status: {', '.join(sorted(one_min['status'].dropna().astype(str).unique()))}.")
    for candidate in ["CIC1_beta_extreme", "CIC2_beta_broad"]:
        sample = one_min[one_min["candidate"].astype(str).eq(candidate)] if "candidate" in one_min.columns else pd.DataFrame()
        if not sample.empty:
            row = sample[pd.to_numeric(sample["cost_single_side_bps"], errors="coerce").eq(10.0)].head(1)
            if not row.empty:
                row = row.iloc[0]
                lines.append(
                    f"- {candidate} 1m: trades={int(row.get('trades_1m', 0))}, "
                    f"net10={row.get('net_1m', np.nan):.4%}, retention={row.get('retention_vs_15m', np.nan):.2f}."
                )
    if not overlap.empty:
        cic = overlap[
            overlap["candidate"].astype(str).isin(["MIR1_intersect_CIC1", "MIR1_only_ex_CIC1"])
            & pd.to_numeric(overlap["cost_single_side_bps"], errors="coerce").eq(10.0)
        ]
        for row in cic.itertuples(index=False):
            lines.append(f"- {row.candidate}: trades={row.trades}, net10={row.net_expectancy:.4%}.")
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07d1_cic_execution_integration(
    feature_path: Path,
    instruments: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path = REPORT_ROOT,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    rank30, rank90, _ = _rank_inputs(feature_path, instruments, config)
    symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= TOP_N]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    regime = _build_regime_streaming(feature_path, rank30, rank90, symbols, config)
    trades_15m, signals, one_min_rows = _stream_15m(feature_path, rank30, rank90, regime, config, report_root)
    frozen_15m = trades_15m[trades_15m["candidate"].astype(str).isin([item.candidate for item in FROZEN_CANDIDATES])].copy()
    one_min_summary, one_min_trades = _one_min_execution(one_min_rows, frozen_15m, config)
    recon = _reconciliation(frozen_15m, one_min_trades)
    included = _included_vs_excluded(recon)
    overlap = _mir1_cic_overlap(trades_15m, signals)
    month = _month_concentration(frozen_15m)
    loo = _leave_one_month_out(frozen_15m)
    portfolio = _portfolio_integration(frozen_15m)
    outputs = {
        "execution_1m_comparison": report_root / "execution_1m_comparison.csv",
        "one_min_reconciliation": report_root / "1m_reconciliation.csv",
        "included_vs_excluded": report_root / "included_vs_excluded.csv",
        "mir1_cic_overlap": report_root / "mir1_cic_overlap.csv",
        "month_concentration": report_root / "month_concentration.csv",
        "leave_one_month_out": report_root / "leave_one_month_out.csv",
        "portfolio_integration": report_root / "portfolio_integration.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    one_min_summary.to_csv(outputs["execution_1m_comparison"], index=False)
    recon.to_csv(outputs["one_min_reconciliation"], index=False)
    included.to_csv(outputs["included_vs_excluded"], index=False)
    overlap.to_csv(outputs["mir1_cic_overlap"], index=False)
    month.to_csv(outputs["month_concentration"], index=False)
    loo.to_csv(outputs["leave_one_month_out"], index=False)
    portfolio.to_csv(outputs["portfolio_integration"], index=False)
    _write_notes(report_root, one_min_summary, overlap)
    return outputs
