from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.entry_policies import EntryPolicy, _entry_for_policy
from pressure_graph.backtest.simulator import _funding_cost
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir
from pressure_graph.reports.v06c import _build_regime_streaming, _rank_inputs
from pressure_graph.reports.v07a import COST_BPS, RECLAIM_1, SIM_WINDOW_BARS, _add_right_tail_labels, _rule, _signal_window
from pressure_graph.reports.v07a1 import NEXT_OPEN
from pressure_graph.reports.v07b import TOP_N
from pressure_graph.reports.v07b2 import _numeric_context
from pressure_graph.reports.v07c import _density_random_leaders, _rank_bucket_map, _shuffled_leaders, _top_leader_map
from pressure_graph.reports.v07c2 import _add_map_gate, _month_setup


REPORT_ROOT = Path("reports/v0_7d_co_impulse_continuation")
RANDOM_PERMUTATIONS = 20
SHUFFLED_PERMUTATIONS = 10


@dataclass(frozen=True)
class CicCandidate:
    candidate: str
    gate_col: str
    event_col: str
    entry_policy: EntryPolicy
    role: str
    negative_control: bool = False


@dataclass(frozen=True)
class ExitModel:
    exit_type: str
    kind: str
    max_hold_bars: int
    tp: float = 0.0
    sl: float = 0.03
    trail_pct: float = 0.04
    partial_tp: float = 0.05
    partial_size: float = 0.5
    capture_ratio: float = 0.0


CANDIDATES = [
    CicCandidate(
        "CIC1_beta_extreme_continuation",
        "c2_bucket_beta_extreme_overextended",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "primary_extreme_continuation",
    ),
    CicCandidate(
        "CIC2_beta_continuation_broad",
        "c2_beta_continuation",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "broad_continuation",
    ),
    CicCandidate(
        "CIC3_directed_leader_shadow",
        "c2_lbc1_real_directed_leader_beta_continuation",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "directed_shadow",
        True,
    ),
    CicCandidate("MIR1_raw_reference", "c2_mir1_raw", "bullish_volume_shock_event", RECLAIM_1, "reference"),
    CicCandidate(
        "beta_extended_control",
        "c2_bucket_beta_extended",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "mechanism_control",
        True,
    ),
    CicCandidate(
        "future_leader_control",
        "c2_beta_continuation_future_leader_1h",
        "bullish_volume_shock_event",
        RECLAIM_1,
        "future_control",
        True,
    ),
    CicCandidate(
        "no_reclaim_next_open_control",
        "c2_beta_continuation",
        "bullish_volume_shock_event",
        NEXT_OPEN,
        "execution_control",
        True,
    ),
    CicCandidate(
        "no_local_volume_shock_control",
        "c2_beta_continuation",
        "neutral_volume_event",
        RECLAIM_1,
        "local_state_control",
        True,
    ),
]

EXIT_MODELS = [
    ExitModel("E0_vol_regime_fast", "vol_regime_fast", 16),
    ExitModel("E1_fixed_5tp_3sl_12h", "fixed", 48, tp=0.05, sl=0.03),
    ExitModel("E2_partial_5tp_trailing_24h", "partial_trailing", 96, sl=0.03, trail_pct=0.04, partial_tp=0.05, partial_size=0.5),
    ExitModel("E3_trailing_no_tp_24h", "trailing", 96, sl=0.03, trail_pct=0.04),
    ExitModel("E3_trailing_no_tp_48h", "trailing", 192, sl=0.03, trail_pct=0.05),
    ExitModel("E4_mfe_capture_50_24h", "mfe_capture", 96, sl=0.03, capture_ratio=0.50),
    ExitModel("E4_mfe_capture_70_24h", "mfe_capture", 96, sl=0.03, capture_ratio=0.70),
]


def _context_cols(data: pd.DataFrame) -> list[str]:
    cols = [
        "exchange",
        "symbol",
        "feature_time",
        "month",
        "dynamic_all_rank",
        "liquidity_bucket",
        "btc_market_state",
        "symbol_volatility_percentile",
        "volume_impulse_density",
        "volume_z_1h",
        "ret_4h",
        "ret_4h_percentile",
        "c2_beta_extension_bucket",
        "c2_beta_extension_score",
        "c2_leader_prior_1h_ratio",
        "c2_future_leader_1h_ratio",
        "c2_directed_edge_weight_active_1h",
        "leader_beta_cluster_id",
        "future_max_up_4h",
        "hit_5pct_4h",
        "mfe_12h",
        "mae_12h",
        "mfe_24h",
        "mae_24h",
        "hit_10pct_12h",
        "hit_20pct_24h",
        "cluster_id",
        "cluster_size",
        "cluster_impulse_density",
        "cluster_positive_return_ratio",
        "cluster_beta_extreme_density",
        "cluster_symbol_rank_within_cluster",
        "cluster_lag_vs_cluster",
    ]
    return [col for col in cols if col in data.columns]


def _vol_regime_exit(entry_row: pd.Series, config: ExperimentConfig) -> ExecutionRule:
    rule, resolver = _rule("vol_regime_fast", config)
    return resolver(entry_row) if resolver else rule


def _fixed_exit(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: ExecutionRule,
) -> tuple[int, float, str, bool]:
    tp_price = entry_price * (1.0 + rule.tp)
    sl_price = entry_price * (1.0 - rule.sl)
    max_idx = min(entry_idx + rule.max_hold_bars - 1, len(group) - 1)
    for idx in range(entry_idx, max_idx + 1):
        row = group.iloc[idx]
        high_hit = float(row["high"]) >= tp_price
        low_hit = float(row["low"]) <= sl_price
        if high_hit and low_hit:
            return idx, sl_price, "sl_ambiguous", True
        if low_hit:
            return idx, sl_price, "sl", False
        if high_hit:
            return idx, tp_price, "tp", False
    return max_idx, float(group.iloc[max_idx]["close"]), "max_hold", False


def _partial_trailing_exit(group: pd.DataFrame, entry_idx: int, entry_price: float, model: ExitModel) -> tuple[int, float, str, bool, float]:
    tp_price = entry_price * (1.0 + model.partial_tp)
    sl_price = entry_price * (1.0 - model.sl)
    max_idx = min(entry_idx + model.max_hold_bars - 1, len(group) - 1)
    took_partial = False
    running_high = entry_price
    realized = 0.0
    for idx in range(entry_idx, max_idx + 1):
        row = group.iloc[idx]
        running_high = max(running_high, float(row["high"]))
        low_hit = float(row["low"]) <= sl_price
        tp_hit = float(row["high"]) >= tp_price
        if not took_partial and low_hit and tp_hit:
            return idx, sl_price, "sl_ambiguous", True, -model.sl
        if not took_partial and low_hit:
            return idx, sl_price, "sl", False, -model.sl
        if not took_partial and tp_hit:
            took_partial = True
            realized = model.partial_size * model.partial_tp
            sl_price = max(entry_price, running_high * (1.0 - model.trail_pct))
        elif took_partial:
            sl_price = max(sl_price, running_high * (1.0 - model.trail_pct))
            if float(row["low"]) <= sl_price:
                remainder = (sl_price / entry_price - 1.0) * (1.0 - model.partial_size)
                return idx, sl_price, "trailing_stop_after_partial", False, realized + remainder
    close_price = float(group.iloc[max_idx]["close"])
    if took_partial:
        remainder = (close_price / entry_price - 1.0) * (1.0 - model.partial_size)
        return max_idx, close_price, "max_hold_after_partial", False, realized + remainder
    return max_idx, close_price, "max_hold", False, close_price / entry_price - 1.0


def _trailing_exit(group: pd.DataFrame, entry_idx: int, entry_price: float, model: ExitModel) -> tuple[int, float, str, bool]:
    sl_price = entry_price * (1.0 - model.sl)
    max_idx = min(entry_idx + model.max_hold_bars - 1, len(group) - 1)
    running_high = entry_price
    for idx in range(entry_idx, max_idx + 1):
        row = group.iloc[idx]
        running_high = max(running_high, float(row["high"]))
        if running_high > entry_price:
            sl_price = max(sl_price, running_high * (1.0 - model.trail_pct))
        if float(row["low"]) <= sl_price:
            return idx, sl_price, "trailing_stop", False
    return max_idx, float(group.iloc[max_idx]["close"]), "max_hold", False


def _mfe_capture_exit(group: pd.DataFrame, entry_idx: int, entry_price: float, model: ExitModel) -> tuple[int, float, str, bool]:
    max_idx = min(entry_idx + model.max_hold_bars - 1, len(group) - 1)
    future = group.iloc[entry_idx : max_idx + 1]
    max_high = float(pd.to_numeric(future["high"], errors="coerce").max())
    min_low = float(pd.to_numeric(future["low"], errors="coerce").min())
    if min_low <= entry_price * (1.0 - model.sl):
        return entry_idx, entry_price * (1.0 - model.sl), "research_sl_before_capture", False
    capture_return = max(0.0, (max_high / entry_price - 1.0) * model.capture_ratio)
    return max_idx, entry_price * (1.0 + capture_return), f"research_mfe_capture_{int(model.capture_ratio * 100)}", False


def _checkpoint_exit(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    model: ExitModel,
    config: ExperimentConfig,
) -> tuple[int, float, str, bool]:
    rule = _vol_regime_exit(group.iloc[entry_idx], config)
    max_hold_bars = model.max_hold_bars or rule.max_hold_bars
    rule = ExecutionRule(tp=rule.tp, sl=rule.sl, max_hold_bars=max_hold_bars)
    checkpoint_idx = min(entry_idx + 8 - 1, len(group) - 1)
    pre_rule = ExecutionRule(tp=rule.tp, sl=rule.sl, max_hold_bars=checkpoint_idx - entry_idx + 1)
    exit_idx, exit_price, reason, ambiguous = _fixed_exit(group, entry_idx, entry_price, pre_rule)
    if exit_idx < checkpoint_idx or reason != "max_hold":
        return exit_idx, exit_price, reason, ambiguous
    checkpoint = group.iloc[checkpoint_idx]
    checkpoint_close = float(checkpoint["close"])
    window = group.iloc[entry_idx : checkpoint_idx + 1]
    mfe = float(pd.to_numeric(window["high"], errors="coerce").max()) / entry_price - 1.0
    current_return = checkpoint_close / entry_price - 1.0
    if model.kind == "checkpoint_return" and current_return < model.partial_tp:
        return checkpoint_idx, checkpoint_close, f"checkpoint_return_lt_{model.partial_tp:.2%}", False
    if model.kind == "checkpoint_mfe" and mfe < model.capture_ratio:
        return checkpoint_idx, checkpoint_close, f"checkpoint_mfe_lt_{model.capture_ratio:.2%}", False
    remaining = group.iloc[checkpoint_idx:].reset_index(drop=True)
    remaining_rule = ExecutionRule(tp=rule.tp, sl=rule.sl, max_hold_bars=max(1, rule.max_hold_bars - (checkpoint_idx - entry_idx)))
    rel_exit_idx, rel_exit_price, rel_reason, rel_ambiguous = _fixed_exit(remaining, 0, entry_price, remaining_rule)
    return checkpoint_idx + rel_exit_idx, rel_exit_price, rel_reason, rel_ambiguous


def _exit_for_model(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    model: ExitModel,
    config: ExperimentConfig,
) -> tuple[int, float, str, bool, float]:
    if model.kind == "vol_regime_fast":
        rule = _vol_regime_exit(group.iloc[entry_idx], config)
        if model.max_hold_bars:
            rule = ExecutionRule(tp=rule.tp, sl=rule.sl, max_hold_bars=model.max_hold_bars)
        exit_idx, exit_price, reason, ambiguous = _fixed_exit(group, entry_idx, entry_price, rule)
        return exit_idx, exit_price, reason, ambiguous, exit_price / entry_price - 1.0
    if model.kind == "fixed":
        rule = ExecutionRule(tp=model.tp, sl=model.sl, max_hold_bars=model.max_hold_bars)
        exit_idx, exit_price, reason, ambiguous = _fixed_exit(group, entry_idx, entry_price, rule)
        return exit_idx, exit_price, reason, ambiguous, exit_price / entry_price - 1.0
    if model.kind == "partial_trailing":
        return _partial_trailing_exit(group, entry_idx, entry_price, model)
    if model.kind == "trailing":
        exit_idx, exit_price, reason, ambiguous = _trailing_exit(group, entry_idx, entry_price, model)
        return exit_idx, exit_price, reason, ambiguous, exit_price / entry_price - 1.0
    if model.kind == "mfe_capture":
        exit_idx, exit_price, reason, ambiguous = _mfe_capture_exit(group, entry_idx, entry_price, model)
        return exit_idx, exit_price, reason, ambiguous, exit_price / entry_price - 1.0
    if model.kind in {"checkpoint_return", "checkpoint_mfe"}:
        exit_idx, exit_price, reason, ambiguous = _checkpoint_exit(group, entry_idx, entry_price, model, config)
        return exit_idx, exit_price, reason, ambiguous, exit_price / entry_price - 1.0
    raise KeyError(model.kind)


def _simulate_candidate_exit(
    data: pd.DataFrame,
    candidate: CicCandidate,
    model: ExitModel,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, int]:
    rank = pd.to_numeric(data["dynamic_all_rank"], errors="coerce")
    mask = (
        (rank <= TOP_N)
        & data[candidate.gate_col].fillna(False).astype(bool)
        & data[candidate.event_col].fillna(False).astype(bool)
    )
    signal_n = int(mask.sum())
    if signal_n <= 0:
        return pd.DataFrame(), 0
    window = _signal_window(data, mask, max(SIM_WINDOW_BARS, model.max_hold_bars + 10))
    window["__v07d_signal"] = mask.loc[window.index].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    context = data[_context_cols(data)].drop_duplicates(["exchange", "symbol", "feature_time"])
    context_lookup = context.set_index(["exchange", "symbol", "feature_time"]).to_dict("index")
    for _, group in window.sort_values(["exchange", "symbol", "bar_open_time"]).groupby(["exchange", "symbol"], sort=False, observed=True):
        group = group.reset_index(drop=True)
        active_until = -1
        for signal_idx, is_signal in enumerate(group["__v07d_signal"].fillna(False).astype(bool)):
            if not is_signal or signal_idx <= active_until:
                continue
            entry = _entry_for_policy(group, signal_idx, "__v07d_signal", candidate.entry_policy)
            if entry is None:
                continue
            entry_idx, entry_price = entry
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            entry_time = pd.Timestamp(group.iloc[entry_idx]["bar_open_time"])
            asof_gate = group.iloc[:entry_idx].copy()
            asof_gate["feature_time"] = pd.to_datetime(asof_gate["feature_time"], utc=True, errors="coerce")
            asof_gate = asof_gate[asof_gate["feature_time"] <= entry_time]
            if asof_gate.empty or not bool(asof_gate.iloc[-1].get(candidate.gate_col, False)):
                continue
            exit_idx, exit_price, exit_reason, ambiguous, gross = _exit_for_model(group, entry_idx, entry_price, model, config)
            exit_time = pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
            signal_time = pd.Timestamp(group.iloc[signal_idx]["feature_time"])
            funding_cost = _funding_cost(group, entry_time, exit_time)
            key = (str(group.iloc[signal_idx]["exchange"]), str(group.iloc[signal_idx]["symbol"]), signal_time)
            payload = {
                "exchange": key[0],
                "symbol": key[1],
                "candidate": candidate.candidate,
                "candidate_role": candidate.role,
                "negative_control": candidate.negative_control,
                "gate_col": candidate.gate_col,
                "event_col": candidate.event_col,
                "entry_policy": candidate.entry_policy.name,
                "exit_type": model.exit_type,
                "exit_kind": model.kind,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(gross),
                "funding_cost": funding_cost,
                "exit_reason": exit_reason,
                "holding_bars": int(exit_idx - entry_idx + 1),
                "bars_from_signal_to_entry": int(entry_idx - signal_idx),
                "ambiguous_exit": ambiguous,
            }
            payload.update(context_lookup.get(key, {}))
            rows.append(payload)
            active_until = exit_idx
    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, signal_n
    frames = []
    for cost in COST_BPS:
        sample = trades.copy()
        sample["cost_single_side_bps"] = float(cost)
        sample["net_return"] = pd.to_numeric(sample["gross_return"], errors="coerce") - 2.0 * float(cost) / 10_000.0
        sample["net_return_ex_funding"] = sample["net_return"] - pd.to_numeric(sample["funding_cost"], errors="coerce").fillna(0.0)
        frames.append(sample)
    return pd.concat(frames, ignore_index=True), signal_n


def _stream_reports(
    feature_path: Path,
    rank30: pd.DataFrame,
    rank90: pd.DataFrame,
    regime: pd.DataFrame,
    config: ExperimentConfig,
    report_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = report_root / "_v07d_trades_tmp.csv"
    signal_path = report_root / "_v07d_signals_tmp.csv"
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    wrote_trades = False
    wrote_signals = False
    months = sorted(pd.to_datetime(rank30["month_start"], utc=True, errors="coerce").dropna().drop_duplicates().tolist())
    for idx, month_start in enumerate(months, start=1):
        month_start = pd.Timestamp(month_start)
        next_month = pd.Timestamp(months[idx]) if idx < len(months) else month_start + pd.DateOffset(months=1)
        sim_window, edges, symbols = _month_setup(feature_path, rank30, rank90, regime, config, month_start, next_month)
        if sim_window.empty:
            continue
        if "mfe_24h" not in sim_window.columns:
            sim_window = _add_right_tail_labels(sim_window)
        month_candidates = CANDIDATES.copy()
        sample = sim_window[pd.to_numeric(sim_window["dynamic_all_rank"], errors="coerce") <= TOP_N].copy()
        symbols_month = sorted(sample["symbol"].dropna().astype(str).unique())
        rank_buckets = _rank_bucket_map(sample)
        real_leaders = _top_leader_map(edges)
        for permutation in range(RANDOM_PERMUTATIONS):
            gate_col = f"__v07d_density_random_p{permutation:03d}"
            sim_window = _add_map_gate(
                sim_window,
                _density_random_leaders(month_start, symbols_month, rank_buckets, permutation),
                gate_col,
            )
            month_candidates.append(
                CicCandidate(
                    f"density_random_leader_p{permutation:03d}",
                    gate_col,
                    "bullish_volume_shock_event",
                    RECLAIM_1,
                    "density_random_control",
                    True,
                )
            )
        for permutation in range(SHUFFLED_PERMUTATIONS):
            gate_col = f"__v07d_shuffled_p{permutation:03d}"
            sim_window = _add_map_gate(sim_window, _shuffled_leaders(month_start, symbols_month, real_leaders, permutation), gate_col)
            month_candidates.append(
                CicCandidate(
                    f"shuffled_leader_p{permutation:03d}",
                    gate_col,
                    "bullish_volume_shock_event",
                    RECLAIM_1,
                    "shuffled_control",
                    True,
                )
            )
        trade_frames: list[pd.DataFrame] = []
        signal_rows: list[dict[str, object]] = []
        for candidate in month_candidates:
            exits = EXIT_MODELS if candidate.candidate in {"CIC1_beta_extreme_continuation", "CIC2_beta_continuation_broad", "CIC3_directed_leader_shadow", "MIR1_raw_reference", "beta_extended_control"} else [EXIT_MODELS[0]]
            for model in exits:
                trades, signals = _simulate_candidate_exit(sim_window, candidate, model, config)
                signal_rows.append(
                    {
                        "candidate": candidate.candidate,
                        "exit_type": model.exit_type,
                        "signals": signals,
                    }
                )
                if not trades.empty:
                    trade_frames.append(trades)
        month_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        if not month_trades.empty:
            month_trades.to_csv(trade_path, mode="a", header=not wrote_trades, index=False)
            wrote_trades = True
        month_signals = pd.DataFrame(signal_rows)
        if not month_signals.empty:
            month_signals.to_csv(signal_path, mode="a", header=not wrote_signals, index=False)
            wrote_signals = True
        print(f"v0.7D month {idx}/{len(months)} {month_start:%Y-%m} symbols={len(symbols)}", flush=True)
        del sim_window, edges, trade_frames, signal_rows, month_trades, month_signals
        gc.collect()
    trades = pd.read_csv(trade_path, low_memory=False) if wrote_trades else pd.DataFrame()
    signals = pd.read_csv(signal_path, low_memory=False) if wrote_signals else pd.DataFrame()
    for path in [trade_path, signal_path]:
        if path.exists():
            path.unlink()
    if not signals.empty:
        signals = signals.groupby(["candidate", "exit_type"], as_index=False, sort=False, dropna=False)["signals"].sum()
    return trades, signals


def _top10_contribution(values: pd.Series) -> float:
    data = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    if data.empty:
        return np.nan
    total = data.sum()
    if total == 0:
        return np.nan
    take = max(1, int(np.ceil(len(data) * 0.10)))
    return float(data.head(take).sum() / total)


def _summary_metrics(group: pd.DataFrame, signals: int, cost: float) -> dict[str, object]:
    net = pd.to_numeric(group["net_return"], errors="coerce")
    gross = pd.to_numeric(group["gross_return"], errors="coerce")
    mfe24 = pd.to_numeric(group.get("mfe_24h"), errors="coerce")
    mae12 = pd.to_numeric(group.get("mae_12h"), errors="coerce")
    mae24 = pd.to_numeric(group.get("mae_24h"), errors="coerce")
    exits = group.get("exit_reason", pd.Series(dtype=str)).astype(str)
    p05 = net.quantile(0.05) if len(net) else np.nan
    p95 = net.quantile(0.95) if len(net) else np.nan
    return {
        "signals": int(signals),
        "trades": int(len(group)),
        "cost_single_side_bps": float(cost),
        "fill_rate": float(len(group) / signals) if signals else np.nan,
        "gross_expectancy": float(gross.mean()) if len(group) else np.nan,
        "net_expectancy": float(net.mean()) if len(group) else np.nan,
        "net_sum": float(net.sum()) if len(group) else 0.0,
        "tp_rate": float(exits.str.startswith("tp").mean()) if len(group) else np.nan,
        "sl_rate": float(exits.str.startswith("sl").mean()) if len(group) else np.nan,
        "trailing_stop_rate": float(exits.str.contains("trailing").mean()) if len(group) else np.nan,
        "timeout_rate": float(exits.str.contains("max_hold").mean()) if len(group) else np.nan,
        "hit5_4h": float(group.get("hit_5pct_4h", pd.Series(dtype=bool)).fillna(False).mean()) if len(group) else np.nan,
        "hit10_12h": float(group.get("hit_10pct_12h", pd.Series(dtype=bool)).fillna(False).mean()) if len(group) else np.nan,
        "hit20_24h": float(group.get("hit_20pct_24h", pd.Series(dtype=bool)).fillna(False).mean()) if len(group) else np.nan,
        "median_mfe": float(mfe24.median()) if len(group) else np.nan,
        "p75_mfe": float(mfe24.quantile(0.75)) if len(group) else np.nan,
        "p90_mfe": float(mfe24.quantile(0.90)) if len(group) else np.nan,
        "p95_mfe": float(mfe24.quantile(0.95)) if len(group) else np.nan,
        "median_mae": float(mae24.median()) if len(group) else np.nan,
        "p75_mae": float(mae24.quantile(0.75)) if len(group) else np.nan,
        "median_mae_when_hit10_12h": float(mae12[group.get("hit_10pct_12h", pd.Series(False, index=group.index)).fillna(False)].median()) if len(group) else np.nan,
        "payoff_skew": float(p95 / abs(p05)) if pd.notna(p95) and pd.notna(p05) and p05 < 0 else np.nan,
        "top10pct_trade_contribution": _top10_contribution(net),
        "avg_holding_bars": float(pd.to_numeric(group.get("holding_bars"), errors="coerce").mean()) if len(group) else np.nan,
    }


def _aggregate_metrics(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    lookup = signals.set_index(["candidate", "exit_type"])["signals"].to_dict() if not signals.empty else {}
    rows = []
    for key, group in trades.groupby(["candidate", "candidate_role", "exit_type", "cost_single_side_bps"], sort=False, dropna=False):
        candidate, role, exit_type, cost = key
        rows.append(
            {
                "candidate": candidate,
                "candidate_role": role,
                "exit_type": exit_type,
                **_summary_metrics(group, int(lookup.get((candidate, exit_type), 0)), float(cost)),
            }
        )
    return pd.DataFrame(rows)


def _candidate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    sample = metrics[pd.to_numeric(metrics["cost_single_side_bps"], errors="coerce").eq(10.0)].copy()
    wide = sample.rename(columns={"net_expectancy": "net10"})
    for cost in [20, 30, 50]:
        add = metrics[pd.to_numeric(metrics["cost_single_side_bps"], errors="coerce").eq(float(cost))][
            ["candidate", "exit_type", "net_expectancy"]
        ].rename(columns={"net_expectancy": f"net{cost}"})
        wide = wide.merge(add, on=["candidate", "exit_type"], how="left")
    return wide


def _contribution(trades: pd.DataFrame, group_col: str) -> pd.DataFrame:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(10.0)].copy()
    if sample.empty:
        return pd.DataFrame()
    rows = []
    for (candidate, exit_type), group in sample.groupby(["candidate", "exit_type"], sort=False, dropna=False):
        total = pd.to_numeric(group["net_return"], errors="coerce").sum()
        for key, local in group.groupby(group_col, sort=False, dropna=False):
            value = pd.to_numeric(local["net_return"], errors="coerce").sum()
            rows.append(
                {
                    "candidate": candidate,
                    "exit_type": exit_type,
                    group_col: key,
                    "trades": int(len(local)),
                    "net_expectancy": float(pd.to_numeric(local["net_return"], errors="coerce").mean()),
                    "net_sum": float(value),
                    "contribution_pct": float(value / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _portfolio_width(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[
        trades["candidate"].astype(str).isin(["CIC1_beta_extreme_continuation", "CIC2_beta_continuation_broad"])
        & pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").isin([10.0, 20.0])
    ].copy()
    if sample.empty:
        return pd.DataFrame()
    sample["entry_time"] = pd.to_datetime(sample["entry_time"], utc=True, errors="coerce")
    sample["exit_time"] = pd.to_datetime(sample["exit_time"], utc=True, errors="coerce")
    sample["rank_first_come_first_served"] = -sample["entry_time"].astype("int64")
    sample["rank_beta_extension_strength"] = _numeric_context(sample, "c2_beta_extension_score", -np.inf)
    sample["rank_local_volume_shock_strength"] = _numeric_context(sample, "volume_z_1h", -np.inf)
    sample["rank_market_impulse_density"] = _numeric_context(sample, "volume_impulse_density", -np.inf)
    sample["rank_liquidity"] = -_numeric_context(sample, "dynamic_all_rank", 999.0)
    hashed = pd.util.hash_pandas_object(sample[["symbol", "signal_time", "exit_type"]].astype(str), index=False)
    sample["rank_random"] = (hashed % 10_000_000).astype(float)
    rank_cols = [
        "rank_beta_extension_strength",
        "rank_local_volume_shock_strength",
        "rank_market_impulse_density",
        "rank_liquidity",
        "rank_first_come_first_served",
        "rank_random",
    ]
    rows = []
    for (candidate, exit_type, cost), group in sample.groupby(["candidate", "exit_type", "cost_single_side_bps"], sort=False, dropna=False):
        for rank_col in rank_cols:
            ranked = group.sort_values(["entry_time", rank_col, "symbol"], ascending=[True, False, True]).reset_index(drop=True)
            for max_positions in [1, 3, 5, 10, 10_000]:
                active: list[pd.Timestamp] = []
                selected = []
                skipped = 0
                max_concurrent = 0
                for row in ranked.itertuples(index=False):
                    entry_time = pd.Timestamp(row.entry_time)
                    active = [exit for exit in active if exit > entry_time]
                    if len(active) >= max_positions:
                        skipped += 1
                        continue
                    selected.append(row)
                    active.append(pd.Timestamp(row.exit_time))
                    max_concurrent = max(max_concurrent, len(active))
                net = pd.Series([float(getattr(item, "net_return", np.nan)) for item in selected], dtype="float64")
                equity = net.cumsum()
                dd = equity - equity.cummax()
                rows.append(
                    {
                        "candidate": candidate,
                        "exit_type": exit_type,
                        "ranking": rank_col.removeprefix("rank_"),
                        "cost_single_side_bps": cost,
                        "max_positions": "unlimited" if max_positions >= 10_000 else max_positions,
                        "selected_trades": int(len(selected)),
                        "skipped_trades": int(skipped),
                        "net_expectancy": float(net.mean()) if len(net) else np.nan,
                        "total_net": float(net.sum()) if len(net) else 0.0,
                        "max_drawdown_proxy": float(dd.min()) if len(dd) else np.nan,
                        "max_concurrent_positions": int(max_concurrent),
                    }
                )
    return pd.DataFrame(rows)


def _controls(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    sample = _candidate_summary(metrics)
    names = [
        "MIR1_raw_reference",
        "CIC1_beta_extreme_continuation",
        "CIC2_beta_continuation_broad",
        "CIC3_directed_leader_shadow",
        "beta_extended_control",
        "future_leader_control",
        "no_reclaim_next_open_control",
        "no_local_volume_shock_control",
    ]
    controls = sample[sample["candidate"].astype(str).isin(names)].copy()
    random = sample[sample["candidate"].astype(str).str.startswith("density_random_leader_p")]
    shuffled = sample[sample["candidate"].astype(str).str.startswith("shuffled_leader_p")]
    for exit_type, group in controls.groupby("exit_type", sort=False, dropna=False):
        cic1 = group[group["candidate"].eq("CIC1_beta_extreme_continuation")]
        if cic1.empty:
            continue
        real_net = float(cic1.iloc[0]["net10"])
        random_exit = pd.to_numeric(random[random["exit_type"].eq(exit_type)]["net10"], errors="coerce").dropna()
        shuffled_exit = pd.to_numeric(shuffled[shuffled["exit_type"].eq(exit_type)]["net10"], errors="coerce").dropna()
        idx = controls["exit_type"].eq(exit_type)
        controls.loc[idx, "cic1_vs_random_median_lift"] = real_net - (float(random_exit.median()) if len(random_exit) else np.nan)
        controls.loc[idx, "cic1_percentile_vs_random"] = float((random_exit <= real_net).mean()) if len(random_exit) else np.nan
        controls.loc[idx, "cic1_vs_shuffled_median_lift"] = real_net - (float(shuffled_exit.median()) if len(shuffled_exit) else np.nan)
        controls.loc[idx, "cic1_percentile_vs_shuffled"] = float((shuffled_exit <= real_net).mean()) if len(shuffled_exit) else np.nan
    return controls


def _mfe_mae_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    sample = trades[pd.to_numeric(trades["cost_single_side_bps"], errors="coerce").eq(10.0)].copy()
    if sample.empty:
        return pd.DataFrame()
    rows = []
    for (candidate, exit_type), group in sample.groupby(["candidate", "exit_type"], sort=False, dropna=False):
        for field in ["mfe_12h", "mfe_24h", "mae_12h", "mae_24h", "net_return"]:
            values = pd.to_numeric(group.get(field), errors="coerce").dropna()
            rows.append(
                {
                    "candidate": candidate,
                    "exit_type": exit_type,
                    "field": field,
                    "count": int(len(values)),
                    "p10": float(values.quantile(0.10)) if len(values) else np.nan,
                    "p25": float(values.quantile(0.25)) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "p75": float(values.quantile(0.75)) if len(values) else np.nan,
                    "p90": float(values.quantile(0.90)) if len(values) else np.nan,
                    "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _write_notes(report_root: Path, summary: pd.DataFrame, controls: pd.DataFrame) -> None:
    lines = [
        "# v0.7D Co-Impulse Continuation / Right-Tail Sleeve",
        "",
        "Purpose: test whether extreme beta continuation under market impulse context is a high-convexity sleeve rather than a directed leader-beta edge.",
        "",
    ]
    for candidate in ["CIC1_beta_extreme_continuation", "CIC2_beta_continuation_broad", "CIC3_directed_leader_shadow"]:
        sample = summary[(summary["candidate"].eq(candidate)) & (summary["exit_type"].eq("E0_vol_regime_fast"))]
        if not sample.empty:
            row = sample.iloc[0]
            lines.append(
                f"- {candidate} / E0: trades={int(row.get('trades', 0))}, net10={row.get('net10', np.nan):.4%}, "
                f"net20={row.get('net20', np.nan):.4%}, hit10_12h={row.get('hit10_12h', np.nan):.2%}, "
                f"p95_mfe={row.get('p95_mfe', np.nan):.2%}."
            )
    if not controls.empty:
        sample = controls[(controls["candidate"].eq("CIC1_beta_extreme_continuation")) & (controls["exit_type"].eq("E0_vol_regime_fast"))]
        if not sample.empty:
            row = sample.iloc[0]
            lines.append(
                f"- CIC1 percentile vs density-random={row.get('cic1_percentile_vs_random', np.nan):.2%}, "
                f"vs shuffled={row.get('cic1_percentile_vs_shuffled', np.nan):.2%}."
            )
    lines.extend(
        [
            "",
            "Decision rule:",
            "- Promote only as research/paper-live shadow if right-tail exits improve net20 or tail capture without collapsing max_positions=3/5.",
            "- Keep MIR1 primary until future paper-live and execution/orderflow validation support replacement.",
        ]
    )
    (report_root / "candidate_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_v07d_co_impulse_continuation(
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
    trades, signals = _stream_reports(feature_path, rank30, rank90, regime, config, report_root)
    metrics = _aggregate_metrics(trades, signals)
    summary = _candidate_summary(metrics)
    controls = _controls(metrics)
    mfe_mae = _mfe_mae_distribution(trades)
    month = _contribution(trades, "month")
    symbol = _contribution(trades, "symbol")
    portfolio = _portfolio_width(trades)
    outputs = {
        "candidate_summary": report_root / "candidate_summary.csv",
        "exit_comparison": report_root / "exit_comparison.csv",
        "right_tail_metrics": report_root / "right_tail_metrics.csv",
        "mfe_mae_distribution": report_root / "mfe_mae_distribution.csv",
        "month_contribution": report_root / "month_contribution.csv",
        "symbol_contribution": report_root / "symbol_contribution.csv",
        "portfolio_width": report_root / "portfolio_width.csv",
        "controls": report_root / "controls.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    summary.to_csv(outputs["candidate_summary"], index=False)
    summary.to_csv(outputs["exit_comparison"], index=False)
    metrics.to_csv(outputs["right_tail_metrics"], index=False)
    mfe_mae.to_csv(outputs["mfe_mae_distribution"], index=False)
    month.to_csv(outputs["month_contribution"], index=False)
    symbol.to_csv(outputs["symbol_contribution"], index=False)
    portfolio.to_csv(outputs["portfolio_width"], index=False)
    controls.to_csv(outputs["controls"], index=False)
    _write_notes(report_root, summary, controls)
    return outputs
