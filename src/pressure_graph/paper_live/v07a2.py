from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.simulator import _funding_cost
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.config.v07a2 import V07A2Baseline, V07A2Candidate, V07A2Config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.reports.v03 import V03_TURNOVER_COLUMNS, _concat_or_empty, _event_flags, _read_existing_columns
from pressure_graph.reports.v06a1 import _read_symbol_features
from pressure_graph.reports.v06c import _market_regime_features, _rank_inputs
from pressure_graph.reports.v07a import _add_motif_columns


REPORT_ROOT = Path("reports/v0_7a2_mir1_paper_live")
PAPER_DATA_ROOT = Path("data/paper/v0_7a2")
SIGNAL_COLUMNS = [
    "signal_id",
    "created_at_utc",
    "candidate",
    "candidate_role",
    "baseline_kind",
    "exchange",
    "symbol",
    "bar_open_time",
    "bar_close_time",
    "feature_time",
    "market_gate",
    "market_gate_time",
    "volume_impulse_density_value",
    "volume_impulse_density_state",
    "local_volume_shock_time",
    "pullback_time",
    "reclaim_time",
    "entry_time",
    "exit_time",
    "universe_top_n",
    "turnover_rank_30d",
    "turnover_rank_90d",
    "btc_state",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_volatility_4h",
    "volume_z_4h",
    "ret_4h",
    "cluster_id",
    "cluster_size",
    "cluster_impulse_density",
    "signal_close",
    "entry_limit",
    "entry_price",
    "entry_valid_until",
    "market_gate_at_signal",
    "market_gate_at_entry",
    "market_gate_at_exit",
    "entry_reason",
    "exit_reason",
    "status",
    "skip_reason",
]
TRADE_COLUMNS = [
    "trade_id",
    "signal_id",
    "candidate",
    "candidate_role",
    "baseline_kind",
    "exchange",
    "symbol",
    "market_gate",
    "market_gate_time",
    "volume_impulse_density_at_signal",
    "volume_impulse_density_at_entry",
    "volume_impulse_density_at_exit",
    "cluster_id_at_signal",
    "cluster_id_at_entry",
    "cluster_size_at_signal",
    "cluster_size_at_entry",
    "cluster_impulse_density_at_signal",
    "cluster_impulse_density_at_entry",
    "beta_extension_bucket_at_signal",
    "beta_extension_score_at_signal",
    "local_volume_shock_strength_at_signal",
    "volume_impulse_density_state",
    "local_volume_shock_time",
    "pullback_time",
    "reclaim_time",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_reason",
    "tp_price",
    "sl_price",
    "max_hold_time",
    "gross_return",
    "net_return_5bp",
    "net_return_10bp",
    "net_return_20bp",
    "net_return_30bp",
    "net_return_50bp",
    "holding_minutes",
    "mae",
    "mfe",
    "tp_first",
    "sl_first",
    "timeout",
    "same_bar_ambiguity",
    "funding_cost",
    "market_gate_at_signal",
    "market_gate_at_entry",
    "market_gate_at_exit",
    "btc_state_at_signal",
    "btc_state_at_entry",
    "btc_state_at_exit",
    "concurrent_positions_at_entry",
    "portfolio_accepted",
    "portfolio_skip_reason",
    "turnover_30d",
]


def _safe_float(value: object, default: float = np.nan) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) else default


def _bool(value: object) -> bool:
    return bool(value) if pd.notna(value) else False


def _cost_suffix(cost: float) -> str:
    return f"{int(cost)}bp" if float(cost).is_integer() else f"{cost:g}bp"


def _vol_regime_rule(row: pd.Series) -> ExecutionRule:
    vol_pct = pd.to_numeric(row.get("symbol_volatility_percentile"), errors="coerce")
    if pd.isna(vol_pct) or vol_pct < 40:
        return ExecutionRule(tp=0.03, sl=0.02, max_hold_bars=16)
    if vol_pct < 80:
        return ExecutionRule(tp=0.04, sl=0.025, max_hold_bars=16)
    return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=16)


def _rule(name: str, row: pd.Series) -> ExecutionRule:
    if name == "vol_regime_fast":
        return _vol_regime_rule(row)
    if name == "swing":
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48)
    raise KeyError(f"unknown v0.7A.2 execution rule: {name}")


def _signal_id(row: pd.Series, name: str, baseline_kind: str) -> str:
    stamp = pd.Timestamp(row["feature_time"]).strftime("%Y%m%dT%H%M%SZ")
    suffix = baseline_kind or "candidate"
    return f"{name}:{suffix}:{row['exchange']}:{row['symbol']}:{stamp}"


def _spec_policy_kind(spec: V07A2Candidate | V07A2Baseline) -> str:
    if spec.entry_policy in {"reclaim", "pullback_reclaim"}:
        return "reclaim"
    if spec.entry_policy in {"pullback", "pullback_only"}:
        return "pullback"
    if spec.entry_policy == "next_open":
        return "next_open"
    if spec.entry_policy == "v9_retest":
        return "v9_retest"
    raise KeyError(f"unknown v0.7A.2 entry policy: {spec.entry_policy}")


def add_v07a2_live_columns(df: pd.DataFrame, config: V07A2Config) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)
    if "bullish_volume_shock_state" not in out.columns:
        out["bullish_volume_shock_state"] = (
            warmup
            & (pd.to_numeric(out.get("volume_z_4h"), errors="coerce") > config.signal.volume_z_4h_min)
            & (pd.to_numeric(out.get("ret_4h"), errors="coerce") > config.signal.ret_4h_min)
        )
    if "bullish_volume_shock_event" not in out.columns:
        out["bullish_volume_shock_event"] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[
                "bullish_volume_shock_state"
            ]
            .apply(lambda series: _event_flags(series, config.signal.cooldown_bars))
            .reindex(out.index)
        )
    if "neutral_volume_event" not in out.columns:
        out["neutral_volume_state"] = warmup & (pd.to_numeric(out.get("volume_z_4h"), errors="coerce").abs() <= 0.5)
        out["neutral_volume_event"] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[
                "neutral_volume_state"
            ]
            .apply(lambda series: _event_flags(series, config.signal.cooldown_bars))
            .reindex(out.index)
        )
    if "matched_random_event" not in out.columns:
        pool = warmup & ~out["bullish_volume_shock_state"].fillna(False).astype(bool)
        hashed = pd.util.hash_pandas_object(out[["symbol", "bar_open_time"]], index=False)
        out["matched_random_event"] = pool & ((hashed % 97) == 0)
    state = out.get("btc_market_state", pd.Series("unknown", index=out.index)).astype(str)
    out["market_btc_up"] = out.get("market_btc_up", state.eq("BTC_up"))
    out["market_btc_chop"] = out.get("market_btc_chop", state.eq("BTC_chop"))
    density = pd.to_numeric(out.get("volume_impulse_density"), errors="coerce")
    if "market_volume_impulse_density_high" not in out.columns:
        out["market_volume_impulse_density_high"] = density >= config.market_graph.density_threshold
    if "market_low_volume_impulse_density" not in out.columns:
        out["market_low_volume_impulse_density"] = density <= config.market_graph.low_density_threshold
    return out


def _market_context_asof(
    group: pd.DataFrame,
    timestamp: pd.Timestamp,
    market_gate: str,
) -> dict[str, object]:
    if group.empty or "feature_time" not in group.columns:
        return {
            "feature_time": pd.NaT,
            "market_gate": False,
            "btc_state": "unknown",
            "volume_impulse_density": np.nan,
        }
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    eligible = group[feature_time <= ts]
    if eligible.empty:
        return {
            "feature_time": pd.NaT,
            "market_gate": False,
            "btc_state": "unknown",
            "volume_impulse_density": np.nan,
        }
    row = eligible.iloc[-1]
    return {
        "feature_time": pd.Timestamp(row.get("feature_time")),
        "market_gate": _bool(row.get(market_gate)),
        "btc_state": str(row.get("btc_market_state", "unknown")),
        "volume_impulse_density": _safe_float(row.get("volume_impulse_density")),
    }


def _base_signal_row(
    row: pd.Series,
    spec: V07A2Candidate | V07A2Baseline,
    role: str,
    created_at: pd.Timestamp,
    baseline_kind: str = "",
) -> dict[str, object]:
    signal_close = _safe_float(row.get("close"))
    gate_value = _bool(row.get(spec.market_gate))
    density = _safe_float(row.get("volume_impulse_density"))
    return {
        "signal_id": _signal_id(row, spec.candidate, baseline_kind),
        "created_at_utc": created_at,
        "candidate": spec.candidate,
        "candidate_role": role,
        "baseline_kind": baseline_kind,
        "exchange": str(row.get("exchange")),
        "symbol": str(row.get("symbol")),
        "bar_open_time": pd.Timestamp(row.get("bar_open_time")),
        "bar_close_time": pd.Timestamp(row.get("bar_close_time")),
        "feature_time": pd.Timestamp(row.get("feature_time")),
        "market_gate": spec.market_gate,
        "market_gate_time": pd.Timestamp(row.get("feature_time")),
        "volume_impulse_density_value": density,
        "volume_impulse_density_state": gate_value,
        "local_volume_shock_time": pd.Timestamp(row.get("feature_time")),
        "pullback_time": pd.NaT,
        "reclaim_time": pd.NaT,
        "entry_time": pd.NaT,
        "exit_time": pd.NaT,
        "universe_top_n": spec.universe_top_n,
        "turnover_rank_30d": _safe_float(row.get("dynamic_all_rank")),
        "turnover_rank_90d": _safe_float(row.get("turnover_rank_90d")),
        "btc_state": str(row.get("btc_market_state", "unknown")),
        "btc_ret_1h": _safe_float(row.get("btc_ret_1h")),
        "btc_ret_4h": _safe_float(row.get("btc_ret_4h")),
        "btc_volatility_4h": _safe_float(row.get("btc_volatility_4h")),
        "volume_z_4h": _safe_float(row.get("volume_z_4h")),
        "ret_4h": _safe_float(row.get("ret_4h")),
        "cluster_id": str(row.get("cluster_id", "")),
        "cluster_size": _safe_float(row.get("cluster_size")),
        "cluster_impulse_density": _safe_float(row.get("cluster_impulse_density")),
        "signal_close": signal_close,
        "entry_limit": signal_close * (1.0 - spec.pullback_pct),
        "entry_price": np.nan,
        "entry_valid_until": pd.Timestamp(row.get("feature_time"))
        + pd.Timedelta(minutes=15 * spec.valid_bars),
        "market_gate_at_signal": gate_value,
        "market_gate_at_entry": False,
        "market_gate_at_exit": False,
        "entry_reason": "",
        "exit_reason": "",
        "status": "detected",
        "skip_reason": "",
    }


def _entry_for_signal(
    group: pd.DataFrame,
    signal_idx: int,
    spec: V07A2Candidate | V07A2Baseline,
) -> dict[str, object]:
    signal = group.iloc[signal_idx]
    signal_close = _safe_float(signal.get("close"))
    trigger = signal_close * (1.0 - spec.pullback_pct)
    valid_end = min(signal_idx + spec.valid_bars, len(group) - 1)
    kind = _spec_policy_kind(spec)
    if kind == "next_open":
        entry_idx = signal_idx + 1
        if entry_idx >= len(group):
            return {"status": "armed", "entry_reason": "waiting_next_open"}
        return {
            "status": "filled",
            "entry_idx": entry_idx,
            "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
            "entry_price": _safe_float(group.iloc[entry_idx]["open"]),
            "pullback_time": pd.NaT,
            "reclaim_time": pd.NaT,
            "entry_reason": "next_open",
        }

    if kind == "v9_retest":
        # P1 port — `pullback_pct` is reinterpreted as the retest buffer:
        # the bar must dip to or below ``signal_close*(1+buffer)`` and close
        # back above both signal_close and ema20.
        from pressure_graph.entry.retest import passes_v9_retest

        retest_time = pd.NaT
        for idx in range(signal_idx + 1, valid_end + 1):
            row = group.iloc[idx]
            ema20 = _safe_float(row.get("ema20"))
            low = _safe_float(row.get("low"))
            close = _safe_float(row.get("close"))
            if not np.isfinite(ema20) or not np.isfinite(low) or not np.isfinite(close):
                continue
            if passes_v9_retest(low, close, signal_close, ema20, retest_buffer=spec.pullback_pct):
                retest_time = pd.Timestamp(row["bar_close_time"])
                entry_idx = idx + 1
                if entry_idx >= len(group):
                    return {
                        "status": "retest_seen",
                        "reclaim_time": retest_time,
                        "entry_reason": "waiting_next_open_after_v9_retest",
                    }
                return {
                    "status": "filled",
                    "entry_idx": entry_idx,
                    "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
                    "entry_price": _safe_float(group.iloc[entry_idx]["open"]),
                    "pullback_time": retest_time,
                    "reclaim_time": retest_time,
                    "entry_reason": "v9_retest_next_open",
                }
        return {
            "status": "expired",
            "pullback_time": retest_time,
            "entry_reason": "expired_no_v9_retest",
        }

    saw_pullback = False
    pullback_time = pd.NaT
    for idx in range(signal_idx + 1, valid_end + 1):
        row = group.iloc[idx]
        if not saw_pullback and _safe_float(row.get("low")) <= trigger:
            saw_pullback = True
            pullback_time = pd.Timestamp(row["bar_open_time"])
            if kind == "pullback":
                return {
                    "status": "filled",
                    "entry_idx": idx,
                    "entry_time": pd.Timestamp(row["bar_open_time"]),
                    "entry_price": trigger,
                    "pullback_time": pullback_time,
                    "reclaim_time": pd.NaT,
                    "entry_reason": "pullback_only",
                }
        if saw_pullback and _safe_float(row.get("close")) >= signal_close:
            entry_idx = idx + 1
            reclaim_time = pd.Timestamp(row["bar_close_time"])
            if entry_idx >= len(group):
                return {
                    "status": "reclaim_seen",
                    "pullback_time": pullback_time,
                    "reclaim_time": reclaim_time,
                    "entry_reason": "waiting_next_open_after_reclaim",
                }
            return {
                "status": "filled",
                "entry_idx": entry_idx,
                "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
                "entry_price": _safe_float(group.iloc[entry_idx]["open"]),
                "pullback_time": pullback_time,
                "reclaim_time": reclaim_time,
                "entry_reason": "pullback_reclaim_next_open",
            }
    return {
        "status": "pullback_seen" if saw_pullback else "expired",
        "pullback_time": pullback_time,
        "entry_reason": "expired_no_reclaim" if saw_pullback else "expired_no_pullback",
    }


def _exit_from_entry(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: ExecutionRule,
) -> dict[str, object]:
    tp_price = entry_price * (1.0 + rule.tp)
    sl_price = entry_price * (1.0 - rule.sl)
    target_exit_idx = entry_idx + rule.max_hold_bars - 1
    max_exit_idx = min(target_exit_idx, len(group) - 1)
    exit_idx = max_exit_idx
    exit_price = _safe_float(group.iloc[max_exit_idx]["close"])
    exit_reason = "open" if target_exit_idx >= len(group) else "max_hold"
    same_bar_ambiguity = False
    for idx in range(entry_idx, max_exit_idx + 1):
        row = group.iloc[idx]
        high_hit = _safe_float(row.get("high")) >= tp_price
        low_hit = _safe_float(row.get("low")) <= sl_price
        if high_hit and low_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl_ambiguous"
            same_bar_ambiguity = True
            break
        if low_hit:
            exit_idx = idx
            exit_price = sl_price
            exit_reason = "sl"
            break
        if high_hit:
            exit_idx = idx
            exit_price = tp_price
            exit_reason = "tp"
            break
    holding = group.iloc[entry_idx : exit_idx + 1]
    entry_time = pd.Timestamp(group.iloc[entry_idx]["bar_open_time"])
    exit_time = pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
    return {
        "exit_idx": exit_idx,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "max_hold_time": pd.Timestamp(group.iloc[max_exit_idx]["bar_close_time"]),
        "gross_return": exit_price / entry_price - 1.0,
        "holding_minutes": (exit_time - entry_time).total_seconds() / 60.0,
        "mae": _safe_float(holding["low"].min()) / entry_price - 1.0,
        "mfe": _safe_float(holding["high"].max()) / entry_price - 1.0,
        "tp_first": exit_reason.startswith("tp"),
        "sl_first": exit_reason.startswith("sl"),
        "timeout": exit_reason in {"max_hold", "open"},
        "same_bar_ambiguity": same_bar_ambiguity,
        "funding_cost": _funding_cost(group, entry_time, exit_time),
    }


def _simulate_trade(
    group: pd.DataFrame,
    signal_idx: int,
    signal: dict[str, object],
    spec: V07A2Candidate | V07A2Baseline,
    role: str,
    config: V07A2Config,
    baseline_kind: str = "",
) -> tuple[dict[str, object], dict[str, object] | None]:
    entry = _entry_for_signal(group, signal_idx, spec)
    signal = dict(signal)
    for key in ["status", "pullback_time", "reclaim_time", "entry_time", "entry_price", "entry_reason"]:
        if key in entry:
            signal[key] = entry[key]
    if entry["status"] != "filled":
        return signal, None
    entry_idx = int(entry["entry_idx"])
    entry_price = float(entry["entry_price"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        signal["status"] = "skipped"
        signal["skip_reason"] = "bad_entry_price"
        return signal, None
    entry_context = _market_context_asof(group, pd.Timestamp(signal["entry_time"]), spec.market_gate)
    signal["market_gate_at_entry"] = bool(entry_context["market_gate"])
    if config.market_graph.strict_signal_and_entry_gate and not signal["market_gate_at_entry"]:
        signal["status"] = "skipped"
        signal["skip_reason"] = f"entry_market_gate_off:{spec.market_gate}"
        return signal, None
    rule = _rule(spec.execution_rule, group.iloc[entry_idx])
    exit_data = _exit_from_entry(group, entry_idx, entry_price, rule)
    exit_context = _market_context_asof(group, pd.Timestamp(exit_data["exit_time"]), spec.market_gate)
    signal_row = group.iloc[signal_idx]
    entry_row = group.iloc[entry_idx]
    signal["exit_time"] = exit_data["exit_time"]
    signal["exit_reason"] = exit_data["exit_reason"]
    signal["market_gate_at_exit"] = bool(exit_context["market_gate"])
    signal["status"] = "filled" if exit_data["exit_reason"] == "open" else "exited"
    gross = float(exit_data["gross_return"])
    trade = {
        "trade_id": f"{signal['signal_id']}:paper",
        "signal_id": signal["signal_id"],
        "candidate": spec.candidate,
        "candidate_role": role,
        "baseline_kind": baseline_kind,
        "exchange": signal["exchange"],
        "symbol": signal["symbol"],
        "market_gate": spec.market_gate,
        "market_gate_time": signal["market_gate_time"],
        "volume_impulse_density_at_signal": signal["volume_impulse_density_value"],
        "volume_impulse_density_at_entry": entry_context["volume_impulse_density"],
        "volume_impulse_density_at_exit": exit_context["volume_impulse_density"],
        "cluster_id_at_signal": str(signal_row.get("cluster_id", "")),
        "cluster_id_at_entry": str(entry_row.get("cluster_id", "")),
        "cluster_size_at_signal": _safe_float(signal_row.get("cluster_size")),
        "cluster_size_at_entry": _safe_float(entry_row.get("cluster_size")),
        "cluster_impulse_density_at_signal": _safe_float(signal_row.get("cluster_impulse_density")),
        "cluster_impulse_density_at_entry": _safe_float(entry_row.get("cluster_impulse_density")),
        "beta_extension_bucket_at_signal": str(signal_row.get("c2_beta_extension_bucket", "")),
        "beta_extension_score_at_signal": _safe_float(signal_row.get("c2_beta_extension_score")),
        "local_volume_shock_strength_at_signal": _safe_float(signal_row.get("volume_z_4h")),
        "volume_impulse_density_state": signal["volume_impulse_density_state"],
        "local_volume_shock_time": signal["local_volume_shock_time"],
        "pullback_time": signal["pullback_time"],
        "reclaim_time": signal["reclaim_time"],
        "entry_time": signal["entry_time"],
        "entry_price": entry_price,
        "exit_time": exit_data["exit_time"],
        "exit_reason": exit_data["exit_reason"],
        "tp_price": exit_data["tp_price"],
        "sl_price": exit_data["sl_price"],
        "max_hold_time": exit_data["max_hold_time"],
        "gross_return": gross,
        "holding_minutes": exit_data["holding_minutes"],
        "mae": exit_data["mae"],
        "mfe": exit_data["mfe"],
        "tp_first": exit_data["tp_first"],
        "sl_first": exit_data["sl_first"],
        "timeout": exit_data["timeout"],
        "same_bar_ambiguity": exit_data["same_bar_ambiguity"],
        "funding_cost": exit_data["funding_cost"],
        "market_gate_at_signal": signal["market_gate_at_signal"],
        "market_gate_at_entry": signal["market_gate_at_entry"],
        "market_gate_at_exit": signal["market_gate_at_exit"],
        "btc_state_at_signal": signal["btc_state"],
        "btc_state_at_entry": entry_context["btc_state"],
        "btc_state_at_exit": exit_context["btc_state"],
        "concurrent_positions_at_entry": 0,
        "portfolio_accepted": role == "primary" and not baseline_kind,
        "portfolio_skip_reason": "",
        "turnover_30d": _safe_float(group.iloc[signal_idx].get("dynamic_all_trailing_turnover")),
    }
    for cost in config.costs.single_side_bps:
        trade[f"net_return_{_cost_suffix(cost)}"] = gross - 2.0 * float(cost) / 10_000.0
    return signal, trade


def _candidate_skip_reason(row: pd.Series, spec: V07A2Candidate | V07A2Baseline) -> str | None:
    rank = _safe_float(row.get("dynamic_all_rank"))
    if pd.isna(rank):
        return "rank_missing"
    if rank > spec.universe_top_n:
        return f"outside_top{spec.universe_top_n}"
    if not _bool(row.get(spec.market_gate)):
        return f"market_gate_off:{spec.market_gate}"
    if not _bool(row.get(spec.event_col)):
        return f"event_off:{spec.event_col}"
    return None


def _apply_primary_portfolio(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    config: V07A2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty or trades.empty:
        return signals, trades
    signals = signals.copy().set_index("signal_id", drop=False)
    data = trades.copy()
    primary = (
        data["candidate"].astype(str).eq(config.experiment.primary_candidate)
        & data["baseline_kind"].fillna("").eq("")
    )
    data.loc[~primary, "portfolio_accepted"] = False
    data.loc[~primary, "portfolio_skip_reason"] = "shadow"
    candidates = data[primary].copy()
    candidates["_entry_sort"] = pd.to_datetime(candidates["entry_time"], utc=True, errors="coerce")
    candidates["_turnover_sort"] = pd.to_numeric(candidates["turnover_30d"], errors="coerce").fillna(-np.inf)
    candidates = candidates.sort_values(["_entry_sort", "_turnover_sort", "symbol"], ascending=[True, False, True])
    active: list[tuple[pd.Timestamp, str]] = []
    accepted: set[str] = set()
    skip_reasons: dict[str, str] = {}
    concurrent: dict[str, int] = {}
    for row in candidates.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        active = [(exit_time, symbol) for exit_time, symbol in active if exit_time > entry_time]
        active_symbols = {symbol for _, symbol in active}
        if config.portfolio.one_position_per_symbol and row.symbol in active_symbols:
            skip_reasons[str(row.signal_id)] = "symbol_position_open"
            continue
        if len(active) >= config.portfolio.primary_max_positions:
            skip_reasons[str(row.signal_id)] = "max_positions"
            continue
        accepted.add(str(row.signal_id))
        concurrent[str(row.signal_id)] = len(active)
        active.append((pd.Timestamp(row.exit_time), str(row.symbol)))
    data["portfolio_accepted"] = data["signal_id"].astype(str).isin(accepted)
    for signal_id, reason in skip_reasons.items():
        signals.loc[signal_id, "status"] = "skipped"
        signals.loc[signal_id, "skip_reason"] = reason
        data.loc[data["signal_id"].astype(str).eq(signal_id), "portfolio_skip_reason"] = reason
    data.loc[primary & ~data["portfolio_accepted"] & data["portfolio_skip_reason"].eq(""), "portfolio_skip_reason"] = (
        "not_primary_portfolio"
    )
    data["concurrent_positions_at_entry"] = [
        concurrent.get(str(signal_id), value)
        for signal_id, value in zip(data["signal_id"], data["concurrent_positions_at_entry"], strict=False)
    ]
    return signals.reset_index(drop=True), data


def _baseline_to_spec(item: V07A2Baseline) -> V07A2Candidate:
    return V07A2Candidate(
        candidate=item.candidate,
        role="baseline",
        universe_top_n=50,
        market_gate=item.market_gate,
        event_col=item.event_col,
        entry_policy=item.entry_policy,
        pullback_pct=item.pullback_pct,
        valid_bars=item.valid_bars,
        execution_rule=item.execution_rule,
        rationale=item.baseline,
    )


def build_v07a2_paper_ledger(
    prepared: pd.DataFrame,
    config: V07A2Config,
    signal_start_time: pd.Timestamp | None = None,
    created_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if prepared.empty:
        return (
            pd.DataFrame(columns=SIGNAL_COLUMNS),
            pd.DataFrame(columns=TRADE_COLUMNS),
            pd.DataFrame(columns=SIGNAL_COLUMNS),
            pd.DataFrame(columns=TRADE_COLUMNS),
        )
    created_at = created_at or pd.Timestamp.now(tz="UTC")
    data = add_v07a2_live_columns(prepared, config)
    for col in ["bar_open_time", "bar_close_time", "feature_time"]:
        data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    if signal_start_time is not None:
        signal_start_time = pd.Timestamp(signal_start_time)
        signal_start_time = (
            signal_start_time.tz_localize("UTC")
            if signal_start_time.tzinfo is None
            else signal_start_time.tz_convert("UTC")
        )
    signal_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    baseline_signal_rows: list[dict[str, object]] = []
    baseline_trade_rows: list[dict[str, object]] = []
    for _, group in data.groupby(["exchange", "symbol"], sort=False, observed=True):
        group = group.reset_index(drop=True)
        for candidate in config.candidates:
            mask = group[candidate.event_col].fillna(False).astype(bool)
            for signal_idx, is_signal in enumerate(mask):
                if not is_signal:
                    continue
                row = group.iloc[signal_idx]
                feature_time = pd.Timestamp(row["feature_time"])
                if signal_start_time is not None and feature_time < signal_start_time:
                    continue
                signal = _base_signal_row(row, candidate, candidate.role, created_at)
                skip = _candidate_skip_reason(row, candidate)
                if skip:
                    signal["status"] = "skipped"
                    signal["skip_reason"] = skip
                    signal_rows.append(signal)
                    continue
                signal, trade = _simulate_trade(group, signal_idx, signal, candidate, candidate.role, config)
                signal_rows.append(signal)
                if trade is not None:
                    trade_rows.append(trade)
        if not config.baselines.enabled:
            continue
        for baseline in config.baselines.items:
            spec = _baseline_to_spec(baseline)
            mask = group[baseline.event_col].fillna(False).astype(bool)
            for signal_idx, is_signal in enumerate(mask):
                if not is_signal:
                    continue
                row = group.iloc[signal_idx]
                feature_time = pd.Timestamp(row["feature_time"])
                if signal_start_time is not None and feature_time < signal_start_time:
                    continue
                signal = _base_signal_row(row, spec, "baseline", created_at, baseline.baseline)
                skip = _candidate_skip_reason(row, spec)
                if skip:
                    signal["status"] = "skipped"
                    signal["skip_reason"] = skip
                    baseline_signal_rows.append(signal)
                    continue
                signal, trade = _simulate_trade(group, signal_idx, signal, spec, "baseline", config, baseline.baseline)
                baseline_signal_rows.append(signal)
                if trade is not None:
                    baseline_trade_rows.append(trade)
    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLUMNS)
    baseline_signals = pd.DataFrame(baseline_signal_rows, columns=SIGNAL_COLUMNS)
    baseline_trades = pd.DataFrame(baseline_trade_rows, columns=TRADE_COLUMNS)
    signals, trades = _apply_primary_portfolio(signals, trades, config)
    return signals, trades, baseline_signals, baseline_trades


def _summary(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    group_cols: list[str],
    accepted_only: bool = False,
) -> pd.DataFrame:
    if signals.empty and trades.empty:
        return pd.DataFrame()
    sample = trades.copy()
    if accepted_only and "portfolio_accepted" in sample:
        sample = sample[sample["portfolio_accepted"].fillna(False)]
    rows = []
    signal_counts = signals.groupby(group_cols, dropna=False).size().to_dict() if not signals.empty else {}
    if sample.empty:
        for key, count in signal_counts.items():
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(group_cols, key, strict=False)), "signals": count, "trades": 0})
        return pd.DataFrame(rows)
    for key, group in sample.groupby(group_cols, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key, strict=False))
        exits = group["exit_reason"].astype(str)
        row = {
            **base,
            "signals": int(signal_counts.get(key if len(key) > 1 else key[0], 0)),
            "trades": int(len(group)),
            "filled_rate": float(len(group) / signal_counts.get(key if len(key) > 1 else key[0], len(group)))
            if signal_counts.get(key if len(key) > 1 else key[0], len(group))
            else np.nan,
            "gross_avg": _safe_float(pd.to_numeric(group["gross_return"], errors="coerce").mean()),
            "tp_rate": float(exits.str.startswith("tp").mean()),
            "sl_rate": float(exits.str.startswith("sl").mean()),
            "timeout_rate": float(exits.isin(["max_hold", "open"]).mean()),
            "avg_holding_minutes": _safe_float(pd.to_numeric(group["holding_minutes"], errors="coerce").mean()),
        }
        for cost in [5, 10, 20, 30, 50]:
            col = f"net_return_{cost}bp"
            if col in group.columns:
                row[f"net_{cost}bp_avg"] = _safe_float(pd.to_numeric(group[col], errors="coerce").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _primary_portfolio_trades(trades: pd.DataFrame, config: V07A2Config) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades[
        trades["candidate"].astype(str).eq(config.experiment.primary_candidate)
        & trades["baseline_kind"].fillna("").eq("")
        & trades["portfolio_accepted"].fillna(False)
    ].copy()


def _sample_status(trades: int) -> tuple[str, str]:
    if trades < 30:
        return "insufficient", "no_decision"
    if trades < 100:
        return "behavior_check", "no_candidate_decision"
    return "candidate_check", "evaluate_candidate"


def _latest_feature_time(prepared: pd.DataFrame) -> pd.Timestamp:
    if prepared.empty or "feature_time" not in prepared:
        return pd.NaT
    return pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()


def _latest_market_state(prepared: pd.DataFrame) -> dict[str, object]:
    if prepared.empty:
        return {
            "btc_state": "unknown",
            "volume_impulse_density": np.nan,
            "market_volume_impulse_density_high": False,
        }
    latest = _latest_feature_time(prepared)
    sample = prepared[pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").eq(latest)]
    btc = sample[sample["symbol"].astype(str).eq("BTCUSDT")]
    row = btc.iloc[-1] if not btc.empty else sample.iloc[-1]
    return {
        "btc_state": str(row.get("btc_market_state", "unknown")),
        "volume_impulse_density": _safe_float(row.get("volume_impulse_density")),
        "market_volume_impulse_density_high": _bool(row.get("market_volume_impulse_density_high")),
    }


def _data_stale(prepared: pd.DataFrame, config: V07A2Config) -> bool:
    latest = _latest_feature_time(prepared)
    if pd.isna(latest):
        return True
    return pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest) > pd.Timedelta(minutes=15 * config.stops.stale_data_bars)


def _daily_summary(signals: pd.DataFrame, trades: pd.DataFrame, config: V07A2Config) -> pd.DataFrame:
    if signals.empty and trades.empty:
        return pd.DataFrame()
    signal_dates = (
        pd.to_datetime(signals.get("feature_time"), utc=True, errors="coerce").dt.date
        if not signals.empty
        else pd.Series(dtype=object)
    )
    trade_dates = (
        pd.to_datetime(trades.get("entry_time"), utc=True, errors="coerce").dt.date
        if not trades.empty
        else pd.Series(dtype=object)
    )
    dates = sorted(set(signal_dates.dropna()).union(set(trade_dates.dropna())))
    rows = []
    for date in dates:
        day_signals = signals[signal_dates.eq(date)] if not signals.empty else signals
        day_trades = trades[trade_dates.eq(date)] if not trades.empty else trades
        primary = (
            day_trades[
                day_trades["candidate"].eq(config.experiment.primary_candidate)
                & day_trades["portfolio_accepted"].fillna(False)
            ]
            if not day_trades.empty
            else day_trades
        )
        rows.append(
            {
                "date": date,
                "signals": int(len(day_signals)),
                "primary_trades": int(len(primary)),
                "all_candidate_trades": int(len(day_trades)),
                "net_10bp_avg": _safe_float(primary.get("net_return_10bp", pd.Series(dtype=float)).mean()),
                "net_20bp_avg": _safe_float(primary.get("net_return_20bp", pd.Series(dtype=float)).mean()),
                "tp_rate": _safe_float(primary.get("tp_first", pd.Series(dtype=bool)).mean()),
                "sl_rate": _safe_float(primary.get("sl_first", pd.Series(dtype=bool)).mean()),
                "timeout_rate": _safe_float(primary.get("timeout", pd.Series(dtype=bool)).mean()),
                "max_concurrent_positions": _safe_float(
                    primary.get("concurrent_positions_at_entry", pd.Series(dtype=float)).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def _market_gate_audit(signals: pd.DataFrame, trades: pd.DataFrame, prepared: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_id",
        "symbol",
        "signal_time",
        "entry_time",
        "market_gate",
        "market_gate_at_signal",
        "market_gate_at_entry",
        "market_gate_at_exit",
        "volume_impulse_density_at_signal",
        "volume_impulse_density_at_entry",
        "volume_impulse_density_now",
        "candidate",
        "is_primary",
        "gate_passed",
    ]
    rows = []
    latest = _latest_market_state(prepared)
    if not trades.empty:
        sample = trades[trades["baseline_kind"].fillna("").eq("")].copy()
        rows.append(
            pd.DataFrame(
                {
                    "trade_id": sample["trade_id"].astype(str),
                    "symbol": sample["symbol"].astype(str),
                    "signal_time": pd.to_datetime(sample["local_volume_shock_time"], utc=True, errors="coerce"),
                    "entry_time": pd.to_datetime(sample["entry_time"], utc=True, errors="coerce"),
                    "market_gate": sample["market_gate"].astype(str),
                    "market_gate_at_signal": sample["market_gate_at_signal"].fillna(False).astype(bool),
                    "market_gate_at_entry": sample["market_gate_at_entry"].fillna(False).astype(bool),
                    "market_gate_at_exit": sample["market_gate_at_exit"].fillna(False).astype(bool),
                    "volume_impulse_density_at_signal": pd.to_numeric(
                        sample["volume_impulse_density_at_signal"], errors="coerce"
                    ),
                    "volume_impulse_density_at_entry": pd.to_numeric(
                        sample["volume_impulse_density_at_entry"], errors="coerce"
                    ),
                    "volume_impulse_density_now": latest["volume_impulse_density"],
                    "candidate": sample["candidate"].astype(str),
                    "is_primary": sample["candidate"].astype(str).eq("MIR1")
                    & sample["portfolio_accepted"].fillna(False),
                    "gate_passed": sample["market_gate_at_signal"].fillna(False).astype(bool)
                    & sample["market_gate_at_entry"].fillna(False).astype(bool),
                }
            )
        )
    if not signals.empty:
        skipped = signals[
            signals["skip_reason"].astype(str).str.startswith("entry_market_gate_off:", na=False)
        ].copy()
        if not skipped.empty:
            rows.append(
                pd.DataFrame(
                    {
                        "trade_id": skipped["signal_id"].astype(str) + ":invalidated",
                        "symbol": skipped["symbol"].astype(str),
                        "signal_time": pd.to_datetime(skipped["feature_time"], utc=True, errors="coerce"),
                        "entry_time": pd.to_datetime(skipped["entry_time"], utc=True, errors="coerce"),
                        "market_gate": skipped["market_gate"].astype(str),
                        "market_gate_at_signal": skipped["market_gate_at_signal"].fillna(False).astype(bool),
                        "market_gate_at_entry": False,
                        "market_gate_at_exit": False,
                        "volume_impulse_density_at_signal": pd.to_numeric(
                            skipped["volume_impulse_density_value"], errors="coerce"
                        ),
                        "volume_impulse_density_at_entry": np.nan,
                        "volume_impulse_density_now": latest["volume_impulse_density"],
                        "candidate": skipped["candidate"].astype(str),
                        "is_primary": skipped["candidate"].astype(str).eq("MIR1"),
                        "gate_passed": False,
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.concat(rows, ignore_index=True)
    return out[columns].sort_values(["entry_time", "candidate", "symbol"]).reset_index(drop=True)


def _daily_trade_rows(
    series_name: str,
    candidate: str,
    baseline_kind: str,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
) -> list[dict[str, object]]:
    if signals.empty and trades.empty:
        return []
    signal_dates = (
        pd.to_datetime(signals.get("feature_time"), utc=True, errors="coerce").dt.date
        if not signals.empty
        else pd.Series(dtype=object)
    )
    trade_dates = (
        pd.to_datetime(trades.get("entry_time"), utc=True, errors="coerce").dt.date
        if not trades.empty
        else pd.Series(dtype=object)
    )
    dates = sorted(set(signal_dates.dropna()).union(set(trade_dates.dropna())))
    rows: list[dict[str, object]] = []
    for date in dates:
        day_signals = signals[signal_dates.eq(date)] if not signals.empty else signals
        day_trades = trades[trade_dates.eq(date)] if not trades.empty else trades
        exits = day_trades["exit_reason"].astype(str) if not day_trades.empty else pd.Series(dtype=str)
        rows.append(
            {
                "date": date,
                "series": series_name,
                "candidate": candidate,
                "baseline_kind": baseline_kind,
                "signals": int(len(day_signals)),
                "trades": int(len(day_trades)),
                "net_10bp_avg": _safe_float(day_trades.get("net_return_10bp", pd.Series(dtype=float)).mean()),
                "net_20bp_avg": _safe_float(day_trades.get("net_return_20bp", pd.Series(dtype=float)).mean()),
                "tp_rate": float(exits.str.startswith("tp").mean()) if not exits.empty else np.nan,
                "sl_rate": float(exits.str.startswith("sl").mean()) if not exits.empty else np.nan,
                "timeout_rate": float(exits.isin(["max_hold", "open"]).mean()) if not exits.empty else np.nan,
            }
        )
    return rows


def _shadow_baselines_live(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_signals: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V07A2Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary = config.experiment.primary_candidate
    primary_signals = signals[signals["candidate"].astype(str).eq(primary)] if not signals.empty else signals
    primary_trades = _primary_portfolio_trades(trades, config)
    rows.extend(_daily_trade_rows("MIR1_primary", primary, "", primary_signals, primary_trades))
    for candidate in config.candidates:
        if candidate.candidate == primary:
            continue
        sig = signals[signals["candidate"].astype(str).eq(candidate.candidate)] if not signals.empty else signals
        trd = trades[trades["candidate"].astype(str).eq(candidate.candidate)] if not trades.empty else trades
        rows.extend(_daily_trade_rows(candidate.candidate, candidate.candidate, "", sig, trd))
    for baseline in config.baselines.items:
        sig = (
            baseline_signals[baseline_signals["baseline_kind"].astype(str).eq(baseline.baseline)]
            if not baseline_signals.empty
            else baseline_signals
        )
        trd = (
            baseline_trades[baseline_trades["baseline_kind"].astype(str).eq(baseline.baseline)]
            if not baseline_trades.empty
            else baseline_trades
        )
        rows.extend(_daily_trade_rows(baseline.baseline, baseline.candidate, baseline.baseline, sig, trd))
    return pd.DataFrame(rows)


def _write_status(
    report_root: Path,
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V07A2Config,
) -> None:
    latest = _latest_feature_time(prepared)
    market = _latest_market_state(prepared)
    primary = _primary_portfolio_trades(trades, config)
    net10 = _safe_float(primary.get("net_return_10bp", pd.Series(dtype=float)).mean())
    net20 = _safe_float(primary.get("net_return_20bp", pd.Series(dtype=float)).mean())
    sample_status, evaluation_status = _sample_status(len(primary))
    lines = [
        "# v0.7A.2 MIR1 Paper-Live Status",
        "",
        f"- strategy_id: {config.experiment.strategy_id}",
        f"- primary_candidate: {config.experiment.primary_candidate}",
        f"- candidate_rating: {config.experiment.candidate_rating}",
        f"- latest_feature_time: {latest}",
        f"- btc_state: {market['btc_state']}",
        f"- volume_impulse_density: {market['volume_impulse_density']}",
        f"- market_volume_impulse_density_high: {market['market_volume_impulse_density_high']}",
        f"- data_stale: {_data_stale(prepared, config)}",
        f"- paper_live_allowed: {config.experiment.paper_live_allowed}",
        f"- real_live_allowed: {config.experiment.real_live_allowed}",
        f"- tick_validation_status: {config.experiment.tick_validation_status}",
        f"- historical_validation_status: {config.experiment.historical_validation_status}",
        f"- total_candidate_signals: {len(signals)}",
        f"- primary_portfolio_trades: {len(primary)}",
        f"- sample_status: {sample_status}",
        f"- evaluation_status: {evaluation_status}",
        f"- primary_net_10bp_avg: {net10:.4%}" if pd.notna(net10) else "- primary_net_10bp_avg: n/a",
        f"- primary_net_20bp_avg: {net20:.4%}" if pd.notna(net20) else "- primary_net_20bp_avg: n/a",
        f"- baseline_trades: {len(baseline_trades)}",
    ]
    (report_root / "current_status.md").write_text("\n".join(lines), encoding="utf-8")
    summary = _summary(signals, trades, ["candidate", "candidate_role"])
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(pd.DataFrame(), baseline_trades, ["candidate", "baseline_kind"])
    note = [
        "# v0.7A.2 Candidate Status",
        "",
        "Decision: MIR1 is frozen as the primary B- graph-context paper-live candidate.",
        "",
        "Definition: market-wide volume impulse density high -> local bullish volume shock -> 1% pullback reclaim -> vol-regime-fast exit.",
        "",
        "Gate rule: market gate at signal and entry must both pass using as-of market features (`feature_time <= decision_time`). Existing paper trades continue to normal exit after market gate changes.",
        "",
        "Real-live: disabled until tick validation is complete and enough future paper-live sample exists.",
        "",
        "Risks: max-month concentration remains high; MIR1 is an impulse-regime amplifier, not a smooth standalone strategy.",
        "",
        "Do not evaluate before primary MIR1 filled trades >= 30 for behavior checks and >= 100 for candidate checks.",
        "",
        f"filled_trades = {len(primary)}",
        f"sample_status = {sample_status}",
        f"evaluation_status = {evaluation_status}",
        "",
        "## Primary Portfolio",
    ]
    if primary_summary.empty:
        note.append("- No primary portfolio trades yet.")
    else:
        for row in primary_summary.itertuples(index=False):
            note.append(
                f"- {row.candidate}: trades={row.trades}, net10={getattr(row, 'net_10bp_avg', np.nan):.4%}, "
                f"net20={getattr(row, 'net_20bp_avg', np.nan):.4%}"
            )
    note.append("")
    note.append("## Reference And Negative Controls")
    if summary.empty:
        note.append("- No candidate rows yet.")
    else:
        for row in summary.itertuples(index=False):
            note.append(
                f"- {row.candidate}/{row.candidate_role}: trades={row.trades}, "
                f"net10={getattr(row, 'net_10bp_avg', np.nan):.4%}"
            )
    note.append("")
    note.append("## Baseline Shadows")
    if baseline_summary.empty:
        note.append("- No baseline shadow trades yet.")
    else:
        for row in baseline_summary.itertuples(index=False):
            note.append(
                f"- {row.candidate}/{row.baseline_kind}: trades={row.trades}, "
                f"net10={getattr(row, 'net_10bp_avg', np.nan):.4%}"
            )
    (report_root / "candidate_status.md").write_text("\n".join(note), encoding="utf-8")


def _append_decision_log(
    report_root: Path,
    prepared: pd.DataFrame,
    trades: pd.DataFrame,
    config: V07A2Config,
) -> None:
    primary = _primary_portfolio_trades(trades, config)
    status, evaluation = _sample_status(len(primary))
    market = _latest_market_state(prepared)
    line = (
        f"{pd.Timestamp.now(tz='UTC').isoformat()} | MIR1 primary filled_trades={len(primary)} | "
        f"volume_impulse_density={market['volume_impulse_density']} | "
        f"market_gate={market['market_volume_impulse_density_high']} | "
        f"sample_status={status} | evaluation_status={evaluation} | "
        f"real_live_allowed={config.experiment.real_live_allowed}\n"
    )
    path = report_root / "decision_log.md"
    if not path.exists():
        path.write_text(
            "# v0.7A.2 Decision Log\n\n"
            "No strategy evaluation is made before 30 MIR1 filled trades; candidate checks start at 100.\n\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def prepare_v07a2_features_from_history(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V07A2Config,
    days: int | None = 30,
) -> pd.DataFrame:
    rank30, rank90, symbols = _rank_inputs(feature_path, instruments, base_config)
    max_top_n = max(
        [candidate.universe_top_n for candidate in config.candidates]
        + [50 for _ in config.baselines.items]
        + [50]
    )
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    latest = pd.to_datetime(turnover["bar_open_time"], utc=True, errors="coerce").max()
    cutoff = latest - pd.Timedelta(days=days) if days is not None and pd.notna(latest) else None
    selected_symbols = sorted(
        rank30[pd.to_numeric(rank30["dynamic_all_rank"], errors="coerce") <= max_top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    if "BTCUSDT" not in selected_symbols:
        selected_symbols = ["BTCUSDT", *selected_symbols]
    frames = []
    for idx, symbol in enumerate(selected_symbols, start=1):
        data = _read_symbol_features(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        if cutoff is not None:
            data = data[pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce") >= cutoff].copy()
        data = data[pd.to_numeric(data["dynamic_all_rank"], errors="coerce") <= max_top_n].copy()
        if data.empty:
            continue
        print(f"v0.7A.2 prepare {idx}/{len(selected_symbols)} {symbol}", flush=True)
        frames.append(data)
    prepared = _concat_or_empty(frames)
    if prepared.empty:
        return prepared
    regime = _market_regime_features(prepared)
    if not regime.empty:
        regime["volume_impulse_density_high"] = (
            pd.to_numeric(regime["volume_impulse_density"], errors="coerce") >= config.market_graph.density_threshold
        )
        regime["low_volume_impulse_density"] = (
            pd.to_numeric(regime["volume_impulse_density"], errors="coerce")
            <= config.market_graph.low_density_threshold
        )
        prepared = _add_motif_columns(prepared, regime, base_config)
    return add_v07a2_live_columns(prepared, config)


def write_v07a2_mir1_paper_live(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V07A2Config,
    report_root: Path = REPORT_ROOT,
    paper_data_root: Path = PAPER_DATA_ROOT,
    days: int | None = 30,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    paper_data_root = ensure_dir(paper_data_root)
    prepared = prepare_v07a2_features_from_history(feature_path, instruments, base_config, config, days)
    signal_start_time = None
    if days is not None and not prepared.empty:
        latest = pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
        signal_start_time = latest - pd.Timedelta(days=days)
    signals, trades, baseline_signals, baseline_trades = build_v07a2_paper_ledger(
        prepared,
        config,
        signal_start_time,
    )
    daily = _daily_summary(signals, trades, config)
    candidate_summary = _summary(signals, trades, ["candidate", "candidate_role"])
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(baseline_signals, baseline_trades, ["candidate", "baseline_kind"])
    skipped = signals[signals["status"].eq("skipped")].copy() if not signals.empty else signals.copy()
    outputs = {
        "paper_signals": report_root / "paper_signals.parquet",
        "paper_trades": report_root / "paper_trades.parquet",
        "baseline_signals": report_root / "baseline_signals.parquet",
        "baseline_trades": report_root / "baseline_trades.parquet",
        "daily_summary": report_root / "daily_summary.csv",
        "candidate_summary": report_root / "candidate_summary.csv",
        "primary_summary": report_root / "primary_summary.csv",
        "baseline_summary": report_root / "baseline_summary.csv",
        "skipped_signals": report_root / "skipped_signals.csv",
        "market_gate_audit": report_root / "market_gate_audit.csv",
        "shadow_baselines_live": report_root / "shadow_baselines_live.csv",
        "current_status": report_root / "current_status.md",
        "candidate_status": report_root / "candidate_status.md",
        "decision_log": report_root / "decision_log.md",
        "paper_signals_data": paper_data_root / "paper_signals.parquet",
        "paper_trades_data": paper_data_root / "paper_trades.parquet",
    }
    write_parquet(signals, outputs["paper_signals"])
    write_parquet(trades, outputs["paper_trades"])
    write_parquet(baseline_signals, outputs["baseline_signals"])
    write_parquet(baseline_trades, outputs["baseline_trades"])
    write_parquet(signals, outputs["paper_signals_data"])
    write_parquet(trades, outputs["paper_trades_data"])
    daily.to_csv(outputs["daily_summary"], index=False)
    candidate_summary.to_csv(outputs["candidate_summary"], index=False)
    primary_summary.to_csv(outputs["primary_summary"], index=False)
    baseline_summary.to_csv(outputs["baseline_summary"], index=False)
    skipped.to_csv(outputs["skipped_signals"], index=False)
    _market_gate_audit(signals, trades, prepared).to_csv(outputs["market_gate_audit"], index=False)
    _shadow_baselines_live(signals, trades, baseline_signals, baseline_trades, config).to_csv(
        outputs["shadow_baselines_live"],
        index=False,
    )
    _write_status(report_root, prepared, signals, trades, baseline_trades, config)
    _append_decision_log(report_root, prepared, trades, config)
    return outputs
