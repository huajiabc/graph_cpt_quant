from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.backtest.simulator import _funding_cost
from pressure_graph.config import ExperimentConfig
from pressure_graph.config.models import ExecutionRule
from pressure_graph.config.v06a3 import V06A3Candidate, V06A3Config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.reports.v03 import (
    V03_TURNOVER_COLUMNS,
    _concat_or_empty,
    _event_flags,
    _read_existing_columns,
)
from pressure_graph.reports.v04 import (
    RANK30_COL,
    RANK90_COL,
    _rank_table_with_lookback,
    _read_symbol_v04,
)


REPORT_ROOT = Path("reports/v0_6a3_paper_live")
PAPER_DATA_ROOT = Path("data/paper/v0_6a3")
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
    "volume_shock_time",
    "pullback_time",
    "reclaim_time",
    "entry_time",
    "universe_top_n",
    "turnover_rank_30d",
    "turnover_rank_90d",
    "core_liquidity",
    "transient_hot",
    "btc_state",
    "btc_ret_1h",
    "btc_ret_4h",
    "btc_volatility_4h",
    "volume_z_4h",
    "ret_4h",
    "signal_close",
    "entry_trigger",
    "entry_price",
    "entry_valid_until",
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
    "volume_shock_time",
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
    "btc_state_at_signal",
    "btc_state_at_entry",
    "btc_state_at_exit",
    "concurrent_positions_at_entry",
    "portfolio_accepted",
    "portfolio_skip_reason",
    "turnover_30d",
    "transient_hot",
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
    if name == "swing":
        return ExecutionRule(tp=0.05, sl=0.03, max_hold_bars=48)
    if name == "vol_regime_fast":
        return _vol_regime_rule(row)
    raise KeyError(f"unknown v0.6A.3 execution rule: {name}")


def _signal_id(
    row: pd.Series,
    candidate: str,
    baseline_kind: str,
) -> str:
    bar_time = pd.Timestamp(row["bar_open_time"]).strftime("%Y%m%dT%H%M%SZ")
    suffix = baseline_kind or "candidate"
    return f"{candidate}:{suffix}:{row['exchange']}:{row['symbol']}:{bar_time}"


def add_v06a3_impulse_columns(df: pd.DataFrame, config: V06A3Config) -> pd.DataFrame:
    out = df.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    warmup = out.get("warmup_complete", True)
    if not isinstance(warmup, pd.Series):
        warmup = pd.Series(bool(warmup), index=out.index)
    volume_z_4h = pd.to_numeric(out.get("volume_z_4h"), errors="coerce")
    ret_4h = pd.to_numeric(out.get("ret_4h"), errors="coerce")
    out["bullish_volume_shock_state"] = (
        warmup
        & (volume_z_4h > config.signal.volume_z_4h_min)
        & (ret_4h > config.signal.ret_4h_min)
    )
    out["neutral_volume_state"] = warmup & (volume_z_4h.abs() <= 0.5)
    out["matched_random_state"] = warmup & ~out["bullish_volume_shock_state"]
    for state_col in ["bullish_volume_shock_state", "neutral_volume_state"]:
        event_col = state_col.replace("_state", "_event")
        out[event_col] = (
            out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[state_col]
            .apply(lambda series: _event_flags(series, config.signal.cooldown_bars))
            .reindex(out.index)
        )
    hashed = pd.util.hash_pandas_object(out[["symbol", "bar_open_time"]], index=False)
    out["matched_random_state"] = out["matched_random_state"] & ((hashed % 97) == 0)
    out["matched_random_event"] = (
        out.groupby(["exchange", "symbol"], group_keys=False, sort=False, observed=True)[
            "matched_random_state"
        ]
        .apply(lambda series: _event_flags(series, config.signal.cooldown_bars))
        .reindex(out.index)
    )
    rank30 = pd.to_numeric(out.get(RANK30_COL), errors="coerce")
    rank90 = pd.to_numeric(out.get(RANK90_COL), errors="coerce")
    out["core_liquidity"] = (rank30 <= 30) & (rank90 <= 50)
    out["transient_hot"] = (
        (rank30 <= config.liquidity.transient_hot.turnover_rank_30d_max)
        & (rank90 > config.liquidity.transient_hot.turnover_rank_90d_min)
    )
    out["liquidity_quality"] = np.select(
        [out["core_liquidity"], out["transient_hot"], rank90.isna()],
        ["core_liquidity", "transient_hot", "rank90_missing"],
        default="non_core_liquidity",
    )
    return out


def _candidate_base_mask(
    data: pd.DataFrame,
    candidate: V06A3Candidate,
    config: V06A3Config,
    event_col: str,
) -> pd.Series:
    state = data["btc_market_state"].astype("string")
    rank = pd.to_numeric(data[RANK30_COL], errors="coerce")
    mask = (rank <= candidate.universe_top_n) & data[event_col].fillna(False)
    if config.regime.btc_gate.enabled:
        mask &= state.eq(config.regime.btc_gate.required_state)
    if config.liquidity.transient_hot_veto:
        mask &= ~data["transient_hot"].fillna(False)
    return mask.fillna(False)


def _candidate_skip_reason(
    row: pd.Series,
    candidate: V06A3Candidate,
    config: V06A3Config,
) -> str | None:
    state = str(row.get("btc_market_state", "unknown"))
    if config.regime.btc_gate.enabled and state != config.regime.btc_gate.required_state:
        return f"btc_not_up:{state}"
    rank30 = _safe_float(row.get(RANK30_COL))
    if pd.isna(rank30):
        return "rank30_missing"
    if rank30 > candidate.universe_top_n:
        return f"outside_top{candidate.universe_top_n}"
    rank90 = _safe_float(row.get(RANK90_COL))
    if pd.isna(rank90):
        return "rank90_missing"
    if config.liquidity.transient_hot_veto and _bool(row.get("transient_hot")):
        return "transient_hot"
    return None


def _base_signal_row(
    row: pd.Series,
    candidate: V06A3Candidate,
    config: V06A3Config,
    created_at: pd.Timestamp,
    baseline_kind: str = "",
) -> dict[str, object]:
    signal_close = _safe_float(row.get("close"))
    return {
        "signal_id": _signal_id(row, candidate.candidate, baseline_kind),
        "created_at_utc": created_at,
        "candidate": candidate.candidate,
        "candidate_role": candidate.role,
        "baseline_kind": baseline_kind,
        "exchange": str(row.get("exchange")),
        "symbol": str(row.get("symbol")),
        "bar_open_time": pd.Timestamp(row.get("bar_open_time")),
        "bar_close_time": pd.Timestamp(row.get("bar_close_time")),
        "feature_time": pd.Timestamp(row.get("feature_time")),
        "volume_shock_time": pd.Timestamp(row.get("feature_time")),
        "pullback_time": pd.NaT,
        "reclaim_time": pd.NaT,
        "entry_time": pd.NaT,
        "universe_top_n": candidate.universe_top_n,
        "turnover_rank_30d": _safe_float(row.get(RANK30_COL)),
        "turnover_rank_90d": _safe_float(row.get(RANK90_COL)),
        "core_liquidity": _bool(row.get("core_liquidity")),
        "transient_hot": _bool(row.get("transient_hot")),
        "btc_state": str(row.get("btc_market_state", "unknown")),
        "btc_ret_1h": _safe_float(row.get("btc_ret_1h")),
        "btc_ret_4h": _safe_float(row.get("btc_ret_4h")),
        "btc_volatility_4h": _safe_float(row.get("btc_volatility_4h")),
        "volume_z_4h": _safe_float(row.get("volume_z_4h")),
        "ret_4h": _safe_float(row.get("ret_4h")),
        "signal_close": signal_close,
        "entry_trigger": signal_close * (1.0 - candidate.pullback_pct),
        "entry_price": np.nan,
        "entry_valid_until": pd.Timestamp(row.get("feature_time"))
        + pd.Timedelta(minutes=15 * candidate.valid_bars),
        "status": "detected",
        "skip_reason": "",
    }


def _entry_for_signal(
    group: pd.DataFrame,
    signal_idx: int,
    candidate: V06A3Candidate,
    baseline_kind: str,
) -> dict[str, object]:
    signal = group.iloc[signal_idx]
    signal_close = _safe_float(signal.get("close"))
    trigger = signal_close * (1.0 - candidate.pullback_pct)
    valid_end = min(signal_idx + candidate.valid_bars, len(group) - 1)
    if baseline_kind == "volume_shock_next_open":
        entry_idx = signal_idx + 1
        if entry_idx >= len(group):
            return {"status": "armed_waiting_next_open"}
        return {
            "status": "filled",
            "entry_idx": entry_idx,
            "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
            "entry_price": _safe_float(group.iloc[entry_idx]["open"]),
            "pullback_time": pd.NaT,
            "reclaim_time": pd.NaT,
        }

    saw_pullback = False
    pullback_time = pd.NaT
    for idx in range(signal_idx + 1, valid_end + 1):
        row = group.iloc[idx]
        if not saw_pullback and _safe_float(row.get("low")) <= trigger:
            saw_pullback = True
            pullback_time = pd.Timestamp(row["bar_open_time"])
            if baseline_kind == "volume_shock_pullback_only":
                return {
                    "status": "filled",
                    "entry_idx": idx,
                    "entry_time": pd.Timestamp(row["bar_open_time"]),
                    "entry_price": trigger,
                    "pullback_time": pullback_time,
                    "reclaim_time": pd.NaT,
                }
        if saw_pullback and _safe_float(row.get("close")) >= signal_close:
            entry_idx = idx + 1
            reclaim_time = pd.Timestamp(row["bar_close_time"])
            if entry_idx >= len(group):
                return {
                    "status": "reclaimed_waiting_entry",
                    "pullback_time": pullback_time,
                    "reclaim_time": reclaim_time,
                }
            return {
                "status": "filled",
                "entry_idx": entry_idx,
                "entry_time": pd.Timestamp(group.iloc[entry_idx]["bar_open_time"]),
                "entry_price": _safe_float(group.iloc[entry_idx]["open"]),
                "pullback_time": pullback_time,
                "reclaim_time": reclaim_time,
            }
    return {"status": "expired_no_reclaim" if saw_pullback else "expired_unfilled", "pullback_time": pullback_time}


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


def _btc_state_asof(group: pd.DataFrame, timestamp: pd.Timestamp) -> str:
    if group.empty or "feature_time" not in group.columns:
        return "unknown"
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    feature_time = pd.to_datetime(group["feature_time"], utc=True, errors="coerce")
    eligible = group[feature_time <= ts]
    if eligible.empty:
        return "unknown"
    return str(eligible.iloc[-1].get("btc_market_state", "unknown"))


def _simulate_trade(
    group: pd.DataFrame,
    signal_idx: int,
    signal: dict[str, object],
    candidate: V06A3Candidate,
    config: V06A3Config,
    baseline_kind: str = "",
) -> tuple[dict[str, object], dict[str, object] | None]:
    entry = _entry_for_signal(group, signal_idx, candidate, baseline_kind)
    signal = dict(signal)
    signal["status"] = entry["status"]
    for key in ["pullback_time", "reclaim_time", "entry_time", "entry_price"]:
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
    entry_state = _btc_state_asof(group, pd.Timestamp(signal["entry_time"]))
    if (
        not baseline_kind
        and candidate.candidate == config.experiment.primary_candidate
        and config.regime.btc_gate.enabled
        and entry_state != config.regime.btc_gate.required_state
    ):
        signal["status"] = "skipped"
        signal["skip_reason"] = f"entry_btc_not_up:{entry_state}"
        return signal, None
    rule = _rule(candidate.execution_rule, group.iloc[entry_idx])
    exit_data = _exit_from_entry(group, entry_idx, entry_price, rule)
    signal["status"] = "open" if exit_data["exit_reason"] == "open" else "exited"
    gross = float(exit_data["gross_return"])
    trade = {
        "trade_id": f"{signal['signal_id']}:paper",
        "signal_id": signal["signal_id"],
        "candidate": candidate.candidate,
        "candidate_role": candidate.role,
        "baseline_kind": baseline_kind,
        "exchange": signal["exchange"],
        "symbol": signal["symbol"],
        "volume_shock_time": signal["volume_shock_time"],
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
        "btc_state_at_signal": signal["btc_state"],
        "btc_state_at_entry": entry_state,
        "btc_state_at_exit": str(group.iloc[int(exit_data["exit_idx"])].get("btc_market_state", "unknown")),
        "concurrent_positions_at_entry": 0,
        "portfolio_accepted": candidate.role == "primary" and not baseline_kind,
        "portfolio_skip_reason": "",
        "turnover_30d": _safe_float(group.iloc[signal_idx].get("trailing_30d_turnover")),
        "transient_hot": signal["transient_hot"],
    }
    for cost in config.costs.single_side_bps:
        trade[f"net_return_{_cost_suffix(cost)}"] = gross - 2.0 * float(cost) / 10_000.0
    return signal, trade


def _apply_primary_portfolio(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    config: V06A3Config,
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
    candidates = candidates.sort_values(
        ["_entry_sort", "_turnover_sort", "symbol"],
        ascending=[True, False, True],
    )
    active: list[tuple[pd.Timestamp, str]] = []
    accepted_ids: set[str] = set()
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
        accepted_ids.add(str(row.signal_id))
        concurrent[str(row.signal_id)] = len(active)
        active.append((pd.Timestamp(row.exit_time), str(row.symbol)))

    data["portfolio_accepted"] = data["signal_id"].astype(str).isin(accepted_ids)
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


def _baseline_candidate(candidate: V06A3Candidate, kind: str) -> V06A3Candidate:
    if kind == "volume_shock_next_open":
        return replace(candidate, entry_policy="next_open")
    if kind == "volume_shock_pullback_only":
        return replace(candidate, entry_policy="pullback")
    return candidate


def _baseline_event_col(kind: str) -> str:
    if kind == "entry_only_reclaim":
        return "neutral_volume_event"
    if kind == "matched_random_reclaim":
        return "matched_random_event"
    return "bullish_volume_shock_event"


def build_v06a3_paper_ledger(
    prepared: pd.DataFrame,
    config: V06A3Config,
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
    data = prepared.sort_values(["exchange", "symbol", "bar_open_time"]).copy()
    for col in ["bar_open_time", "bar_close_time", "feature_time"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    data = add_v06a3_impulse_columns(data, config)
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
            event_mask = group["bullish_volume_shock_event"].fillna(False).astype(bool)
            for signal_idx, is_signal in enumerate(event_mask):
                if not is_signal:
                    continue
                row = group.iloc[signal_idx]
                feature_time = pd.Timestamp(row["feature_time"])
                if signal_start_time is not None and feature_time < signal_start_time:
                    continue
                signal = _base_signal_row(row, candidate, config, created_at)
                skip = _candidate_skip_reason(row, candidate, config)
                if skip:
                    signal["status"] = "skipped"
                    signal["skip_reason"] = skip
                    signal_rows.append(signal)
                    continue
                signal, trade = _simulate_trade(group, signal_idx, signal, candidate, config)
                signal_rows.append(signal)
                if trade is not None:
                    trade_rows.append(trade)

            if not config.baselines.enabled:
                continue
            for kind in config.baselines.kinds:
                base_candidate = _baseline_candidate(candidate, kind)
                base_event = _baseline_event_col(kind)
                base_mask = _candidate_base_mask(group, candidate, config, base_event)
                for signal_idx, is_signal in enumerate(base_mask.astype(bool)):
                    if not is_signal:
                        continue
                    row = group.iloc[signal_idx]
                    feature_time = pd.Timestamp(row["feature_time"])
                    if signal_start_time is not None and feature_time < signal_start_time:
                        continue
                    signal = _base_signal_row(row, base_candidate, config, created_at, kind)
                    signal, trade = _simulate_trade(group, signal_idx, signal, base_candidate, config, kind)
                    baseline_signal_rows.append(signal)
                    if trade is not None:
                        baseline_trade_rows.append(trade)

    signals = pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS)
    trades = pd.DataFrame(trade_rows)
    baseline_signals = pd.DataFrame(baseline_signal_rows, columns=SIGNAL_COLUMNS)
    baseline_trades = pd.DataFrame(baseline_trade_rows)
    if trades.empty:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
    if baseline_trades.empty:
        baseline_trades = pd.DataFrame(columns=TRADE_COLUMNS)
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
            if not isinstance(key, tuple):
                key = (key,)
            rows.append({**dict(zip(group_cols, key, strict=False)), "signals": count, "trades": 0})
        return pd.DataFrame(rows)
    for key, group in sample.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
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


def _daily_summary(signals: pd.DataFrame, trades: pd.DataFrame, config: V06A3Config) -> pd.DataFrame:
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
        primary = day_trades[
            day_trades["candidate"].eq(config.experiment.primary_candidate)
            & day_trades["portfolio_accepted"].fillna(False)
        ] if not day_trades.empty else day_trades
        rows.append(
            {
                "date": date,
                "signals": len(day_signals),
                "primary_trades": len(primary),
                "all_candidate_trades": len(day_trades),
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


def _latest_btc_state(prepared: pd.DataFrame) -> str:
    if prepared.empty:
        return "unknown"
    btc = prepared[prepared["symbol"].astype(str).eq("BTCUSDT")].sort_values("feature_time")
    return str(btc.iloc[-1].get("btc_market_state", "unknown")) if not btc.empty else "unknown"


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


def _data_stale(prepared: pd.DataFrame, config: V06A3Config) -> bool:
    latest_time = _latest_feature_time(prepared)
    stale_cutoff = pd.Timedelta(minutes=15 * config.stops.stale_data_bars)
    return pd.notna(latest_time) and pd.Timestamp.now(tz="UTC") - pd.Timestamp(latest_time) > stale_cutoff


def _primary_portfolio_trades(trades: pd.DataFrame, config: V06A3Config) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    return trades[
        trades["candidate"].astype(str).eq(config.experiment.primary_candidate)
        & trades["baseline_kind"].fillna("").eq("")
        & trades["portfolio_accepted"].fillna(False)
    ].copy()


def _gate_audit(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    prepared: pd.DataFrame,
    config: V06A3Config,
) -> pd.DataFrame:
    columns = [
        "trade_id",
        "symbol",
        "signal_time",
        "entry_time",
        "btc_state_at_signal",
        "btc_state_at_entry",
        "btc_state_now",
        "candidate",
        "is_primary",
        "gate_passed",
    ]
    rows: list[pd.DataFrame] = []
    if trades.empty:
        sample = pd.DataFrame()
    else:
        sample = trades[trades["baseline_kind"].fillna("").eq("")].copy()
    signal_context = pd.DataFrame(columns=["signal_id", "signal_time", "signal_btc_state"])
    if not signals.empty:
        signal_context = signals[["signal_id", "feature_time", "btc_state"]].rename(
            columns={"feature_time": "signal_time", "btc_state": "signal_btc_state"}
        )
    required = config.regime.btc_gate.required_state
    if not sample.empty:
        sample = sample.merge(signal_context, on="signal_id", how="left")
        signal_state = sample.get("signal_btc_state", sample.get("btc_state_at_signal")).fillna(
            sample.get("btc_state_at_signal")
        )
        entry_state = sample.get("btc_state_at_entry", pd.Series("unknown", index=sample.index)).fillna("unknown")
        is_primary = (
            sample["candidate"].astype(str).eq(config.experiment.primary_candidate)
            & sample["portfolio_accepted"].fillna(False)
        )
        rows.append(
            pd.DataFrame(
                {
                    "trade_id": sample["trade_id"].astype(str),
                    "symbol": sample["symbol"].astype(str),
                    "signal_time": pd.to_datetime(
                        sample.get("signal_time", sample.get("volume_shock_time")),
                        utc=True,
                        errors="coerce",
                    ),
                    "entry_time": pd.to_datetime(sample["entry_time"], utc=True, errors="coerce"),
                    "btc_state_at_signal": signal_state.astype(str),
                    "btc_state_at_entry": entry_state.astype(str),
                    "btc_state_now": _latest_btc_state(prepared),
                    "candidate": sample["candidate"].astype(str),
                    "is_primary": is_primary.astype(bool),
                    "gate_passed": signal_state.astype(str).eq(required)
                    & entry_state.astype(str).eq(required),
                }
            )
        )
    invalidated = _invalidated_trades(signals, config)
    if not invalidated.empty:
        rows.append(
            pd.DataFrame(
                {
                    "trade_id": invalidated["trade_id"].astype(str),
                    "symbol": invalidated["symbol"].astype(str),
                    "signal_time": invalidated["signal_time"],
                    "entry_time": invalidated["entry_time"],
                    "btc_state_at_signal": invalidated["btc_state_at_signal"].astype(str),
                    "btc_state_at_entry": invalidated["btc_state_at_entry"].astype(str),
                    "btc_state_now": _latest_btc_state(prepared),
                    "candidate": invalidated["candidate"].astype(str),
                    "is_primary": invalidated["is_primary"].astype(bool),
                    "gate_passed": False,
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    audit = pd.concat(rows, ignore_index=True)
    return audit[columns].sort_values(["entry_time", "candidate", "symbol"]).reset_index(drop=True)


def _invalidated_trades(signals: pd.DataFrame, config: V06A3Config) -> pd.DataFrame:
    columns = [
        "trade_id",
        "candidate",
        "symbol",
        "signal_time",
        "entry_time",
        "btc_state_at_signal",
        "btc_state_at_entry",
        "invalid_reason",
        "is_primary",
    ]
    if signals.empty or "skip_reason" not in signals.columns:
        return pd.DataFrame(columns=columns)
    sample = signals[signals["skip_reason"].astype(str).str.startswith("entry_btc_not_up:", na=False)].copy()
    if sample.empty:
        return pd.DataFrame(columns=columns)
    entry_state = sample["skip_reason"].astype(str).str.split(":", n=1).str[-1]
    out = pd.DataFrame(
        {
            "trade_id": sample["signal_id"].astype(str) + ":invalidated",
            "candidate": sample["candidate"].astype(str),
            "symbol": sample["symbol"].astype(str),
            "signal_time": pd.to_datetime(sample["feature_time"], utc=True, errors="coerce"),
            "entry_time": pd.to_datetime(sample["entry_time"], utc=True, errors="coerce"),
            "btc_state_at_signal": sample["btc_state"].astype(str),
            "btc_state_at_entry": entry_state,
            "invalid_reason": sample["skip_reason"].astype(str),
            "is_primary": (
                sample["candidate"].astype(str).eq(config.experiment.primary_candidate)
                & sample["baseline_kind"].fillna("").eq("")
            ),
        }
    )
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
        row = {
            "date": date,
            "series": series_name,
            "candidate": candidate,
            "baseline_kind": baseline_kind,
            "signals": int(len(day_signals)),
            "trades": int(len(day_trades)),
            "net_5bp_avg": _safe_float(day_trades.get("net_return_5bp", pd.Series(dtype=float)).mean()),
            "net_10bp_avg": _safe_float(day_trades.get("net_return_10bp", pd.Series(dtype=float)).mean()),
            "net_20bp_avg": _safe_float(day_trades.get("net_return_20bp", pd.Series(dtype=float)).mean()),
            "net_30bp_avg": _safe_float(day_trades.get("net_return_30bp", pd.Series(dtype=float)).mean()),
            "net_50bp_avg": _safe_float(day_trades.get("net_return_50bp", pd.Series(dtype=float)).mean()),
            "tp_rate": float(exits.str.startswith("tp").mean()) if not exits.empty else np.nan,
            "sl_rate": float(exits.str.startswith("sl").mean()) if not exits.empty else np.nan,
            "timeout_rate": float(exits.isin(["max_hold", "open"]).mean()) if not exits.empty else np.nan,
            "avg_holding_minutes": _safe_float(
                day_trades.get("holding_minutes", pd.Series(dtype=float)).mean()
            ),
        }
        rows.append(row)
    return rows


def _shadow_baselines_live(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_signals: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V06A3Config,
) -> pd.DataFrame:
    columns = [
        "date",
        "series",
        "candidate",
        "baseline_kind",
        "signals",
        "trades",
        "net_5bp_avg",
        "net_10bp_avg",
        "net_20bp_avg",
        "net_30bp_avg",
        "net_50bp_avg",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
        "avg_holding_minutes",
    ]
    primary_candidate = config.experiment.primary_candidate
    primary_signals = signals[signals["candidate"].astype(str).eq(primary_candidate)] if not signals.empty else signals
    primary_trades = _primary_portfolio_trades(trades, config)
    rows = _daily_trade_rows("IR2_primary", primary_candidate, "", primary_signals, primary_trades)
    for kind in config.baselines.kinds:
        bs = (
            baseline_signals[
                baseline_signals["candidate"].astype(str).eq(primary_candidate)
                & baseline_signals["baseline_kind"].astype(str).eq(kind)
            ]
            if not baseline_signals.empty
            else baseline_signals
        )
        bt = (
            baseline_trades[
                baseline_trades["candidate"].astype(str).eq(primary_candidate)
                & baseline_trades["baseline_kind"].astype(str).eq(kind)
            ]
            if not baseline_trades.empty
            else baseline_trades
        )
        rows.extend(_daily_trade_rows(kind, primary_candidate, kind, bs, bt))
    return pd.DataFrame(rows, columns=columns)


def _regime_shadow(signals: pd.DataFrame, trades: pd.DataFrame, config: V06A3Config) -> pd.DataFrame:
    columns = [
        "date",
        "btc_state",
        "candidate",
        "candidate_role",
        "signals",
        "skipped",
        "trades",
        "net_10bp_avg",
        "net_20bp_avg",
    ]
    if signals.empty:
        return pd.DataFrame(columns=columns)
    sample = signals[signals["btc_state"].astype(str).isin(config.regime.shadow_states)].copy()
    if sample.empty:
        return pd.DataFrame(columns=columns)
    sample["_date"] = pd.to_datetime(sample["feature_time"], utc=True, errors="coerce").dt.date
    trade_context = pd.DataFrame()
    if not trades.empty:
        trade_context = trades.merge(
            sample[["signal_id", "_date", "btc_state"]],
            on="signal_id",
            how="inner",
        )
    rows = []
    for key, group in sample.groupby(["_date", "btc_state", "candidate", "candidate_role"], dropna=False):
        date, btc_state, candidate, role = key
        group_trades = (
            trade_context[
                trade_context["_date"].eq(date)
                & trade_context["btc_state"].astype(str).eq(str(btc_state))
                & trade_context["candidate"].astype(str).eq(str(candidate))
            ]
            if not trade_context.empty
            else trade_context
        )
        rows.append(
            {
                "date": date,
                "btc_state": btc_state,
                "candidate": candidate,
                "candidate_role": role,
                "signals": int(len(group)),
                "skipped": int(group["status"].astype(str).eq("skipped").sum()),
                "trades": int(len(group_trades)),
                "net_10bp_avg": _safe_float(
                    group_trades.get("net_return_10bp", pd.Series(dtype=float)).mean()
                ),
                "net_20bp_avg": _safe_float(
                    group_trades.get("net_return_20bp", pd.Series(dtype=float)).mean()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _candidate_health(
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    config: V06A3Config,
) -> str:
    latest = _latest_feature_time(prepared)
    cutoff = latest - pd.Timedelta(days=7) if pd.notna(latest) else pd.NaT
    lines = [
        "# v0.6A.3 Candidate Health",
        "",
        f"- feature_time: {latest}",
        f"- btc_state: {_latest_btc_state(prepared)}",
        f"- data_stale: {_data_stale(prepared, config)}",
        f"- real_live_allowed: {config.experiment.real_live_allowed}",
        f"- tick_validation_status: {config.experiment.tick_validation_status}",
        "",
    ]
    for candidate in config.candidates:
        sig = signals[signals["candidate"].astype(str).eq(candidate.candidate)] if not signals.empty else signals
        trd = trades[trades["candidate"].astype(str).eq(candidate.candidate)] if not trades.empty else trades
        if candidate.candidate == config.experiment.primary_candidate:
            trd = trd[trd["portfolio_accepted"].fillna(False)] if not trd.empty else trd
        if pd.notna(cutoff):
            sig_time = pd.to_datetime(sig.get("feature_time"), utc=True, errors="coerce")
            trd_time = pd.to_datetime(trd.get("entry_time"), utc=True, errors="coerce")
            sig_7d = sig[sig_time >= cutoff] if not sig.empty else sig
            trd_7d = trd[trd_time >= cutoff] if not trd.empty else trd
        else:
            sig_7d = sig
            trd_7d = trd
        status, evaluation = _sample_status(len(trd))
        open_trades = (
            int(trd["exit_reason"].astype(str).eq("open").sum())
            if not trd.empty and "exit_reason" in trd
            else 0
        )
        net10_7d = _safe_float(trd_7d.get("net_return_10bp", pd.Series(dtype=float)).mean())
        net20_7d = _safe_float(trd_7d.get("net_return_20bp", pd.Series(dtype=float)).mean())
        lines.extend(
            [
                f"## {candidate.candidate} ({candidate.role})",
                "",
                f"- signals_7d: {len(sig_7d)}",
                f"- fills_7d: {len(trd_7d)}",
                f"- open_trades: {open_trades}",
                f"- net10_7d: {net10_7d:.4%}" if pd.notna(net10_7d) else "- net10_7d: n/a",
                f"- net20_7d: {net20_7d:.4%}" if pd.notna(net20_7d) else "- net20_7d: n/a",
                f"- filled_trades: {len(trd)}",
                f"- sample_status: {status}",
                f"- evaluation_status: {evaluation}",
                "",
            ]
        )
    return "\n".join(lines)


def _append_decision_log(
    report_root: Path,
    prepared: pd.DataFrame,
    trades: pd.DataFrame,
    config: V06A3Config,
) -> None:
    primary = _primary_portfolio_trades(trades, config)
    status, evaluation = _sample_status(len(primary))
    now = pd.Timestamp.now(tz="UTC")
    line = (
        f"{now.isoformat()} | IR2 primary filled_trades={len(primary)} | "
        f"current_btc_state={_latest_btc_state(prepared)} | sample_status={status} | "
        f"evaluation_status={evaluation} | real_live_allowed={config.experiment.real_live_allowed}\n"
    )
    path = report_root / "decision_log.md"
    if not path.exists():
        path.write_text(
            "# v0.6A.3 Decision Log\n\n"
            "No strategy evaluation is made before 30 primary filled trades; candidate checks start at 100.\n\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def write_v06a3_audit_files(
    report_root: Path,
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_signals: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V06A3Config,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    outputs = {
        "gate_audit": report_root / "gate_audit.csv",
        "invalidated_trades": report_root / "invalidated_trades.csv",
        "shadow_baselines_live": report_root / "shadow_baselines_live.csv",
        "regime_shadow": report_root / "regime_shadow.csv",
        "candidate_health": report_root / "candidate_health.md",
        "decision_log": report_root / "decision_log.md",
    }
    _gate_audit(signals, trades, prepared, config).to_csv(outputs["gate_audit"], index=False)
    _invalidated_trades(signals, config).to_csv(outputs["invalidated_trades"], index=False)
    _shadow_baselines_live(signals, trades, baseline_signals, baseline_trades, config).to_csv(
        outputs["shadow_baselines_live"],
        index=False,
    )
    _regime_shadow(signals, trades, config).to_csv(outputs["regime_shadow"], index=False)
    outputs["candidate_health"].write_text(_candidate_health(prepared, signals, trades, config), encoding="utf-8")
    _append_decision_log(report_root, prepared, trades, config)
    return outputs


def _write_status(
    report_root: Path,
    prepared: pd.DataFrame,
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    config: V06A3Config,
) -> None:
    latest_time = _latest_feature_time(prepared)
    primary = _primary_portfolio_trades(trades, config)
    net10 = _safe_float(primary.get("net_return_10bp", pd.Series(dtype=float)).mean())
    net20 = _safe_float(primary.get("net_return_20bp", pd.Series(dtype=float)).mean())
    sample_status, evaluation_status = _sample_status(len(primary))
    data_stale = _data_stale(prepared, config)
    lines = [
        "# v0.6A.3 Impulse Reclaim Paper-Live Status",
        "",
        f"- strategy_id: {config.experiment.strategy_id}",
        f"- primary_candidate: {config.experiment.primary_candidate}",
        f"- latest_feature_time: {latest_time}",
        f"- btc_state: {_latest_btc_state(prepared)}",
        f"- data_stale: {bool(data_stale)}",
        f"- tick_validation_status: {config.experiment.tick_validation_status}",
        f"- real_live_allowed: {config.experiment.real_live_allowed}",
        f"- total_candidate_signals: {len(signals)}",
        f"- primary_portfolio_trades: {len(primary)}",
        f"- sample_status: {sample_status}",
        f"- evaluation_status: {evaluation_status}",
        f"- primary_net_10bp_avg: {net10:.4%}" if pd.notna(net10) else "- primary_net_10bp_avg: n/a",
        f"- primary_net_20bp_avg: {net20:.4%}" if pd.notna(net20) else "- primary_net_20bp_avg: n/a",
        f"- baseline_trades: {len(baseline_trades)}",
    ]
    (report_root / "current_status.md").write_text("\n".join(lines), encoding="utf-8")

    summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=False)
    primary_summary = _summary(signals, trades, ["candidate", "candidate_role"], accepted_only=True)
    baseline_summary = _summary(
        pd.DataFrame(),
        baseline_trades,
        ["candidate", "baseline_kind"],
        accepted_only=False,
    )
    note = [
        "# v0.6A.3 Candidate Status",
        "",
        "Decision: IR2 is frozen as the primary paper-live candidate; IR1 and IR3 are shadow only.",
        "",
        "No parameter tuning, no real-live execution, and tick validation remains pending.",
        "",
        "Entry gate: signal and entry must both be BTC_up using as-of state only (`feature_time <= decision_time`). Existing positions continue to use their normal exit rule after regime changes.",
        "",
        "Historical revalidation: v0.6A.3.1 passed after stricter entry gate; IR2 remains a B- primary paper-live candidate.",
        "",
        "Risk: ex-top-month and month-capped edge is thin; this is still a conditional paper-live overlay, not a standalone real-live strategy.",
        "",
        "Real-live: disabled until tick validation is complete and sufficient paper-live sample is available.",
        "",
        "Do not evaluate before primary IR2 filled trades >= 30 for behavior checks and >= 100 for candidate checks.",
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
    note.append("## Candidate Shadows")
    if summary.empty:
        note.append("- No candidate shadow trades yet.")
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


def prepare_v06a3_features_from_history(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V06A3Config,
    days: int | None = None,
) -> pd.DataFrame:
    turnover = _read_existing_columns(feature_path, V03_TURNOVER_COLUMNS)
    rank30 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        30,
        RANK30_COL,
        "trailing_30d_turnover",
    )
    rank90 = _rank_table_with_lookback(
        turnover,
        instruments,
        base_config,
        90,
        RANK90_COL,
        "trailing_90d_turnover",
    )
    max_time = pd.to_datetime(turnover["bar_open_time"], utc=True, errors="coerce").max()
    cutoff = max_time - pd.Timedelta(days=days) if days is not None and pd.notna(max_time) else None
    max_top_n = max(candidate.universe_top_n for candidate in config.candidates)
    symbols = sorted(
        rank30[pd.to_numeric(rank30[RANK30_COL], errors="coerce") <= max_top_n]["symbol"]
        .dropna()
        .astype(str)
        .unique()
    )
    frames = []
    for symbol in symbols:
        data = _read_symbol_v04(feature_path, rank30, rank90, symbol, base_config)
        if data.empty:
            continue
        if cutoff is not None:
            data = data[pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce") >= cutoff].copy()
        frames.append(data)
    prepared = _concat_or_empty(frames)
    return add_v06a3_impulse_columns(prepared, config) if not prepared.empty else prepared


def write_v06a3_paper_live(
    feature_path: Path,
    instruments: pd.DataFrame,
    base_config: ExperimentConfig,
    config: V06A3Config,
    report_root: Path = REPORT_ROOT,
    paper_data_root: Path = PAPER_DATA_ROOT,
    days: int | None = 30,
) -> dict[str, Path]:
    report_root = ensure_dir(report_root)
    paper_data_root = ensure_dir(paper_data_root)
    prepared = prepare_v06a3_features_from_history(feature_path, instruments, base_config, config, days)
    signal_start_time = None
    if days is not None and not prepared.empty:
        latest = pd.to_datetime(prepared["feature_time"], utc=True, errors="coerce").max()
        signal_start_time = latest - pd.Timedelta(days=days)
    signals, trades, baseline_signals, baseline_trades = build_v06a3_paper_ledger(
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
        "current_status": report_root / "current_status.md",
        "candidate_status": report_root / "candidate_status.md",
        "gate_audit": report_root / "gate_audit.csv",
        "invalidated_trades": report_root / "invalidated_trades.csv",
        "shadow_baselines_live": report_root / "shadow_baselines_live.csv",
        "regime_shadow": report_root / "regime_shadow.csv",
        "candidate_health": report_root / "candidate_health.md",
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
    _write_status(report_root, prepared, signals, trades, baseline_trades, config)
    write_v06a3_audit_files(report_root, prepared, signals, trades, baseline_signals, baseline_trades, config)
    return outputs
