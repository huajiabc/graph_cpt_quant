from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd

from pressure_graph.backtest.simulator import AmbiguityPolicy, _funding_cost
from pressure_graph.config.models import ExecutionRule


KEYS = ["exchange", "symbol"]


@dataclass(frozen=True)
class EntryPolicy:
    name: str
    kind: str
    valid_bars: int = 0
    pullback_pct: float = 0.0
    persist_bars: int = 1
    breakout_window: int = 0


@dataclass(frozen=True)
class PolicyTrade:
    exchange: str
    symbol: str
    path_name: str
    entry_policy: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return: float
    net_return_ex_fee_slippage: float
    net_return_ex_fee_slippage_funding: float
    funding_cost: float
    cost_single_side_bps: float
    exit_reason: str
    holding_bars: int
    bars_from_signal_to_entry: int
    ambiguous_exit: bool


ENTRY_POLICIES = [
    EntryPolicy("E0_next_open", "next_open"),
    EntryPolicy("E1_persist_2_bars", "persist", persist_bars=2),
    EntryPolicy("E2_break_signal_high_valid_4_bars", "break_signal_high", valid_bars=4),
    EntryPolicy("E3_break_1h_high_valid_8_bars", "break_rolling_high", valid_bars=8, breakout_window=4),
    EntryPolicy("E4_pullback_0.5pct_valid_4_bars", "pullback", valid_bars=4, pullback_pct=0.005),
    EntryPolicy("E5_pullback_1.0pct_valid_8_bars", "pullback", valid_bars=8, pullback_pct=0.010),
    EntryPolicy(
        "E6_pullback_0.5pct_then_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.005,
    ),
    EntryPolicy(
        "E7_pullback_1.0pct_then_reclaim_signal_close_valid_8_bars",
        "pullback_reclaim",
        valid_bars=8,
        pullback_pct=0.010,
    ),
]


def _rolling_high_before_or_at(group: pd.DataFrame, idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    return float(pd.to_numeric(group.iloc[start : idx + 1]["high"], errors="coerce").max())


def _entry_for_policy(
    group: pd.DataFrame,
    signal_idx: int,
    state_signal_col: str,
    policy: EntryPolicy,
) -> tuple[int, float] | None:
    signal = group.iloc[signal_idx]
    signal_close = float(signal["close"])
    if policy.kind == "next_open":
        entry_idx = signal_idx + 1
        if entry_idx >= len(group):
            return None
        return entry_idx, float(group.iloc[entry_idx]["open"])

    if policy.kind == "persist":
        end_idx = signal_idx + policy.persist_bars - 1
        entry_idx = end_idx + 1
        if entry_idx >= len(group) or end_idx >= len(group):
            return None
        if not group.iloc[signal_idx : end_idx + 1][state_signal_col].fillna(False).astype(bool).all():
            return None
        return entry_idx, float(group.iloc[entry_idx]["open"])

    if policy.kind == "break_signal_high":
        trigger = float(signal["high"])
        for idx in range(signal_idx + 1, min(signal_idx + policy.valid_bars + 1, len(group))):
            if float(group.iloc[idx]["high"]) >= trigger:
                return idx, trigger
        return None

    if policy.kind == "break_rolling_high":
        trigger = _rolling_high_before_or_at(group, signal_idx, policy.breakout_window)
        for idx in range(signal_idx + 1, min(signal_idx + policy.valid_bars + 1, len(group))):
            if float(group.iloc[idx]["high"]) >= trigger:
                return idx, trigger
        return None

    if policy.kind == "pullback":
        trigger = signal_close * (1.0 - policy.pullback_pct)
        for idx in range(signal_idx + 1, min(signal_idx + policy.valid_bars + 1, len(group))):
            if float(group.iloc[idx]["low"]) <= trigger:
                return idx, trigger
        return None

    if policy.kind == "pullback_reclaim":
        trigger = signal_close * (1.0 - policy.pullback_pct)
        saw_pullback = False
        for idx in range(signal_idx + 1, min(signal_idx + policy.valid_bars + 1, len(group))):
            row = group.iloc[idx]
            if float(row["low"]) <= trigger:
                saw_pullback = True
            if saw_pullback and float(row["close"]) >= signal_close:
                entry_idx = idx + 1
                if entry_idx >= len(group):
                    return None
                return entry_idx, float(group.iloc[entry_idx]["open"])
        return None

    raise ValueError(f"unknown policy kind: {policy.kind}")


def _exit_trade(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: ExecutionRule,
    ambiguity_policy: AmbiguityPolicy,
) -> tuple[int, float, str, bool]:
    tp_price = entry_price * (1.0 + rule.tp)
    sl_price = entry_price * (1.0 - rule.sl)
    max_exit_idx = min(entry_idx + rule.max_hold_bars - 1, len(group) - 1)
    exit_idx = max_exit_idx
    exit_price = float(group.iloc[max_exit_idx]["close"])
    exit_reason = "max_hold"
    ambiguous = False

    for idx in range(entry_idx, max_exit_idx + 1):
        row = group.iloc[idx]
        high_hit = float(row["high"]) >= tp_price
        low_hit = float(row["low"]) <= sl_price
        if high_hit and low_hit:
            ambiguous = True
            exit_idx = idx
            if ambiguity_policy == "skip":
                return -1, np.nan, "skip_ambiguous", True
            if ambiguity_policy == "tp_first":
                return idx, tp_price, "tp_ambiguous", True
            return idx, sl_price, "sl_ambiguous", True
        if low_hit:
            return idx, sl_price, "sl", ambiguous
        if high_hit:
            return idx, tp_price, "tp", ambiguous
    return exit_idx, exit_price, exit_reason, ambiguous


def simulate_entry_policy_trades(
    df: pd.DataFrame,
    signal_col: str,
    path_name: str,
    policy: EntryPolicy,
    rule: ExecutionRule,
    cost_single_side_bps: float,
    ambiguity_policy: AmbiguityPolicy = "sl_first",
    one_position_per_symbol: bool = True,
    state_signal_col: str | None = None,
    rule_resolver: Callable[[pd.Series], ExecutionRule] | None = None,
) -> pd.DataFrame:
    if signal_col not in df.columns or df.empty:
        return pd.DataFrame()
    if state_signal_col is None:
        state_signal_col = signal_col.removesuffix("_event")
    if state_signal_col not in df.columns:
        state_signal_col = signal_col
    trades: list[PolicyTrade] = []
    for _, group in df.sort_values(KEYS + ["bar_open_time"]).groupby(
        KEYS, sort=False, observed=True
    ):
        group = group.reset_index(drop=True)
        active_until = -1
        for signal_idx, is_signal in enumerate(group[signal_col].fillna(False).astype(bool)):
            if not is_signal or (one_position_per_symbol and signal_idx <= active_until):
                continue
            entry = _entry_for_policy(group, signal_idx, state_signal_col, policy)
            if entry is None:
                continue
            entry_idx, entry_price = entry
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            trade_rule = rule_resolver(group.iloc[entry_idx]) if rule_resolver else rule
            exit_idx, exit_price, exit_reason, ambiguous = _exit_trade(
                group, entry_idx, entry_price, trade_rule, ambiguity_policy
            )
            if exit_idx < 0:
                continue
            gross = exit_price / entry_price - 1.0
            round_trip_cost = 2.0 * cost_single_side_bps / 10_000.0
            net = gross - round_trip_cost
            entry_time = pd.Timestamp(group.iloc[entry_idx]["bar_open_time"])
            exit_time = pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
            funding_cost = _funding_cost(group, entry_time, exit_time)
            trade = PolicyTrade(
                exchange=str(group.iloc[entry_idx]["exchange"]),
                symbol=str(group.iloc[entry_idx]["symbol"]),
                path_name=path_name,
                entry_policy=policy.name,
                signal_time=pd.Timestamp(group.iloc[signal_idx]["feature_time"]),
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=gross,
                net_return_ex_fee_slippage=net,
                net_return_ex_fee_slippage_funding=net - funding_cost,
                funding_cost=funding_cost,
                cost_single_side_bps=cost_single_side_bps,
                exit_reason=exit_reason,
                holding_bars=exit_idx - entry_idx + 1,
                bars_from_signal_to_entry=entry_idx - signal_idx,
                ambiguous_exit=ambiguous,
            )
            trades.append(trade)
            if one_position_per_symbol:
                active_until = exit_idx
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(trade) for trade in trades])
