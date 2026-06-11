from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from pressure_graph.config.models import ExecutionRule


KEYS = ["exchange", "symbol"]
AmbiguityPolicy = Literal["sl_first", "tp_first", "skip"]


@dataclass(frozen=True)
class TradeResult:
    exchange: str
    symbol: str
    path_name: str
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
    ambiguous: bool


def _funding_cost(group: pd.DataFrame, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
    if "funding_time" not in group.columns or "funding_rate_settled" not in group.columns:
        return 0.0
    funding = group[["funding_time", "funding_rate_settled"]].dropna()
    if funding.empty:
        return 0.0
    mask = (funding["funding_time"] >= entry_time) & (funding["funding_time"] <= exit_time)
    return float(pd.to_numeric(funding.loc[mask, "funding_rate_settled"], errors="coerce").sum())


def _simulate_one(
    group: pd.DataFrame,
    signal_idx: int,
    path_name: str,
    rule: ExecutionRule,
    cost_single_side_bps: float,
    ambiguity_policy: AmbiguityPolicy,
) -> TradeResult | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(group):
        return None
    entry_row = group.iloc[entry_idx]
    entry_price = float(entry_row["open"])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    tp_price = entry_price * (1.0 + rule.tp)
    sl_price = entry_price * (1.0 - rule.sl)
    max_exit_idx = min(entry_idx + rule.max_hold_bars - 1, len(group) - 1)

    exit_idx = max_exit_idx
    exit_price = float(group.iloc[max_exit_idx]["close"])
    exit_reason = "max_hold"
    ambiguous = False

    for bar_idx in range(entry_idx, max_exit_idx + 1):
        row = group.iloc[bar_idx]
        high_hit = float(row["high"]) >= tp_price
        low_hit = float(row["low"]) <= sl_price
        if high_hit and low_hit:
            ambiguous = True
            if ambiguity_policy == "skip":
                return None
            exit_idx = bar_idx
            if ambiguity_policy == "tp_first":
                exit_price = tp_price
                exit_reason = "tp_ambiguous"
            else:
                exit_price = sl_price
                exit_reason = "sl_ambiguous"
            break
        if low_hit:
            exit_idx = bar_idx
            exit_price = sl_price
            exit_reason = "sl"
            break
        if high_hit:
            exit_idx = bar_idx
            exit_price = tp_price
            exit_reason = "tp"
            break

    gross_return = exit_price / entry_price - 1.0
    round_trip_cost = 2.0 * cost_single_side_bps / 10_000.0
    net = gross_return - round_trip_cost
    entry_time = pd.Timestamp(entry_row["bar_open_time"])
    exit_time = pd.Timestamp(group.iloc[exit_idx]["bar_close_time"])
    funding_cost = _funding_cost(group, entry_time, exit_time)
    return TradeResult(
        exchange=str(entry_row["exchange"]),
        symbol=str(entry_row["symbol"]),
        path_name=path_name,
        signal_time=pd.Timestamp(group.iloc[signal_idx]["feature_time"]),
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=gross_return,
        net_return_ex_fee_slippage=net,
        net_return_ex_fee_slippage_funding=net - funding_cost,
        funding_cost=funding_cost,
        cost_single_side_bps=cost_single_side_bps,
        exit_reason=exit_reason,
        holding_bars=exit_idx - entry_idx + 1,
        ambiguous=ambiguous,
    )


def simulate_trades(
    df: pd.DataFrame,
    signal_col: str,
    path_name: str,
    rule: ExecutionRule,
    cost_single_side_bps: float,
    ambiguity_policy: AmbiguityPolicy = "sl_first",
    one_position_per_symbol: bool = True,
) -> pd.DataFrame:
    trades: list[TradeResult] = []
    if signal_col not in df.columns or df.empty:
        return pd.DataFrame()
    for _, group in df.sort_values(KEYS + ["bar_open_time"]).groupby(KEYS, sort=False):
        group = group.reset_index(drop=True)
        active_until = -1
        for idx, is_signal in enumerate(group[signal_col].fillna(False).astype(bool)):
            if not is_signal:
                continue
            if one_position_per_symbol and idx <= active_until:
                continue
            trade = _simulate_one(
                group,
                idx,
                path_name,
                rule,
                cost_single_side_bps,
                ambiguity_policy,
            )
            if trade is None:
                continue
            trades.append(trade)
            if one_position_per_symbol:
                active_until = idx + trade.holding_bars
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(trade) for trade in trades])
