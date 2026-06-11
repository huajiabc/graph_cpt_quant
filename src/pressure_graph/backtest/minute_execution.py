from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable

import pandas as pd

from pressure_graph.backtest.entry_policies import EntryPolicy
from pressure_graph.config.models import ExecutionRule
from pressure_graph.io import ensure_dir


@dataclass(frozen=True)
class MinuteTrade:
    exchange: str
    symbol: str
    path_name: str
    entry_policy: str
    execution_rule: str
    cost_single_side_bps: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    gross_return: float
    net_expectancy: float
    exit_reason: str
    bars_from_signal_to_entry_1m: int
    holding_minutes: int
    unresolved_1m_same_bar: bool


def _minute_window(minute_bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    bars = minute_bars[
        (minute_bars["bar_open_time"] >= start) & (minute_bars["bar_open_time"] <= end)
    ].copy()
    return bars.sort_values("bar_open_time").reset_index(drop=True)


def _rolling_high_15m(signal_rows: pd.DataFrame, idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    return float(pd.to_numeric(signal_rows.iloc[start : idx + 1]["high"], errors="coerce").max())


def _entry_price_1m(
    signal_rows: pd.DataFrame,
    signal_idx: int,
    minute_bars: pd.DataFrame,
    policy: EntryPolicy,
    signal_col: str,
) -> tuple[pd.Timestamp, float] | None:
    signal = signal_rows.iloc[signal_idx]
    signal_time = pd.Timestamp(signal["feature_time"])
    signal_close = float(signal["close"])
    start = signal_time

    if policy.kind == "next_open":
        bars = _minute_window(minute_bars, start, start)
        if bars.empty:
            return None
        row = bars.iloc[0]
        return pd.Timestamp(row["bar_open_time"]), float(row["open"])

    if policy.kind == "persist":
        end_idx = signal_idx + policy.persist_bars - 1
        if end_idx >= len(signal_rows):
            return None
        if not signal_rows.iloc[signal_idx : end_idx + 1][signal_col].fillna(False).astype(bool).all():
            return None
        entry_time = pd.Timestamp(signal_rows.iloc[end_idx]["feature_time"])
        bars = _minute_window(minute_bars, entry_time, entry_time)
        if bars.empty:
            return None
        row = bars.iloc[0]
        return pd.Timestamp(row["bar_open_time"]), float(row["open"])

    valid_end = start + pd.Timedelta(minutes=15 * policy.valid_bars - 1)
    bars = _minute_window(minute_bars, start, valid_end)
    if bars.empty:
        return None

    if policy.kind == "break_signal_high":
        trigger = float(signal["high"])
        for row in bars.itertuples(index=False):
            if float(row.high) >= trigger:
                return pd.Timestamp(row.bar_open_time), trigger
        return None

    if policy.kind == "break_rolling_high":
        trigger = _rolling_high_15m(signal_rows, signal_idx, policy.breakout_window)
        for row in bars.itertuples(index=False):
            if float(row.high) >= trigger:
                return pd.Timestamp(row.bar_open_time), trigger
        return None

    if policy.kind == "pullback":
        trigger = signal_close * (1.0 - policy.pullback_pct)
        for row in bars.itertuples(index=False):
            if float(row.low) <= trigger:
                return pd.Timestamp(row.bar_open_time), trigger
        return None

    if policy.kind == "pullback_reclaim":
        trigger = signal_close * (1.0 - policy.pullback_pct)
        saw_pullback = False
        for pos, row in enumerate(bars.itertuples(index=False)):
            if float(row.low) <= trigger:
                saw_pullback = True
            if saw_pullback and float(row.close) >= signal_close:
                next_pos = pos + 1
                if next_pos >= len(bars):
                    return None
                next_row = bars.iloc[next_pos]
                return pd.Timestamp(next_row["bar_open_time"]), float(next_row["open"])
        return None

    raise ValueError(f"unknown entry policy: {policy.kind}")


def _resolve_exit_1m(
    minute_bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    rule: ExecutionRule,
) -> tuple[pd.Timestamp, float, str, bool, int] | None:
    start = entry_time
    end = entry_time + pd.Timedelta(minutes=15 * rule.max_hold_bars - 1)
    bars = _minute_window(minute_bars, start, end)
    if bars.empty:
        return None
    tp_price = entry_price * (1.0 + rule.tp)
    sl_price = entry_price * (1.0 - rule.sl)
    unresolved = False
    for offset, row in enumerate(bars.itertuples(index=False), start=1):
        high_hit = float(row.high) >= tp_price
        low_hit = float(row.low) <= sl_price
        if high_hit and low_hit:
            unresolved = True
            return pd.Timestamp(row.bar_open_time), sl_price, "sl_1m_same_bar", unresolved, offset
        if low_hit:
            return pd.Timestamp(row.bar_open_time), sl_price, "sl", unresolved, offset
        if high_hit:
            return pd.Timestamp(row.bar_open_time), tp_price, "tp", unresolved, offset
    row = bars.iloc[-1]
    return pd.Timestamp(row["bar_open_time"]), float(row["close"]), "max_hold", unresolved, len(bars)


def simulate_1m_execution(
    signals_15m: pd.DataFrame,
    minute_bars: pd.DataFrame,
    signal_col: str,
    path_name: str,
    policy: EntryPolicy,
    execution_rule_name: str,
    execution_rule: ExecutionRule,
    cost_single_side_bps: float,
    rule_resolver: Callable[[pd.Series], ExecutionRule] | None = None,
) -> pd.DataFrame:
    if signals_15m.empty or minute_bars.empty:
        return pd.DataFrame()
    rows: list[MinuteTrade] = []
    state_signal_col = signal_col.removesuffix("_event")
    for (exchange, symbol), group in signals_15m.sort_values(
        ["exchange", "symbol", "bar_open_time"]
    ).groupby(["exchange", "symbol"], sort=False):
        group = group.reset_index(drop=True)
        minutes = minute_bars[
            (minute_bars["exchange"] == exchange) & (minute_bars["symbol"] == symbol)
        ].copy()
        if minutes.empty:
            continue
        active_until = pd.Timestamp.min.tz_localize("UTC")
        for signal_idx, is_signal in enumerate(group[signal_col].fillna(False).astype(bool)):
            if not is_signal:
                continue
            signal_time = pd.Timestamp(group.iloc[signal_idx]["feature_time"])
            if signal_time <= active_until:
                continue
            entry = _entry_price_1m(group, signal_idx, minutes, policy, state_signal_col)
            if entry is None:
                continue
            entry_time, entry_price = entry
            trade_rule = rule_resolver(group.iloc[signal_idx]) if rule_resolver else execution_rule
            exit_result = _resolve_exit_1m(minutes, entry_time, entry_price, trade_rule)
            if exit_result is None:
                continue
            exit_time, exit_price, exit_reason, unresolved, holding_minutes = exit_result
            gross = exit_price / entry_price - 1.0
            net = gross - 2.0 * cost_single_side_bps / 10_000.0
            rows.append(
                MinuteTrade(
                    exchange=str(exchange),
                    symbol=str(symbol),
                    path_name=path_name,
                    entry_policy=policy.name,
                    execution_rule=execution_rule_name,
                    cost_single_side_bps=cost_single_side_bps,
                    signal_time=signal_time,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    gross_return=gross,
                    net_expectancy=net,
                    exit_reason=exit_reason,
                    bars_from_signal_to_entry_1m=int((entry_time - signal_time) / pd.Timedelta(minutes=1)),
                    holding_minutes=holding_minutes,
                    unresolved_1m_same_bar=unresolved,
                )
            )
            active_until = exit_time
    return pd.DataFrame([asdict(row) for row in rows])


def summarize_1m_execution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    group_cols = ["path_name", "entry_policy", "execution_rule", "cost_single_side_bps"]
    rows = []
    for key, group in trades.groupby(group_cols, sort=False):
        path_name, entry_policy, execution_rule, cost = key
        rows.append(
            {
                "path_name": path_name,
                "entry_policy": entry_policy,
                "execution_rule": execution_rule,
                "cost_single_side_bps": cost,
                "trade_n": len(group),
                "net_expectancy_1m": float(group["net_expectancy"].mean()),
                "gross_expectancy_1m": float(group["gross_return"].mean()),
                "tp_rate_1m": float(group["exit_reason"].astype(str).str.startswith("tp").mean()),
                "sl_rate_1m": float(group["exit_reason"].astype(str).str.startswith("sl").mean()),
                "unresolved_1m_same_bar_rate": float(group["unresolved_1m_same_bar"].mean()),
                "median_minutes_to_entry": float(group["bars_from_signal_to_entry_1m"].median()),
                "median_holding_minutes": float(group["holding_minutes"].median()),
            }
        )
    return pd.DataFrame(rows)


def write_1m_execution_outputs(trades: pd.DataFrame, report_root: Path) -> dict[str, Path]:
    ensure_dir(report_root)
    trades_path = report_root / "entry_policy_1m_trades.csv"
    summary_path = report_root / "entry_policy_1m_comparison.csv"
    trades.to_csv(trades_path, index=False)
    summarize_1m_execution(trades).to_csv(summary_path, index=False)
    return {"entry_policy_1m_trades": trades_path, "entry_policy_1m_comparison": summary_path}
