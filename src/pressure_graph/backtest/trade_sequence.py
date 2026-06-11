from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable

import pandas as pd

from pressure_graph.config.models import ExecutionRule
from pressure_graph.config.v02 import FillPolicy
from pressure_graph.io import ensure_dir


@dataclass(frozen=True)
class SequenceTrade:
    candidate: str
    exchange: str
    symbol: str
    path_name: str
    entry_policy: str
    execution_rule: str
    fill_policy: str
    cost_single_side_bps: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp | None
    exit_time: pd.Timestamp | None
    entry_price: float | None
    exit_price: float | None
    gross_return: float | None
    net_return: float | None
    exit_reason: str
    filled: bool
    missed_then_hit: bool
    holding_minutes: float | None
    bars_from_signal_to_entry_minutes: float | None


def _trades_between(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return trades[(trades["timestamp"] >= start) & (trades["timestamp"] <= end)].copy()


def _first_trade_at_or_after(trades: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    subset = trades[trades["timestamp"] >= ts]
    if subset.empty:
        return None
    return subset.iloc[0]


def _rolling_high(signal_rows: pd.DataFrame, idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    return float(pd.to_numeric(signal_rows.iloc[start : idx + 1]["high"], errors="coerce").max())


def _touch_price_with_buffer(price: float, buffer_bps: float) -> float:
    return price * (1.0 - buffer_bps / 10_000.0)


def _entry_trade_sequence(
    signal_rows: pd.DataFrame,
    signal_idx: int,
    trades: pd.DataFrame,
    entry_policy: str,
    fill_policy: FillPolicy,
    state_signal_col: str,
) -> tuple[pd.Timestamp, float] | None:
    signal = signal_rows.iloc[signal_idx]
    signal_time = pd.Timestamp(signal["feature_time"])
    signal_close = float(signal["close"])

    if entry_policy == "E0_next_open":
        trade = _first_trade_at_or_after(trades, signal_time)
        if trade is None:
            return None
        return pd.Timestamp(trade["timestamp"]), float(trade["price"])

    if entry_policy == "E1_persist_2_bars":
        end_idx = signal_idx + 1
        if end_idx >= len(signal_rows):
            return None
        if not signal_rows.iloc[signal_idx : end_idx + 1][state_signal_col].fillna(False).all():
            return None
        entry_time = pd.Timestamp(signal_rows.iloc[end_idx]["feature_time"])
        trade = _first_trade_at_or_after(trades, entry_time)
        if trade is None:
            return None
        return pd.Timestamp(trade["timestamp"]), float(trade["price"])

    if entry_policy == "E2_break_signal_high_valid_4_bars":
        trigger = float(signal["high"])
        valid_end = signal_time + pd.Timedelta(minutes=59)
        for row in _trades_between(trades, signal_time, valid_end).itertuples(index=False):
            if float(row.price) >= trigger:
                return pd.Timestamp(row.timestamp), trigger
        return None

    if entry_policy == "E3_break_1h_high_valid_8_bars":
        trigger = _rolling_high(signal_rows, signal_idx, 4)
        valid_end = signal_time + pd.Timedelta(minutes=119)
        for row in _trades_between(trades, signal_time, valid_end).itertuples(index=False):
            if float(row.price) >= trigger:
                return pd.Timestamp(row.timestamp), trigger
        return None

    if entry_policy in {
        "E4_pullback_0.5pct_valid_4_bars",
        "E5_pullback_1.0pct_valid_8_bars",
        "E6_pullback_0.5pct_then_reclaim_signal_close_valid_8_bars",
        "E7_pullback_1.0pct_then_reclaim_signal_close_valid_8_bars",
    }:
        pullback_pct = 0.005 if "0.5pct" in entry_policy else 0.010
        valid_minutes = 59 if "valid_4_bars" in entry_policy else 119
        valid_end = signal_time + pd.Timedelta(minutes=valid_minutes)
        limit_price = signal_close * (1.0 - pullback_pct)
        required_touch = _touch_price_with_buffer(limit_price, fill_policy.touch_buffer_bps)
        window = _trades_between(trades, signal_time, valid_end)
        if window.empty:
            return None
        touched = False
        for row in window.itertuples(index=False):
            price = float(row.price)
            ts = pd.Timestamp(row.timestamp)
            if price <= required_touch:
                touched = True
                if "then_reclaim" not in entry_policy and fill_policy.mode != "reclaim_market":
                    return ts, limit_price
            if touched and ("then_reclaim" in entry_policy or fill_policy.mode == "reclaim_market"):
                if price >= signal_close:
                    return ts, price
        return None

    raise ValueError(f"unsupported entry policy: {entry_policy}")


def _exit_trade_sequence(
    trades: pd.DataFrame,
    entry_time: pd.Timestamp,
    entry_price: float,
    rule: ExecutionRule,
) -> tuple[pd.Timestamp, float, str] | None:
    end = entry_time + pd.Timedelta(minutes=15 * rule.max_hold_bars)
    window = _trades_between(trades, entry_time, end)
    if window.empty:
        return None
    tp = entry_price * (1.0 + rule.tp)
    sl = entry_price * (1.0 - rule.sl)
    for row in window.itertuples(index=False):
        price = float(row.price)
        ts = pd.Timestamp(row.timestamp)
        if price <= sl:
            return ts, sl, "sl"
        if price >= tp:
            return ts, tp, "tp"
    row = window.iloc[-1]
    return pd.Timestamp(row["timestamp"]), float(row["price"]), "timeout"


def simulate_trade_sequence_candidate(
    signal_rows: pd.DataFrame,
    trades: pd.DataFrame,
    candidate: str,
    path_name: str,
    signal_col: str,
    entry_policy: str,
    execution_rule_name: str,
    execution_rule: ExecutionRule,
    fill_policy: FillPolicy,
    cost_single_side_bps: float,
    rule_resolver: Callable[[pd.Series], ExecutionRule] | None = None,
) -> pd.DataFrame:
    rows: list[SequenceTrade] = []
    state_signal_col = signal_col.removesuffix("_event")
    for (exchange, symbol), group in signal_rows.sort_values(
        ["exchange", "symbol", "bar_open_time"]
    ).groupby(["exchange", "symbol"], sort=False):
        group = group.reset_index(drop=True)
        symbol_trades = trades[
            (trades["exchange"] == exchange) & (trades["symbol"] == symbol)
        ].sort_values("timestamp")
        active_until = pd.Timestamp.min.tz_localize("UTC")
        for idx, is_signal in enumerate(group[signal_col].fillna(False).astype(bool)):
            if not is_signal:
                continue
            signal_time = pd.Timestamp(group.iloc[idx]["feature_time"])
            if signal_time <= active_until:
                continue
            entry = _entry_trade_sequence(
                group, idx, symbol_trades, entry_policy, fill_policy, state_signal_col
            )
            if entry is None:
                rows.append(
                    SequenceTrade(
                        candidate,
                        str(exchange),
                        str(symbol),
                        path_name,
                        entry_policy,
                        execution_rule_name,
                        fill_policy.name,
                        cost_single_side_bps,
                        signal_time,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "missed_entry",
                        False,
                        False,
                        None,
                        None,
                    )
                )
                continue
            entry_time, entry_price = entry
            trade_rule = rule_resolver(group.iloc[idx]) if rule_resolver else execution_rule
            exit_result = _exit_trade_sequence(symbol_trades, entry_time, entry_price, trade_rule)
            if exit_result is None:
                continue
            exit_time, exit_price, exit_reason = exit_result
            gross = exit_price / entry_price - 1.0
            net = gross - 2.0 * cost_single_side_bps / 10_000.0
            active_until = exit_time
            rows.append(
                SequenceTrade(
                    candidate,
                    str(exchange),
                    str(symbol),
                    path_name,
                    entry_policy,
                    execution_rule_name,
                    fill_policy.name,
                    cost_single_side_bps,
                    signal_time,
                    entry_time,
                    exit_time,
                    entry_price,
                    exit_price,
                    gross,
                    net,
                    exit_reason,
                    True,
                    False,
                    float((exit_time - entry_time) / pd.Timedelta(minutes=1)),
                    float((entry_time - signal_time) / pd.Timedelta(minutes=1)),
                )
            )
    return pd.DataFrame([asdict(row) for row in rows])


def summarize_sequence_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    group_cols = [
        "candidate",
        "path_name",
        "entry_policy",
        "execution_rule",
        "fill_policy",
        "cost_single_side_bps",
    ]
    rows = []
    for key, group in trades.groupby(group_cols, sort=False):
        filled = group[group["filled"].fillna(False)]
        returns = pd.to_numeric(filled["net_return"], errors="coerce")
        gross = pd.to_numeric(filled["gross_return"], errors="coerce")
        rows.append(
            {
                **dict(zip(group_cols, key, strict=False)),
                "execution_granularity": "trade_sequence",
                "trades": int(len(filled)),
                "signals": int(len(group)),
                "fill_rate": float(len(filled) / len(group)) if len(group) else float("nan"),
                "gross_expectancy": float(gross.mean()) if len(filled) else float("nan"),
                "net_expectancy": float(returns.mean()) if len(filled) else float("nan"),
                "tp_first_rate": float(filled["exit_reason"].eq("tp").mean()) if len(filled) else float("nan"),
                "sl_first_rate": float(filled["exit_reason"].eq("sl").mean()) if len(filled) else float("nan"),
                "timeout_rate": float(filled["exit_reason"].eq("timeout").mean()) if len(filled) else float("nan"),
                "same_bar_ambiguous_rate": 0.0,
                "median_holding_minutes": float(filled["holding_minutes"].median()) if len(filled) else float("nan"),
                "p25_return": float(returns.quantile(0.25)) if len(filled) else float("nan"),
                "p75_return": float(returns.quantile(0.75)) if len(filled) else float("nan"),
                "max_loss": float(returns.min()) if len(filled) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def write_trade_sequence_outputs(trades: pd.DataFrame, report_root: Path) -> dict[str, Path]:
    ensure_dir(report_root)
    trades_path = report_root / "tick_execution_trades.csv"
    summary_path = report_root / "tick_execution_comparison.csv"
    trades.to_csv(trades_path, index=False)
    summarize_sequence_trades(trades).to_csv(summary_path, index=False)
    return {"tick_execution_trades": trades_path, "tick_execution_comparison": summary_path}
