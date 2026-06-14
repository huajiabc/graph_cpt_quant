"""Squeeze-aware short trade simulator (pure functions).

The short instruction doc (§5) demands stricter risk accounting than the long
path: higher slippage, a shorter validity window, and explicit short-squeeze
tracking. This module walks forward bars from a short entry and records not
just the exit but the full adverse (up) excursion so the atlas can separate
"direction was wrong" from "direction was right but squeezed out first".

Short P&L convention (entry high, cover low is profit):
    gross_return = (entry_price - exit_price) / entry_price

A short stop is ABOVE entry (price rallied); a short take-profit is BELOW entry
(price fell). Ambiguous bars (both touched in one bar) resolve stop-first by
default — the conservative assumption for a squeeze-sensitive book.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ShortExitRule:
    take_profit: float  # downside target magnitude, e.g. 0.03 = cover 3% lower
    stop_loss: float  # upside stop magnitude, e.g. 0.02 = stop 2% higher
    max_hold_bars: int


@dataclass(frozen=True)
class ShortExit:
    exit_idx: int
    exit_price: float
    exit_reason: str
    gross_return: float
    max_adverse_excursion: float  # peak up-move fraction during the hold (>=0)
    max_favorable_excursion: float  # peak down-move fraction during the hold (<=0)
    squeezed: bool  # adverse stop touched before the downside target
    holding_bars: int


def simulate_short_exit(
    group: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    rule: ShortExitRule,
    ambiguity: str = "stop_first",
) -> ShortExit:
    """Walk forward from ``entry_idx`` and resolve one short trade."""
    tp_price = entry_price * (1.0 - rule.take_profit)
    sl_price = entry_price * (1.0 + rule.stop_loss)
    max_exit_idx = min(entry_idx + rule.max_hold_bars - 1, len(group) - 1)
    running_high = entry_price
    running_low = entry_price
    for idx in range(entry_idx, max_exit_idx + 1):
        row = group.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        running_high = max(running_high, high)
        running_low = min(running_low, low)
        stop_hit = high >= sl_price
        target_hit = low <= tp_price
        if stop_hit and target_hit:
            # Ambiguous bar: conservative books assume the squeeze first.
            if ambiguity == "target_first":
                return _build_exit(entry_price, idx, tp_price, "tp_ambiguous", running_high, running_low, True, entry_idx)
            return _build_exit(entry_price, idx, sl_price, "stop_ambiguous", running_high, running_low, True, entry_idx)
        if stop_hit:
            return _build_exit(entry_price, idx, sl_price, "stop", running_high, running_low, True, entry_idx)
        if target_hit:
            return _build_exit(entry_price, idx, tp_price, "take_profit", running_high, running_low, False, entry_idx)
    close_price = float(group.iloc[max_exit_idx]["close"])
    return _build_exit(entry_price, max_exit_idx, close_price, "max_hold", running_high, running_low, False, entry_idx)


def _build_exit(
    entry_price: float,
    exit_idx: int,
    exit_price: float,
    reason: str,
    running_high: float,
    running_low: float,
    squeezed: bool,
    entry_idx: int,
) -> ShortExit:
    return ShortExit(
        exit_idx=exit_idx,
        exit_price=exit_price,
        exit_reason=reason,
        gross_return=(entry_price - exit_price) / entry_price,
        max_adverse_excursion=running_high / entry_price - 1.0,
        max_favorable_excursion=running_low / entry_price - 1.0,
        squeezed=squeezed,
        holding_bars=exit_idx - entry_idx + 1,
    )


def short_net_return(gross: float, cost_single_side_bps: float, extra_slippage_bps: float) -> float:
    """Round-trip short net of fees and the doc-mandated extra short slippage."""
    round_trip = 2.0 * (cost_single_side_bps + extra_slippage_bps) / 10_000.0
    return gross - round_trip


__all__ = [
    "ShortExit",
    "ShortExitRule",
    "short_net_return",
    "simulate_short_exit",
]
