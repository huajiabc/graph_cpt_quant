from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10_short_mirror import (
    ShortCandidate,
    simulate_short_candidate,
)


def _rows() -> pd.DataFrame:
    base = pd.Timestamp("2026-06-01T00:00:00Z")
    rows = []
    prices = [
        (100.0, 101.0, 99.0, 100.0),
        (100.0, 102.0, 98.5, 98.8),
        (99.8, 100.0, 96.5, 97.0),
    ]
    for idx, (open_, high, low, close) in enumerate(prices):
        start = base + pd.Timedelta(minutes=15 * idx)
        rows.append(
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "bar_open_time": start,
                "bar_close_time": start + pd.Timedelta(minutes=15),
                "feature_time": start + pd.Timedelta(minutes=15),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "dynamic_all_rank": 1,
                "symbol_volatility_percentile": 20.0,
                "btc_market_state": "BTC_down",
            }
        )
    return pd.DataFrame(rows)


def test_short_bounce_reject_enters_and_profits_on_down_move() -> None:
    data = _rows()
    data["bear_gate"] = [True, True, True]
    data["bear_event"] = [True, False, False]
    candidate = ShortCandidate(
        "S",
        "bearish_mirror",
        "bear_gate",
        "bear_event",
        "bounce_reject",
        strict_entry_gate=True,
    )

    trades = simulate_short_candidate(data, candidate)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["entry_reason"] == "bounce_reject_next_open"
    assert trade["exit_reason"] == "tp"
    assert trade["net10"] > 0


def test_long_failure_break_signal_low_enters_short() -> None:
    data = _rows()
    data["long_gate"] = [True, False, False]
    data["long_event"] = [True, False, False]
    candidate = ShortCandidate(
        "F",
        "long_failure",
        "long_gate",
        "long_event",
        "break_signal_low",
        strict_entry_gate=False,
    )

    trades = simulate_short_candidate(data, candidate)

    assert len(trades) == 1
    assert trades.iloc[0]["entry_reason"] == "break_signal_low_next_open"
    assert trades.iloc[0]["net10"] > 0
