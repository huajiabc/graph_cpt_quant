from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v167_alt_bucket_vol_front_btc_straddle import (
    calculate_short_straddle_trade,
    calculate_straddle_trade,
    causal_percentile,
    select_atm_straddle,
)


def _option_row(
    option_type: str,
    strike: float,
    expiry: str,
    bid: float,
    ask: float,
    delta: float,
) -> dict[str, object]:
    return {
        "snapshot_time": pd.Timestamp("2023-07-01 01:00:00Z"),
        "expiration_time": pd.Timestamp(expiry),
        "strike_price": strike,
        "option_type": option_type,
        "symbol": f"BTC-X-{strike:.0f}-{option_type}",
        "best_bid_price": bid,
        "best_ask_price": ask,
        "best_bid_qty": 5.0,
        "best_ask_qty": 5.0,
        "delta": delta,
    }


def test_causal_percentile_excludes_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0])
    result = causal_percentile(values, lookback=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == 1.0


def test_select_atm_straddle_prefers_nearest_strike_at_nearest_expiry() -> None:
    rows = []
    for strike in (29_000.0, 30_000.0):
        rows.extend(
            [
                _option_row("C", strike, "2023-07-31 08:00:00Z", 900, 910, 0.5),
                _option_row("P", strike, "2023-07-31 08:00:00Z", 850, 860, -0.5),
            ]
        )
    selected = select_atm_straddle(pd.DataFrame(rows), btc_price=30_100)
    assert set(selected["option_type"]) == {"C", "P"}
    assert selected["strike_price"].eq(30_000).all()


def test_calculate_straddle_trade_crosses_spread_and_hedges_entry_delta() -> None:
    entry = pd.DataFrame(
        [
            _option_row("C", 30_000, "2023-07-31 08:00:00Z", 990, 1_000, 0.55),
            _option_row("P", 30_000, "2023-07-31 08:00:00Z", 890, 900, -0.45),
        ]
    )
    exit_ = entry.copy()
    exit_["best_bid_price"] = [1_090, 990]
    result = calculate_straddle_trade(entry, exit_, entry_btc=30_000, exit_btc=30_300)
    assert np.isclose(result["entry_delta"], 0.10)
    assert np.isclose(result["hedge_quantity"], -0.10)
    assert np.isclose(result["option_pnl"], 180.0)
    assert np.isclose(result["hedge_pnl"], -30.0)
    assert result["primary_net_return"] < result["gross_return"]
    assert result["stress_net_return"] < result["primary_net_return"]


def test_short_straddle_uses_bid_to_ask_and_opposite_hedge() -> None:
    entry = pd.DataFrame(
        [
            _option_row("C", 30_000, "2023-07-31 08:00:00Z", 990, 1_000, 0.55),
            _option_row("P", 30_000, "2023-07-31 08:00:00Z", 890, 900, -0.45),
        ]
    )
    exit_ = entry.copy()
    exit_["best_ask_price"] = [1_110, 1_010]
    result = calculate_short_straddle_trade(entry, exit_, entry_btc=30_000, exit_btc=30_300)
    assert np.isclose(result["short_entry_option_delta"], 0.10)
    assert np.isclose(result["short_hedge_quantity"], 0.10)
    assert np.isclose(result["short_option_pnl"], -240.0)
    assert np.isclose(result["short_hedge_pnl"], 30.0)
    assert result["short_primary_net_return"] < result["short_gross_return"]
    assert result["short_stress_net_return"] < result["short_primary_net_return"]
