from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import UNWIND
from pressure_graph.reports.v187_unwind_volatility_transfer_bucket import (
    CANDIDATES,
    V187Config,
    build_v187_events,
    build_v187_monthly_risk,
)


def test_monthly_risk_uses_prior_month_data() -> None:
    index = pd.date_range("2025-12-01", periods=2_100, freq="15min", tz="UTC")
    btc = pd.Series([position / 1_000_000 for position in range(len(index))], index=index)
    alt = 2 * btc + pd.Series(
        [((position % 7) - 3) / 1_000_000 for position in range(len(index))],
        index=index,
    )
    returns = pd.DataFrame({"BTCUSDT": btc, "ALTUSDT": alt})
    risk = build_v187_monthly_risk(
        returns,
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
        V187Config(risk_min_samples=2_000),
    )
    assert len(risk) == 1
    assert risk.loc[0, "samples"] == 2_004
    assert risk.loc[0, "risk_month"] == pd.Timestamp("2026-01-01", tz="UTC")


def test_event_builds_both_beta_hedged_reversal_buckets() -> None:
    previous = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    source = previous + pd.Timedelta(minutes=15)
    exit_time = source + pd.Timedelta(minutes=30)
    receivers = [f"A{position}USDT" for position in range(8)]
    index = pd.DatetimeIndex([previous, source, exit_time])
    close = pd.DataFrame(100.0, index=index, columns=["BTCUSDT", *receivers])
    close.loc[source, receivers] = [101 + position / 10 for position in range(8)]
    close.loc[exit_time, receivers] = close.loc[source, receivers] * 0.99
    returns = close.pct_change(fill_method=None)
    flow = pd.DataFrame(0.2, index=index, columns=close.columns)
    oi_change = pd.DataFrame(-0.01, index=index, columns=close.columns)
    risk = pd.DataFrame(
        {
            "risk_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "receiver": receivers,
            "samples": 2_100,
            "btc_beta": 0.5,
            "residual_volatility": 0.001,
            "return_volatility": 0.002,
        }
    )
    signals = pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "kind": [UNWIND],
            "candidate": ["old"],
            "source_sign": [1.0],
        }
    )
    events = build_v187_events(
        signals,
        risk,
        close,
        returns,
        flow,
        oi_change,
        V187Config(min_receiver_bucket=5),
    )
    assert set(events["candidate"]) == set(CANDIDATES)
    assert events["receiver_count"].eq(8).all()
    assert events["trade_direction"].eq(-1.0).all()
    assert events["btc_hedge_weight"].eq(0.5).all()
    assert all(math.isclose(value, 0.01 / 1.5) for value in events["gross_return"])
