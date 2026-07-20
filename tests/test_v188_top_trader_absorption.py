from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v188_top_trader_absorption import (
    BUCKET_CANDIDATE,
    DIRECT_CANDIDATE,
    V188Config,
    build_v188_bucket_events,
    build_v188_direct_events,
    build_v188_features,
)


def test_direct_absorption_event_is_reversal_with_direct_cost() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [100, 101, 100, 99]}, index=index)
    signals = pd.DataFrame(
        {
            "feature_time": [index[1]],
            "source_feature_time": [index[1]],
            "kind": ["unwind"],
            "candidate": ["old"],
            "source_sign": [1.0],
        }
    )
    events = build_v188_direct_events(signals, close, V188Config())
    assert events.loc[0, "candidate"] == DIRECT_CANDIDATE
    assert events.loc[0, "trade_direction"] == -1.0
    expected = -(99 / 101 - 1)
    assert math.isclose(events.loc[0, "gross_return"], expected)
    assert math.isclose(events.loc[0, "primary_net_return"], expected - 0.001)


def test_bucket_requires_price_flow_oi_and_toptrader_absorption() -> None:
    previous = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    source = previous + pd.Timedelta(minutes=15)
    exit_time = source + pd.Timedelta(minutes=30)
    names = [f"A{position}USDT" for position in range(8)]
    index = pd.DatetimeIndex([previous, source, exit_time])
    close = pd.DataFrame(100.0, index=index, columns=["BTCUSDT", *names])
    close.loc[source, names] = 101.0
    close.loc[exit_time, names] = 99.99
    panels = {
        "sum_taker_long_short_vol_ratio": pd.DataFrame(
            1.2, index=index, columns=close.columns
        ),
        "sum_open_interest": pd.DataFrame(
            [[100] * 9, [99] * 9, [99] * 9], index=index, columns=close.columns
        ),
        "sum_toptrader_long_short_ratio": pd.DataFrame(
            [[1.2] * 9, [1.1] * 9, [1.1] * 9], index=index, columns=close.columns
        ),
    }
    features = build_v188_features(close, panels)
    risk = pd.DataFrame(
        {
            "risk_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "receiver": names,
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
            "kind": ["unwind"],
            "candidate": ["old"],
            "source_sign": [1.0],
        }
    )
    events = build_v188_bucket_events(
        signals, risk, close, features, V188Config()
    )
    assert len(events) == 1
    assert events.loc[0, "candidate"] == BUCKET_CANDIDATE
    assert events.loc[0, "receiver_count"] == 8
    assert events.loc[0, "trade_direction"] == -1.0
    assert events.loc[0, "btc_hedge_weight"] == 0.5
