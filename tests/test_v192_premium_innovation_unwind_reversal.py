from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v192_premium_innovation_unwind_reversal import (
    BUCKET_CANDIDATE,
    DIRECT_CANDIDATE,
    V192Config,
    build_v192_bucket_events,
    build_v192_direct_events,
)


def test_direct_candidate_reverses_premium_confirmed_source() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [100, 99, 100, 101]}, index=index)
    signals = pd.DataFrame(
        {
            "feature_time": [index[1]],
            "source_feature_time": [index[1]],
            "kind": ["unwind"],
            "candidate": ["old"],
            "source_sign": [-1.0],
            "source_side": ["long_liquidation"],
        }
    )
    events = build_v192_direct_events(signals, close, V192Config())
    assert events.loc[0, "candidate"] == DIRECT_CANDIDATE
    assert events.loc[0, "trade_direction"] == 1.0
    expected = 101 / 99 - 1
    assert math.isclose(events.loc[0, "gross_return"], expected)
    assert math.isclose(events.loc[0, "primary_net_return"], expected - 0.001)


def test_bucket_selects_largest_aligned_premium_receivers() -> None:
    source = pd.Timestamp("2026-01-15 00:15", tz="UTC")
    exit_time = source + pd.Timedelta(minutes=30)
    names = [f"A{position}USDT" for position in range(10)]
    close = pd.DataFrame(
        100.0,
        index=pd.DatetimeIndex([source, exit_time]),
        columns=["BTCUSDT", *names],
    )
    close.loc[exit_time, names] = 99.0
    innovation_z = pd.DataFrame(
        [[0.0, *[-(position + 1) for position in range(10)]]],
        index=pd.DatetimeIndex([source]),
        columns=["BTCUSDT", *names],
    )
    risk = pd.DataFrame(
        {
            "risk_month": pd.Timestamp("2026-01-01", tz="UTC"),
            "receiver": names,
            "btc_beta": 0.5,
        }
    )
    signals = pd.DataFrame(
        {
            "feature_time": [source],
            "source_feature_time": [source],
            "kind": ["unwind"],
            "candidate": ["old"],
            "source_sign": [-1.0],
            "source_side": ["long_liquidation"],
        }
    )
    events = build_v192_bucket_events(
        signals, risk, close, innovation_z, V192Config()
    )
    assert len(events) == 1
    assert events.loc[0, "candidate"] == BUCKET_CANDIDATE
    assert events.loc[0, "receiver_count"] == 8
    assert set(events.loc[0, "receivers"].split("|")) == set(names[2:])
    assert events.loc[0, "trade_direction"] == 1.0
