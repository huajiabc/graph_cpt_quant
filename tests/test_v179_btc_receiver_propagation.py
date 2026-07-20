from __future__ import annotations

import math

import pandas as pd

from pressure_graph.reports.v179_btc_receiver_propagation import (
    NEUTRAL_CANDIDATE,
    RAW_CANDIDATE,
    SPREAD_CANDIDATE,
    V179Config,
    assign_v179_buckets,
    build_v179_events,
)


def _graph() -> pd.DataFrame:
    month = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for index in range(20):
        signed = 0.2 - index * 0.02
        rows.append(
            {
                "graph_month": month,
                "receiver": f"A{index}USDT",
                "btc_beta": 0.5 if index < 8 else 0.2,
                "signed_forward_correlation": signed,
                "receiver_score": signed,
            }
        )
    return pd.DataFrame(rows)


def test_bucket_assignment_is_ranked_disjoint_and_causal_month() -> None:
    assignments = assign_v179_buckets(_graph())
    receivers = assignments[assignments["bucket"].eq("receiver")]
    insulators = assignments[assignments["bucket"].eq("insulator")]
    assert len(receivers) == 8
    assert len(insulators) == 8
    assert receivers["signed_forward_correlation"].gt(0).all()
    assert set(receivers["receiver"]).isdisjoint(insulators["receiver"])
    assert assignments["graph_month"].eq(pd.Timestamp("2026-01-01", tz="UTC")).all()


def test_event_candidate_formulas() -> None:
    cfg = V179Config(min_bucket_size=5)
    assignments = assign_v179_buckets(_graph(), cfg)
    entry = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    exit_time = entry + pd.Timedelta(minutes=15)
    receivers = assignments.loc[
        assignments["bucket"].eq("receiver"), "receiver"
    ].tolist()
    insulators = assignments.loc[
        assignments["bucket"].eq("insulator"), "receiver"
    ].tolist()
    close = pd.DataFrame(
        100.0,
        index=pd.DatetimeIndex([entry, exit_time]),
        columns=["BTCUSDT", *receivers, *insulators],
    )
    close.loc[exit_time, "BTCUSDT"] = 101.0
    close.loc[exit_time, receivers] = 102.0
    close.loc[exit_time, insulators] = 100.5
    signals = pd.DataFrame({"feature_time": [entry], "direction": [1.0]})
    events = build_v179_events(signals, assignments, close, cfg).set_index("candidate")

    assert math.isclose(events.loc[RAW_CANDIDATE, "gross_return"], 0.02)
    assert math.isclose(events.loc[NEUTRAL_CANDIDATE, "gross_return"], 0.01)
    expected_spread = (0.5 * (0.02 - 0.005) - 0.15 * 0.01) / 1.15
    assert math.isclose(
        events.loc[SPREAD_CANDIDATE, "gross_return"], expected_spread
    )


def test_rank_reversal_swaps_receiver_and_insulator_returns() -> None:
    cfg = V179Config(min_bucket_size=5)
    assignments = assign_v179_buckets(_graph(), cfg)
    entry = pd.Timestamp("2026-01-15 00:00", tz="UTC")
    exit_time = entry + pd.Timedelta(minutes=15)
    names = assignments["receiver"].tolist()
    close = pd.DataFrame(
        100.0,
        index=pd.DatetimeIndex([entry, exit_time]),
        columns=["BTCUSDT", *names],
    )
    close.loc[exit_time, "BTCUSDT"] = 100.0
    receiver_names = assignments.loc[
        assignments["bucket"].eq("receiver"), "receiver"
    ]
    close.loc[exit_time, receiver_names] = 102.0
    signals = pd.DataFrame({"feature_time": [entry], "direction": [1.0]})
    normal = build_v179_events(signals, assignments, close, cfg)
    reversed_events = build_v179_events(
        signals, assignments, close, cfg, reverse_buckets=True
    )
    normal_raw = normal.loc[normal["candidate"].eq(RAW_CANDIDATE), "gross_return"].iloc[0]
    reversed_raw = reversed_events.loc[
        reversed_events["candidate"].eq(RAW_CANDIDATE), "gross_return"
    ].iloc[0]
    assert math.isclose(normal_raw, 0.02)
    assert math.isclose(reversed_raw, 0.0)
