from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v175_deribit_skew_receiver_insulator_spread import (
    STRESS_CANDIDATE,
    V175Config,
    build_v175_events,
    receiver_insulator_buckets,
)


def _graph(month: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"]):
        rows.append(
            {
                "graph_month": month,
                "receiver": symbol,
                "sample_n": 2_000,
                "forward_abs_correlation": 0.8 - index * 0.08,
                "direction_advantage": 0.4 - index * 0.04,
                "selected": index < 4,
                "receiver_rank": index + 1 if index < 4 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def test_receiver_and_insulator_buckets_are_disjoint() -> None:
    month = pd.Timestamp("2024-01-01", tz="UTC")
    receivers, insulators = receiver_insulator_buckets(_graph(month), month)
    assert receivers == ["A", "B", "C", "D"]
    assert insulators == ["H", "G", "F", "E"]
    assert set(receivers).isdisjoint(insulators)


def test_stress_spread_shorts_receivers_and_longs_insulators() -> None:
    entry = pd.Timestamp("2024-01-02", tz="UTC")
    exit_time = entry + pd.Timedelta(hours=24)
    columns = list("ABCDEFGH")
    prices = pd.DataFrame(
        [
            [100.0] * 8,
            [90.0, 90.0, 90.0, 90.0, 100.0, 100.0, 100.0, 100.0],
        ],
        index=[entry, exit_time],
        columns=columns,
    )
    signals = pd.DataFrame(
        {"feature_time": [entry], "event_type": ["stress"], "threshold": [1.0]}
    )
    events = build_v175_events(
        signals,
        _graph(pd.Timestamp("2024-01-01", tz="UTC")),
        prices,
        V175Config(primary_cost=0.004),
    )
    assert events.iloc[0]["candidate"] == STRESS_CANDIDATE
    assert events.iloc[0]["gross_return"] == pytest.approx(0.05)
    assert events.iloc[0]["primary_net_return"] == pytest.approx(0.046)
