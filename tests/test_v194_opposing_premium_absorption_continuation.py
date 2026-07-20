import numpy as np
import pandas as pd

from pressure_graph.reports.v194_opposing_premium_absorption_continuation import (
    BUCKET_CANDIDATE,
    DIRECT_CANDIDATE,
    V194Config,
    _receiver_scores,
    build_v194_bucket_events,
    build_v194_direct_events,
)


def _signals(timestamp: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_time": [timestamp],
            "source_feature_time": [timestamp],
            "source_sign": [1.0],
            "source_side": ["up_move"],
            "source_shape": ["absorption"],
        }
    )


def test_v194_direct_continues_source_direction() -> None:
    index = pd.date_range("2026-02-01", periods=4, freq="15min", tz="UTC")
    close = pd.DataFrame({"BTCUSDT": [100.0, 100.5, 101.0, 101.5]}, index=index)
    events = build_v194_direct_events(_signals(index[0]), close)
    row = events.iloc[0]
    assert row["candidate"] == DIRECT_CANDIDATE
    assert row["trade_direction"] == 1.0
    assert np.isclose(row["gross_return"], 0.01)
    assert np.isclose(row["primary_net_return"], 0.009)


def test_v194_bucket_ranks_largest_absorption_ranges() -> None:
    timestamp = pd.Timestamp("2026-02-01 00:00:00", tz="UTC")
    index = pd.date_range(timestamp, periods=4, freq="15min", tz="UTC")
    names = [f"A{i}USDT" for i in range(10)]
    columns = ["BTCUSDT", *names]
    close = pd.DataFrame(
        np.asarray(
            [
                np.repeat(100.0, len(columns)),
                np.repeat(100.2, len(columns)),
                np.asarray([101.0, *np.linspace(101.0, 102.0, 10)]),
                np.repeat(102.0, len(columns)),
            ]
        ),
        index=index,
        columns=columns,
    )
    risk = pd.DataFrame(
        {
            "risk_month": pd.Timestamp("2026-02-01", tz="UTC"),
            "receiver": names,
            "btc_beta": np.repeat(1.0, 10),
        }
    )
    features = {
        "range_z": pd.DataFrame(
            [[0.0, *np.arange(1.0, 11.0)]], index=[timestamp], columns=columns
        ),
        "body_z": pd.DataFrame(
            [[0.0, *np.repeat(-1.0, 10)]], index=[timestamp], columns=columns
        ),
        "close_location": pd.DataFrame(
            [[0.0, *np.repeat(-1.0, 10)]], index=[timestamp], columns=columns
        ),
    }
    cfg = V194Config(risk_min_samples=1)
    scores = _receiver_scores(timestamp, 1.0, risk, features, cfg)
    assert len(scores) == 10
    events = build_v194_bucket_events(
        _signals(timestamp), risk, close, features, cfg
    )
    row = events.iloc[0]
    assert row["candidate"] == BUCKET_CANDIDATE
    assert row["receivers"].split("|") == names[2:][::-1]
    assert row["receiver_count"] == 8
