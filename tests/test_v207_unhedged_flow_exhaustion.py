import numpy as np
import pandas as pd

from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    EVENT_WIDE,
)
from pressure_graph.reports.v207_unhedged_flow_exhaustion import (
    V207Config,
    build_v207_events,
)


def test_unhedged_receiver_book_and_btc_control() -> None:
    time = pd.Timestamp("2026-01-01 00:15", tz="UTC")
    features = pd.DataFrame(
        {
            "source_event_id": ["E1"],
            "feature_time": [time],
            "period": ["holdout"],
            "entry_day": [time.date()],
            "entry_month": ["2026-01"],
            "community_id": ["C1"],
            "source_sign": [1.0],
            "candidate": [EVENT_WIDE],
            "candidate_receivers": ["A|B"],
        }
    )
    close = pd.DataFrame(
        {
            "BTCUSDT": [100.0, 99.5],
            "A": [100.0, 99.0],
            "B": [100.0, 98.0],
        },
        index=[time, time + pd.Timedelta(minutes=15)],
    )
    events = build_v207_events(features, close, V207Config())
    assert len(events) == 1
    assert np.isclose(events.loc[0, "gross_notional"], 1.0)
    assert np.isclose(events.loc[0, "receiver_gross_return"], 0.015)
    assert np.isclose(events.loc[0, "btc_control_gross_return"], 0.005)
    assert np.isclose(events.loc[0, "paired_receiver_minus_btc_return"], 0.01)
