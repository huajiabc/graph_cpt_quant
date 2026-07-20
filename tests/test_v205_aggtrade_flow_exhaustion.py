import numpy as np
import pandas as pd

from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    EVENT_WIDE,
)
from pressure_graph.reports.v205_aggtrade_flow_exhaustion import (
    V205Config,
    build_v205_events,
    state_spread_weights,
)


def test_state_spread_weights_are_beta_and_gross_neutralized() -> None:
    beta = pd.Series({"A": 1.2, "B": 0.8, "C": 1.0})
    weights = state_spread_weights(["A"], ["B", "C"], 1.0, beta)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    exposure = weights["BTCUSDT"] + sum(
        weights[symbol] * beta[symbol] for symbol in beta.index
    )
    assert np.isclose(exposure, 0.0)
    assert weights["A"] < 0
    assert weights["B"] > 0
    assert weights["C"] > 0


def test_event_wide_reveal_fades_source_direction() -> None:
    time = pd.Timestamp("2026-01-01 00:15", tz="UTC")
    features = pd.DataFrame(
        {
            "source_event_id": ["E1"],
            "feature_time": [time],
            "period": ["validation"],
            "entry_day": [time.date()],
            "entry_month": ["2026-01"],
            "community_id": ["C1"],
            "source_sign": [1.0],
            "candidate": [EVENT_WIDE],
            "candidate_receiver_count": [3],
            "candidate_receivers": ["A|B|C"],
            "strict_exhausted_receivers": [""],
            "persistent_receivers": [""],
        }
    )
    risk = pd.DataFrame(
        {
            "risk_month": [time.floor("D").replace(day=1)] * 3,
            "receiver": ["A", "B", "C"],
            "btc_beta": [1.0, 1.0, 1.0],
        }
    )
    close = pd.DataFrame(
        {
            "BTCUSDT": [100.0, 100.0],
            "A": [100.0, 99.0],
            "B": [100.0, 99.0],
            "C": [100.0, 99.0],
        },
        index=[time, time + pd.Timedelta(minutes=15)],
    )
    events = build_v205_events(features, risk, close, V205Config())
    assert len(events) == 1
    assert events.loc[0, "gross_return"] > 0
    assert np.isclose(events.loc[0, "gross_notional"], 1.0)
    assert np.isclose(events.loc[0, "residual_btc_beta"], 0.0)
