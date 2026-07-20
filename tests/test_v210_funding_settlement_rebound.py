import numpy as np
import pandas as pd

from pressure_graph.reports.v209_funding_settlement_feature_audit import (
    ALL_NEGATIVE,
)
from pressure_graph.reports.v210_funding_settlement_rebound import (
    V210Config,
    beta_neutral_long_weights,
    build_v210_events,
)


def test_beta_neutral_long_weights() -> None:
    beta = pd.Series({"A": 1.2, "B": 0.8})
    weights = beta_neutral_long_weights(["A", "B"], beta)
    assert np.isclose(sum(abs(weight) for weight in weights.values()), 1.0)
    assert np.isclose(
        weights["BTCUSDT"] + weights["A"] * 1.2 + weights["B"] * 0.8,
        0.0,
    )


def test_event_builder_uses_delayed_post_settlement_window() -> None:
    settlement = pd.Timestamp("2025-08-01 00:00", tz="UTC")
    entry = settlement + pd.Timedelta(minutes=15)
    exit_time = entry + pd.Timedelta(minutes=60)
    features = pd.DataFrame(
        {
            "settlement_time": [settlement],
            "entry_time": [entry],
            "exit_time": [exit_time],
            "period": ["development"],
            "entry_day": [entry.date()],
            "entry_month": ["2025-08"],
            "candidate": [ALL_NEGATIVE],
            "selection_count": [2],
            "selection_symbols": ["A|B"],
        }
    )
    risk = pd.DataFrame(
        {
            "risk_month": [pd.Timestamp("2025-08-01", tz="UTC")] * 2,
            "receiver": ["A", "B"],
            "btc_beta": [1.0, 1.0],
        }
    )
    close = pd.DataFrame(
        {
            "BTCUSDT": [100.0, 100.0],
            "A": [100.0, 102.0],
            "B": [100.0, 102.0],
        },
        index=[entry, exit_time],
    )
    events = build_v210_events(features, risk, close, V210Config())
    assert len(events) == 1
    assert events.loc[0, "gross_return"] > 0
    assert events.loc[0, "entry_time"] == entry
    assert np.isclose(events.loc[0, "residual_btc_beta"], 0.0)
