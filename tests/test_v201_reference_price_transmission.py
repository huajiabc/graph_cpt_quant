import numpy as np
import pandas as pd

from pressure_graph.reports.v185_btc_leverage_flow_graph import BTC
from pressure_graph.reports.v201_reference_price_transmission import (
    COMMUNITY_REFERENCE,
    GLOBAL_REFERENCE,
    beta_neutral_weights,
    select_v201_feature_events,
)


def test_beta_neutral_weights_have_unit_gross() -> None:
    beta = pd.Series({"A": 0.8, "B": 1.2, "C": 1.0})
    weights = beta_neutral_weights(["A", "B", "C"], 1.0, beta)
    residual = weights[BTC] + sum(
        weights[symbol] * beta[symbol] for symbol in beta.index
    )
    assert abs(residual) < 1e-12
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert all(weights[symbol] > 0 for symbol in beta.index)
    assert weights[BTC] < 0


def test_candidate_selection_uses_frozen_scope_and_threshold() -> None:
    frame = pd.DataFrame(
        {
            "source_scope": [
                "GLOBAL_BTC_INDEX_SHOCK",
                "COMMUNITY_COHERENT_INDEX_SHOCK",
                "COMMUNITY_COHERENT_INDEX_SHOCK",
            ],
            "family": [
                "REFERENCE_RESIDUAL_INDEX_LEAD_CATCHUP",
                "REFERENCE_RESIDUAL_INDEX_LEAD_CATCHUP",
                "REFERENCE_RESIDUAL_INDEX_LEAD_CATCHUP",
            ],
            "source_setting": ["q90", "z2.0", "z2.0"],
            "receiver_z_threshold": [1.5, 1.0, 1.5],
            "receiver_count": [3, 3, 3],
            "feature_time": pd.date_range(
                "2026-01-01", periods=3, freq="15min", tz="UTC"
            ),
            "community_id": ["GLOBAL", "C1", "C1"],
        }
    )
    assert len(select_v201_feature_events(frame, GLOBAL_REFERENCE)) == 1
    selected = select_v201_feature_events(frame, COMMUNITY_REFERENCE)
    assert len(selected) == 1
    assert selected["receiver_z_threshold"].iloc[0] == 1.0
