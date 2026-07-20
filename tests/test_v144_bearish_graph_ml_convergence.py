import numpy as np
import pandas as pd

from pressure_graph.reports.v144_bearish_graph_ml_convergence import (
    FEATURE_COLUMNS,
    V144Config,
    build_v144_model_portfolio,
    permute_training_labels,
)


def test_v144_label_permutation_preserves_each_month_multiset() -> None:
    frame = pd.DataFrame(
        {
            "entry_month": ["2025-08"] * 4 + ["2025-09"] * 4,
            "residual_gross_12h": np.arange(8, dtype=float),
        }
    )
    shuffled = permute_training_labels(frame, 0, V144Config())
    assert sorted(shuffled[:4]) == [0.0, 1.0, 2.0, 3.0]
    assert sorted(shuffled[4:]) == [4.0, 5.0, 6.0, 7.0]


def test_v144_portfolio_applies_hurdle_and_twelve_hour_cooldown() -> None:
    times = pd.to_datetime(
        ["2026-01-01 00:00Z", "2026-01-01 06:00Z", "2026-01-01 12:00Z"],
        utc=True,
    )
    predictions = pd.DataFrame(
        {
            "feature_time": times,
            "entry_day": ["2026-01-01"] * 3,
            "entry_month": ["2026-01"] * 3,
            "period": ["validation"] * 3,
            "source_community": ["A"] * 3,
            "receiver_community": ["B"] * 3,
            "predicted_residual_gross": [0.005, 0.010, 0.006],
            "training_rows": [300] * 3,
            "raw_gross_12h": [0.01] * 3,
            "residual_gross_12h": [0.01] * 3,
        }
    )
    portfolio = build_v144_model_portfolio(predictions, V144Config())
    assert portfolio["feature_time"].tolist() == [times[0], times[2]]
    assert portfolio["residual_net_12h_40bp"].eq(0.006).all()


def test_v144_feature_contract_is_frozen() -> None:
    assert len(FEATURE_COLUMNS) == 10
    assert "edge_weight" in FEATURE_COLUMNS
    assert "return_z_gap" in FEATURE_COLUMNS
