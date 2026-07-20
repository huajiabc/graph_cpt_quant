import numpy as np
import pandas as pd

from pressure_graph.reports.v218_spot_perp_flow_inventory_feature_audit import (
    COMMUNITY_SPREAD,
    GLOBAL_SPREAD,
    V218FeatureConfig,
    build_v218_candidate_features,
    build_v218_flow_features,
)


def test_flow_normalization_uses_strictly_prior_history() -> None:
    index = pd.date_range("2025-08-01", periods=12, freq="1h", tz="UTC")
    quote = pd.DataFrame({"A": 100.0}, index=index)
    imbalance = np.linspace(-0.2, 0.3, len(index))
    taker = quote.mul((imbalance + 1.0) / 2.0, axis=0)
    cfg = V218FeatureConfig(
        normalization_lookback_hours=5,
        normalization_minimum_hours=4,
        activity_lookback_hours=5,
        activity_minimum_hours=4,
    )
    features = build_v218_flow_features(quote, taker, cfg)
    timestamp = index[-1]
    history = pd.Series(imbalance, index=index).iloc[-6:-1]
    expected = (imbalance[-1] - history.mean()) / history.std(ddof=1)
    assert np.isclose(features["zscore"].at[timestamp, "A"], expected)


def test_global_and_community_candidate_rules() -> None:
    timestamp = pd.Timestamp("2025-09-01 00:00", tz="UTC")
    rows = []
    scores = [-3.0, -2.5, -2.0, -1.5, -1.2, 1.2, 1.5, 2.0, 2.5, 3.0]
    for index, score in enumerate(scores):
        rows.append(
            {
                "feature_time": timestamp,
                "entry_time": timestamp + pd.Timedelta(hours=1),
                "period": "development",
                "entry_day": timestamp.floor("D"),
                "entry_month": "2025-09",
                "community_id": f"C{index // 2}",
                "symbol": f"S{index}",
                "feature_eligible": True,
                "spot_minus_perp_flow_z": score,
            }
        )
    cfg = V218FeatureConfig(
        global_bucket_size=5,
        global_minimum_leg=5,
        community_minimum_cross_section=2,
        community_minimum_pairs=2,
    )
    candidates = build_v218_candidate_features(pd.DataFrame(rows), cfg)
    assert GLOBAL_SPREAD in set(candidates["candidate"])
    assert COMMUNITY_SPREAD not in set(candidates["candidate"])
    global_row = candidates[candidates["candidate"].eq(GLOBAL_SPREAD)].iloc[0]
    assert global_row["long_count"] == 5
    assert global_row["short_count"] == 5
