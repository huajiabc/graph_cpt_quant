import numpy as np
import pandas as pd

from pressure_graph.reports.v200_reference_price_transmission_feature_audit import (
    REFERENCE_LAG,
    V200FeatureConfig,
    _receiver_state,
    causal_rolling_residual_z,
    cross_sectional_residual_z,
)


def test_causal_residual_does_not_change_before_future_perturbation() -> None:
    times = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    x = pd.DataFrame({"A": np.linspace(-0.01, 0.01, len(times))}, index=times)
    y = pd.DataFrame(
        {"A": 0.7 * x["A"].to_numpy() + np.sin(np.arange(len(times))) * 0.001},
        index=times,
    )
    cfg = V200FeatureConfig(lookback_bars=6, minimum_bars=4)
    left = causal_rolling_residual_z(y, x, cfg)
    changed = x.copy()
    changed.iloc[-1, 0] += 10.0
    right = causal_rolling_residual_z(y, changed, cfg)
    pd.testing.assert_series_equal(left.iloc[:-1, 0], right.iloc[:-1, 0])


def test_reference_lag_receiver_score_prefers_underreacting_mark() -> None:
    timestamp = pd.Timestamp("2026-01-01 00:15", tz="UTC")
    columns = ["A", "B"]
    features = {
        "reference_residual_z": pd.DataFrame(
            [[-2.0, 1.0]], index=[timestamp], columns=columns
        ),
        "index_return_z": pd.DataFrame(
            [[1.0, 1.0]], index=[timestamp], columns=columns
        ),
        "premium_innovation_z": pd.DataFrame(
            [[0.1, 0.2]], index=[timestamp], columns=columns
        ),
    }
    state = _receiver_state(
        timestamp,
        columns,
        1.0,
        REFERENCE_LAG,
        features,
        V200FeatureConfig(),
    )
    assert state.loc["A", "score"] == 2.0
    assert state.loc["B", "score"] == -1.0


def test_cross_sectional_residual_is_orthogonal_at_feature_time() -> None:
    timestamp = pd.Timestamp("2026-01-01 00:15", tz="UTC")
    columns = [f"S{index}" for index in range(6)]
    regressor = pd.DataFrame(
        [np.arange(6.0)], index=[timestamp], columns=columns
    )
    dependent = pd.DataFrame(
        [[0.1, 1.4, 1.8, 3.6, 3.9, 5.5]], index=[timestamp], columns=columns
    )
    residual = cross_sectional_residual_z(
        dependent, regressor, minimum_symbols=5
    )
    assert abs(residual.loc[timestamp].corr(regressor.loc[timestamp])) < 1e-12
    assert np.isclose(residual.loc[timestamp].std(ddof=1), 1.0)
