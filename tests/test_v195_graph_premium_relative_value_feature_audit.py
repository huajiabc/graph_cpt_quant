import numpy as np
import pandas as pd

from pressure_graph.reports.v195_graph_premium_relative_value_feature_audit import (
    V195FeatureConfig,
    _funding_orthogonal_residual,
    _neutralize_weights,
    build_v195_premium_z,
)


def test_v195_premium_z_scale_is_shifted() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    values = pd.DataFrame(
        {"BTCUSDT": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]}, index=index
    )
    close = pd.DataFrame({"BTCUSDT": np.arange(100.0, 108.0)}, index=index)
    cfg = V195FeatureConfig(lookback_bars=4, minimum_bars=3)
    original = build_v195_premium_z(values, close, cfg)
    mutated = values.copy()
    mutated.loc[index[-1], "BTCUSDT"] = 100.0
    changed = build_v195_premium_z(mutated, close, cfg)
    assert original.loc[index[-2], "BTCUSDT"] == changed.loc[index[-2], "BTCUSDT"]
    prior = values.loc[index[-5:-1], "BTCUSDT"]
    expected = (values.loc[index[-1], "BTCUSDT"] - prior.mean()) / prior.std(ddof=1)
    assert np.isclose(original.loc[index[-1], "BTCUSDT"], expected)


def test_v195_funding_residual_is_orthogonal() -> None:
    funding = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    peer = pd.Series([1.0, 0.0, 2.0, 4.0, 3.0])
    residual = _funding_orthogonal_residual(peer, funding)
    assert abs(residual.corr(funding)) <= 1e-12
    assert abs(residual.mean()) <= 1e-12


def test_v195_weights_are_beta_neutral_and_gross_one() -> None:
    raw = {"A": 0.5, "B": -0.5}
    beta = pd.Series({"A": 1.5, "B": 0.5})
    weights = _neutralize_weights(raw, beta)
    residual_beta = weights["A"] * 1.5 + weights["B"] * 0.5 + weights["BTCUSDT"]
    assert abs(residual_beta) <= 1e-12
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
