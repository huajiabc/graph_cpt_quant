import numpy as np
import pandas as pd

from pressure_graph.reports.v216_dex_community_relative_spread import (
    beta_neutral_spread_weights,
)


def test_spread_weights_are_dollar_and_beta_neutral_before_and_after_hedge() -> None:
    beta = pd.Series({"L1": 0.8, "L2": 1.0, "H1": 1.1, "H2": 1.3})
    weights = beta_neutral_spread_weights(["L1", "L2"], ["H1", "H2"], beta, 1.0)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert np.isclose(weights["L1"] + weights["L2"] + weights["H1"] + weights["H2"], 0.0)
    residual = weights["BTCUSDT"] + sum(weights[symbol] * beta[symbol] for symbol in beta.index)
    assert np.isclose(residual, 0.0)
    assert weights["L1"] > 0 and weights["H1"] < 0
