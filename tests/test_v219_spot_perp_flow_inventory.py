import numpy as np
import pandas as pd

from pressure_graph.reports.v219_spot_perp_flow_inventory import (
    beta_neutral_spread_weights,
)


def test_spread_weights_are_unit_gross_dollar_and_beta_neutral() -> None:
    beta = pd.Series({"L1": 0.8, "L2": 1.0, "S1": 1.1, "S2": 1.3})
    weights = beta_neutral_spread_weights(["L1", "L2"], ["S1", "S2"], beta)
    assert np.isclose(sum(abs(value) for value in weights.values()), 1.0)
    assert np.isclose(
        weights["L1"] + weights["L2"] + weights["S1"] + weights["S2"],
        0.0,
    )
    residual = weights["BTCUSDT"] + sum(weights[symbol] * beta[symbol] for symbol in beta.index)
    assert np.isclose(residual, 0.0)
    assert weights["L1"] > 0 and weights["S1"] < 0
