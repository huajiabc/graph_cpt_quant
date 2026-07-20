from __future__ import annotations

import numpy as np

from pressure_graph.reports.v153_crowded_source_exhaustion_reversal import (
    source_beta_neutral_weights,
)


def test_v153_one_sided_source_book_is_beta_neutral_and_gross_one() -> None:
    weights = source_beta_neutral_weights(["A", "B"], [], {"A": 1.2, "B": 0.8})
    beta = weights["A"] * 1.2 + weights["B"] * 0.8 + weights["BTCUSDT"]
    assert np.isclose(beta, 0.0, atol=1e-12)
    assert np.isclose(sum(abs(weight) for weight in weights.values()), 1.0)


def test_v153_reversed_control_negates_alt_and_hedge_weights() -> None:
    betas = {"A": 1.2, "B": 0.8}
    direct = source_beta_neutral_weights(["A"], ["B"], betas)
    reversed_weights = source_beta_neutral_weights(
        ["A"], ["B"], betas, reverse=True
    )
    assert set(direct) == set(reversed_weights)
    for symbol in direct:
        assert np.isclose(direct[symbol], -reversed_weights[symbol])
