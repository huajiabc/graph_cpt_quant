import pandas as pd
import pytest

from pressure_graph.reports.v147_funding_sign_spread import (
    BTC,
    V147Config,
    beta_neutral_components,
    sign_distributions,
)


def _local() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["N1", "N2", "N3", "N4", "P1", "P2", "P3", "P4"],
            "score_7d": [-1, -2, -3, -4, 1, 2, 3, 4],
            "price_return": [0.01] * 8,
            "future_funding": [-0.001] * 4 + [0.001] * 4,
            "btc_beta": [1.0] * 8,
            "btc_return": [0.02] * 8,
            "btc_future_funding": [0.0001] * 8,
        }
    )


def test_v147_sign_distributions_are_equal_weight_without_rank() -> None:
    spread, positive_short, negative_n, positive_n = sign_distributions(
        _local(), V147Config()
    )
    assert negative_n == positive_n == 4
    assert set(spread.values()) == {0.125, -0.125}
    assert set(positive_short.values()) == {-0.25}


def test_v147_components_are_gross_one_and_beta_neutral() -> None:
    spread, _, _, _ = sign_distributions(_local(), V147Config())
    weights, components = beta_neutral_components(_local(), spread)
    assert sum(abs(value) for value in weights.values()) == pytest.approx(1.0)
    assert components["residual_btc_beta"] == pytest.approx(0.0, abs=1e-15)
    assert components["funding_return"] > 0
    assert weights[BTC] == pytest.approx(0.0, abs=1e-15)


def test_v147_positive_short_adds_long_btc_hedge() -> None:
    _, positive_short, _, _ = sign_distributions(_local(), V147Config())
    weights, components = beta_neutral_components(_local(), positive_short)
    assert weights[BTC] > 0
    assert components["residual_btc_beta"] == pytest.approx(0.0, abs=1e-15)
