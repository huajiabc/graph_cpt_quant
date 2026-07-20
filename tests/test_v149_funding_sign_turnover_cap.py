import numpy as np
import pandas as pd

from pressure_graph.reports.v149_funding_sign_turnover_cap import (
    _execute_capped_array_transition,
    execute_capped_transition,
    weight_turnover,
)


def _local() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "btc_beta": [1.2, 0.6],
            "price_return": [0.03, -0.01],
            "future_funding": [-0.002, 0.001],
            "btc_return": [0.01, 0.01],
            "btc_future_funding": [0.0002, 0.0002],
        }
    )


def test_v149_uses_target_when_transition_is_below_cap() -> None:
    target = {"A": 0.2, "B": -0.3, "BTCUSDT": -0.1}
    weights, _, fraction, turnover, breach = execute_capped_transition(
        _local(), target, target, 0.70
    )
    assert weights == target
    assert fraction == 1.0
    assert turnover == 0.0
    assert breach == 0.0


def test_v149_caps_transition_and_preserves_constraints() -> None:
    previous = {"A": 0.4, "B": -0.2, "BTCUSDT": -0.2}
    target = {"A": -0.25, "B": 0.35, "BTCUSDT": 0.15}
    weights, components, fraction, turnover, breach = execute_capped_transition(
        _local(), previous, target, 0.30
    )
    assert 0.0 < fraction < 1.0
    assert turnover <= 0.30 + 1e-12
    assert breach <= 1e-12
    assert np.isclose(components["gross_notional"], 1.0, atol=1e-12)
    assert np.isclose(components["residual_btc_beta"], 0.0, atol=1e-12)
    assert np.isclose(weight_turnover(previous, weights), turnover, atol=1e-12)


def test_v149_array_execution_matches_dictionary_execution() -> None:
    previous = {"A": 0.4, "B": -0.2, "BTCUSDT": -0.2}
    target = {"A": -0.25, "B": 0.35, "BTCUSDT": 0.15}
    weights, _, _, turnover, _ = execute_capped_transition(
        _local(), previous, target, 0.30
    )
    alt, btc, array_turnover = _execute_capped_array_transition(
        np.asarray([previous["A"], previous["B"]]),
        previous["BTCUSDT"],
        np.asarray([target["A"], target["B"]]),
        target["BTCUSDT"],
        np.asarray([1.2, 0.6]),
        np.asarray([True, True]),
        0.30,
        48,
    )
    assert np.allclose(alt, [weights["A"], weights["B"]], atol=1e-12)
    assert np.isclose(btc, weights["BTCUSDT"], atol=1e-12)
    assert np.isclose(array_turnover, turnover, atol=1e-12)
