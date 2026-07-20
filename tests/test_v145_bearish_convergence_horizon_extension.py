import pytest

from pressure_graph.reports.v145_bearish_convergence_horizon_extension import (
    CANDIDATES,
    HORIZONS,
    bearish_convergence_return,
)


def test_v145_spread_is_long_source_and_short_receiver() -> None:
    assert bearish_convergence_return(-0.02, -0.05) == pytest.approx(0.015)
    assert bearish_convergence_return(0.01, 0.03) == pytest.approx(-0.01)


def test_v145_horizon_family_is_frozen() -> None:
    assert HORIZONS == (18, 24)
    assert CANDIDATES[0].endswith("18H")
    assert CANDIDATES[1].endswith("24H")
