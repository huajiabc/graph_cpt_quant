import numpy as np

from pressure_graph.reports.v228_vacuum_pressure_cross_section_spread import (
    V228Config,
    build_v228_outcomes,
    estimate_v228_monthly_betas,
    load_v228_inputs,
)


def test_v228_weights_are_exactly_gross_one_and_beta_neutral() -> None:
    features, prices = load_v228_inputs()
    betas = estimate_v228_monthly_betas(features, prices)
    outcomes, weights = build_v228_outcomes(features, prices, betas)
    assert len(outcomes) >= 150
    gross = weights.groupby("entry_time")["weight"].apply(lambda x: x.abs().sum())
    beta = weights.assign(component=weights["weight"] * weights["btc_beta"]).groupby(
        "entry_time"
    )["component"].sum()
    assert np.allclose(gross, 1.0, atol=1e-12)
    assert np.allclose(beta, 0.0, atol=1e-12)
    assert np.allclose(
        outcomes["primary_net_return_4h"],
        outcomes["gross_return_4h"] - V228Config().primary_cost,
    )
