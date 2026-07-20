import numpy as np
import pandas as pd

from pressure_graph.reports.v106_directed_residual_bucket import (
    V106Config,
    build_v106_month_edges,
    estimate_v106_betas,
    residualize_v106_returns,
)


def test_v106_residualization_removes_btc_beta() -> None:
    btc = pd.Series(np.linspace(-0.01, 0.01, 200))
    frame = pd.DataFrame({"BTCUSDT": btc, "A": 2.0 * btc + 0.001})
    betas = estimate_v106_betas(frame)
    residual = residualize_v106_returns(frame, betas)
    assert abs(float(betas["A"]) - 2.0) < 1e-9
    assert float(residual["A"].std()) < 1e-12


def test_v106_edges_recover_planted_direction() -> None:
    rng = np.random.default_rng(7)
    leader = rng.normal(0, 1, 1500)
    follower = np.r_[0.0, leader[:-1]] + rng.normal(0, 0.05, 1500)
    noise_a = rng.normal(0, 1, 1500)
    noise_b = rng.normal(0, 1, 1500)
    residual = pd.DataFrame(
        {"LEADER": leader, "FOLLOWER": follower, "N1": noise_a, "N2": noise_b}
    )
    cfg = V106Config(min_edge_samples=1000, leaders_per_follower=3)
    edges = build_v106_month_edges(
        residual, pd.Timestamp("2026-01-01", tz="UTC"), cfg
    )
    planted = edges[
        edges["leader_symbol"].eq("LEADER")
        & edges["follower_symbol"].eq("FOLLOWER")
    ]
    assert len(planted) == 1
    assert int(planted.iloc[0]["lag_bars"]) == 1
    assert float(planted.iloc[0]["direction_advantage"]) > 0.9
