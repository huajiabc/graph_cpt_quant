import pandas as pd

from pressure_graph.reports.v236_two_sigma_oco_temporal_confirmation import (
    V236Config,
    decide_v236,
)


def test_v236_decision_requires_holdout_and_random_gates() -> None:
    summary = pd.DataFrame(
        [
            {
                "scope": scope,
                "triggered_trades": 30,
                "mean_primary_net_return_per_event_bp": 1.0,
                "ambiguous_trade_fraction": 0.0,
                "mean_reversed_primary_net_return_per_event_bp": -1.0,
            }
            for scope in ("all", "development", "validation", "holdout")
        ]
    )
    random = pd.DataFrame(
        {"event_mean_primary_net": [0.001], "control_mean_primary_net": [-0.001]}
    )
    bootstrap = pd.DataFrame({"mean_primary_net_return": [0.001] * 100})
    decision, verdict = decide_v236(
        summary, random, random, bootstrap, V236Config()
    )
    assert decision["passed"].all()
    assert "supported" in verdict
