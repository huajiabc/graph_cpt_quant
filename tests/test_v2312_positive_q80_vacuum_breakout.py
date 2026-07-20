import pandas as pd

from pressure_graph.reports.v2312_positive_q80_vacuum_breakout import (
    V2312Config,
    decide_v2312,
)


def test_v2312_decision_requires_every_frozen_gate() -> None:
    rows = []
    for scope in ("all", "development", "validation", "holdout"):
        rows.append(
            {
                "scope": scope,
                "triggered_trades": 90 if scope == "all" else 25,
                "ambiguous_trades": 0,
                "mean_primary_net_return_bp": 1.0,
                "mean_stress_net_return_bp": 1.0,
            }
        )
    summary = pd.DataFrame(rows)
    random = pd.DataFrame(
        {
            "matched_random_percentile": [95.0] * 4,
            "unmatched_events": [0] * 4,
        }
    )
    bootstrap = pd.DataFrame({"mean_primary_net_return": [0.001] * 100})
    leave = pd.DataFrame({"mean_primary_net_return_bp": [1.0] * 12})
    decision, verdict = decide_v2312(
        summary, summary, random, bootstrap, leave, V2312Config()
    )
    assert decision["passed"].all()
    assert "supported" in verdict
