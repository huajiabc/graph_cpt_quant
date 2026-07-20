import pandas as pd

from pressure_graph.reports.v238_positive_pressure_narrow_breakout_robustness import (
    classify_v238,
)


def test_v238_classifies_structural_pass_without_absolute_ci_as_forward_shadow() -> None:
    gates = pd.DataFrame(
        {
            "gate": [
                "structural_one",
                "absolute_month_bootstrap_lower_above_zero",
            ],
            "passed": [True, False],
        }
    )
    assert classify_v238(gates) == (
        "forward_shadow_candidate_not_statistically_confirmed"
    )
