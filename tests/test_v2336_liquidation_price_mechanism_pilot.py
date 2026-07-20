from pressure_graph.reports.v2336_liquidation_price_mechanism_pilot import (
    run_v2336,
)


def test_v2336_retrospective_mechanism_pilot_is_complete_but_not_promotable() -> None:
    result = run_v2336()
    assert result["audit"]["passed"].all(), result["audit"].loc[
        ~result["audit"]["passed"]
    ].to_dict("records")
    assert result["metadata"]["outcomes_loaded"] is True
    assert result["metadata"]["promotion_allowed"] is False
    assert len(result["correlations"]) == 54
