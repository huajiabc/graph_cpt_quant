from pressure_graph.reports.v2338_liquidation_filtered_oco_execution_pilot import (
    run_v2338,
)


def test_v2338_execution_pilot_remains_retrospective_and_non_promotable() -> None:
    audit, outcomes, summary, metadata = run_v2338()
    assert len(outcomes) >= 85
    assert len(summary) == 4
    assert metadata["promotion_allowed"] is False
    assert audit.loc[
        audit["check"].eq("retrospective_quartile_not_promotion_evidence"), "passed"
    ].iloc[0]
