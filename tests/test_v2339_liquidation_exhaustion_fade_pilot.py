from pressure_graph.reports.v2339_liquidation_exhaustion_fade_pilot import (
    run_v2339,
)


def test_v2339_fade_pilot_covers_both_horizons_without_promotion() -> None:
    audit, outcomes, summary, metadata = run_v2339()
    assert set(outcomes["horizon_minutes"]) == {60, 240}
    assert len(summary) == 8
    assert metadata["promotion_allowed"] is False
    assert audit.loc[
        audit["check"].eq("retrospective_threshold_not_promotion_evidence"),
        "passed",
    ].iloc[0]
