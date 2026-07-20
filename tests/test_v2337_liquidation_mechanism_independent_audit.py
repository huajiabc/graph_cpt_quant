from pressure_graph.reports.v2337_liquidation_mechanism_independent_audit import (
    run_v2337,
)


def test_v2337_classifies_liquidations_as_regime_marker_not_standalone_alpha() -> None:
    audit, metrics, metadata = run_v2337()
    assert audit["passed"].all(), audit.loc[~audit["passed"]].to_dict("records")
    assert len(metrics) == 11
    assert metadata["candidate_role"] == "volatility_regime_filter_or_oco_overlay"
    assert metadata["promotion_allowed"] is False
