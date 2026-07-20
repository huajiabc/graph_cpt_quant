from pressure_graph.reports.v2311_positive_q80_vacuum_breakout_feature_audit import (
    audit_v2311_features,
    build_v2311_features,
    summarize_v2311,
)


def test_v2311_positive_q80_features_are_outcome_free_and_well_covered() -> None:
    _, states, features = build_v2311_features()
    summary = summarize_v2311(features)
    checks = audit_v2311_features(states, features, summary)
    assert checks["passed"].all()
    assert len(features) >= 80
