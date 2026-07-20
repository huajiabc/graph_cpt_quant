from pressure_graph.reports.v230_book_vacuum_implied_variance_feature_audit import (
    V230Config,
    audit_v230_features,
    build_v230_event_features,
    load_v230_inputs,
    summarize_v230,
)


def test_v230_features_are_causal_and_cover_all_three_periods() -> None:
    cfg = V230Config()
    events, surface, prices = load_v230_inputs(cfg)
    features = build_v230_event_features(events, surface, prices, cfg)
    summary = summarize_v230(features)
    checks = audit_v230_features(events, surface, features, summary, cfg)
    assert checks["passed"].all()
    assert set(features["period"]) == {"development", "validation", "holdout"}
    assert features["surface_feature_time"].le(features["entry_time"]).all()
