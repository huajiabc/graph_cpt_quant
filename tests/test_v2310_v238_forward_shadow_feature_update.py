from pressure_graph.reports.v2310_v238_forward_shadow_feature_update import (
    audit_v2310_feature_update,
    run_v2310_feature_update,
)


def test_v2310_isolated_forward_update_is_complete_and_outcome_free() -> None:
    forward, states, events, metadata = run_v2310_feature_update()
    checks = audit_v2310_feature_update(forward, states, events, metadata)
    assert checks["passed"].all()
    assert metadata["outcomes_loaded"] is False
