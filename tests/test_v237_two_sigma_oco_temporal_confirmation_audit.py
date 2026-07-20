from pressure_graph.reports.v237_two_sigma_oco_temporal_confirmation_audit import (
    run_v237_audit,
)


def test_v237_audit_validates_temporal_rejection() -> None:
    checks, errors, _ = run_v237_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
