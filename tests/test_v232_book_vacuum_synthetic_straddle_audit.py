from pressure_graph.reports.v232_book_vacuum_synthetic_straddle_audit import (
    run_v232_audit,
)


def test_v232_independent_audit_validates_rejection() -> None:
    checks, errors, _ = run_v232_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
