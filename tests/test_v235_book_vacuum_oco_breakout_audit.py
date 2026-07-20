from pressure_graph.reports.v235_book_vacuum_oco_breakout_audit import run_v235_audit


def test_v235_independent_audit_validates_rejection() -> None:
    checks, errors, _ = run_v235_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
