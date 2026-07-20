from pressure_graph.reports.v226_alt_book_vacuum_pressure_audit import run_v226_audit


def test_v226_independent_audit_validates_rejection() -> None:
    checks, errors, gates, _ = run_v226_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
    assert not gates["passed"].all()
    assert not gates.loc[gates["gate"].eq("positive_primary"), "passed"].iloc[0]
