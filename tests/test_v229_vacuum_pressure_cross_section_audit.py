from pressure_graph.reports.v229_vacuum_pressure_cross_section_audit import (
    run_v229_audit,
)


def test_v229_independent_audit_validates_rejection() -> None:
    checks, errors, gates, _ = run_v229_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
    assert not gates["passed"].all()
    assert not gates.loc[gates["gate"].eq("positive_gross"), "passed"].iloc[0]
