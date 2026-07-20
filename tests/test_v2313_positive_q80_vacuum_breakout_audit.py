from pressure_graph.reports.v2313_positive_q80_vacuum_breakout_audit import (
    run_v2313_audit,
)


def test_v2313_audit_validates_q80_rejection() -> None:
    checks, diagnostics = run_v2313_audit()
    assert checks["passed"].all()
    errors = diagnostics[diagnostics["diagnostic"].str.contains("error")]
    assert errors["value"].max() <= 1e-12
