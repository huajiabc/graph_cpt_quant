from pressure_graph.reports.v2316_positive_q85_vacuum_breakout_audit import (
    run_v2316_audit,
)


def test_v2316_audit_validates_q85_rejection() -> None:
    checks, diagnostics = run_v2316_audit()
    assert checks["passed"].all()
    errors = diagnostics[diagnostics["diagnostic"].str.contains("error")]
    assert errors["value"].max() <= 1e-12
