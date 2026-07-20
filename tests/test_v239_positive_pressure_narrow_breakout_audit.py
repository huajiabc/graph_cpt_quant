from pressure_graph.reports.v239_positive_pressure_narrow_breakout_audit import (
    run_v239_audit,
)


def test_v239_audit_validates_forward_shadow_only_status() -> None:
    checks, diagnostics = run_v239_audit()
    assert checks["passed"].all()
    errors = diagnostics[diagnostics["diagnostic"].str.contains("error")]
    assert errors["value"].max() <= 1e-12
