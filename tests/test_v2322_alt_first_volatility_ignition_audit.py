from pressure_graph.reports.v2322_alt_first_volatility_ignition_audit import (
    run_v2322_audit,
)


def test_v2322_independent_audit_passes() -> None:
    checks, diagnostics = run_v2322_audit()
    assert checks["passed"].all()
    assert diagnostics["exact_within_tolerance"].all()
