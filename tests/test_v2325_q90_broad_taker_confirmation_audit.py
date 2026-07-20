from pressure_graph.reports.v2325_q90_broad_taker_confirmation_audit import (
    run_v2325_audit,
)


def test_v2325_independent_audit_passes() -> None:
    checks, diagnostics = run_v2325_audit()
    assert checks["passed"].all()
    assert diagnostics["exact_within_tolerance"].all()
