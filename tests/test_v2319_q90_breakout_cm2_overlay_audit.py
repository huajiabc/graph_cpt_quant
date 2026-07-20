from pressure_graph.reports.v2319_q90_breakout_cm2_overlay_audit import (
    run_v2319_audit,
)


def test_v2319_independent_audit_passes() -> None:
    checks, diagnostics = run_v2319_audit()
    assert checks["passed"].all()
    assert diagnostics["exact_within_tolerance"].all()
