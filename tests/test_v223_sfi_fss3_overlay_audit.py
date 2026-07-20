from pressure_graph.reports.v223_sfi_fss3_overlay_audit import run_v223_audit


def test_v223_independent_audit_passes_and_rejection_is_real() -> None:
    checks, errors, gates, _ = run_v223_audit()
    assert checks["passed"].all()
    assert errors["maximum_absolute_error"].max() <= 1e-12
    assert not gates["passed"].all()
    assert not gates.loc[
        gates["gate"].eq("positive_active_fss3_primary"), "passed"
    ].iloc[0]
