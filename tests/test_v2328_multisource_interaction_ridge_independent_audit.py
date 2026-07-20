import pandas as pd

from pressure_graph.reports.v2328_multisource_interaction_ridge_independent_audit import (
    write_v2328_multisource_interaction_ridge_independent_audit,
)


def test_v2328_independent_audit_passes() -> None:
    paths = write_v2328_multisource_interaction_ridge_independent_audit()
    checks = pd.read_csv(paths["checks"])
    predictions = pd.read_parquet(paths["predictions"])
    assert len(checks) == 13
    assert checks["passed"].all()
    assert len(predictions) == 192
