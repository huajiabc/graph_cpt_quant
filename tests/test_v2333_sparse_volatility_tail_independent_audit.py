import pandas as pd

from pressure_graph.reports.v2333_sparse_volatility_tail_independent_audit import (
    write_v2333_sparse_volatility_tail_independent_audit,
)


def test_v2333_independent_audit_passes() -> None:
    paths = write_v2333_sparse_volatility_tail_independent_audit()
    checks = pd.read_csv(paths["checks"])
    assert len(checks) == 12
    assert checks["passed"].all()
