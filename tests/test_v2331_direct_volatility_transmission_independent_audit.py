import pandas as pd

from pressure_graph.reports.v2331_direct_volatility_transmission_independent_audit import (
    write_v2331_direct_volatility_transmission_independent_audit,
)


def test_v2331_independent_audit_passes() -> None:
    paths = write_v2331_direct_volatility_transmission_independent_audit()
    checks = pd.read_csv(paths["checks"])
    assert len(checks) == 12
    assert checks["passed"].all()
