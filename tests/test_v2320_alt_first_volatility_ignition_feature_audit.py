import pandas as pd

from pressure_graph.reports.v2320_alt_first_volatility_ignition_feature_audit import (
    write_v2320_alt_first_volatility_ignition_feature_audit,
)


def test_v2320_real_feature_audit_passes() -> None:
    paths = write_v2320_alt_first_volatility_ignition_feature_audit()
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["features"])
    assert checks["passed"].all()
    assert len(features) == 100
    assert "primary_net_return" not in features.columns
