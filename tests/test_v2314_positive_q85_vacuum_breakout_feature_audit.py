import pandas as pd

from pressure_graph.reports.v2314_positive_q85_vacuum_breakout_feature_audit import (
    write_v2314_positive_q85_vacuum_breakout_feature_audit,
)


def test_v2314_q85_feature_audit_passes() -> None:
    paths = write_v2314_positive_q85_vacuum_breakout_feature_audit()
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["features"])
    assert checks["passed"].all()
    assert len(features) >= 70
