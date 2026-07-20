import pandas as pd

from pressure_graph.reports.v2323_q90_broad_taker_confirmation_feature_audit import (
    write_v2323_q90_broad_taker_confirmation_feature_audit,
)


def test_v2323_real_feature_audit_passes() -> None:
    paths = write_v2323_q90_broad_taker_confirmation_feature_audit()
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["features"])
    assert checks["passed"].all()
    assert len(features) == 26
    assert "primary_net_return" not in features.columns
