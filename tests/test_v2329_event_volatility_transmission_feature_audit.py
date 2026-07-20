import pandas as pd

from pressure_graph.reports.v2329_event_volatility_transmission_feature_audit import (
    VOLATILITY_FEATURES,
    write_v2329_event_volatility_transmission_feature_audit,
)


def test_v2329_real_feature_audit_passes() -> None:
    paths = write_v2329_event_volatility_transmission_feature_audit()
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["features"])
    assert len(features) == 159
    assert len(VOLATILITY_FEATURES) == 17
    assert checks["passed"].all()
    assert "primary_net_return" not in features.columns
