import pandas as pd

from pressure_graph.reports.v2326_multisource_oco_model_feature_audit import (
    MODEL_FEATURES,
    write_v2326_multisource_oco_model_feature_audit,
)


def test_v2326_real_feature_audit_passes() -> None:
    paths = write_v2326_multisource_oco_model_feature_audit()
    checks = pd.read_csv(paths["checks"])
    features = pd.read_parquet(paths["features"])
    assert checks["passed"].all()
    assert len(features) == 159
    assert len(MODEL_FEATURES) == 19
    assert "primary_net_return" not in features.columns
