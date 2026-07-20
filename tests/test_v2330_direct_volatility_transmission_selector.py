import pandas as pd

from pressure_graph.reports.v2330_direct_volatility_transmission_selector import (
    SCORE_FEATURES,
    write_v2330_direct_volatility_transmission_selector,
)


def test_v2330_real_report_reconciles() -> None:
    paths = write_v2330_direct_volatility_transmission_selector()
    selection = pd.read_parquet(paths["selection"])
    summary = pd.read_csv(paths["summary"])
    gates = pd.read_csv(paths["gates"])
    assert len(selection) == 96
    assert len(SCORE_FEATURES) == 4
    assert set(summary["scope"]) == {"validation", "holdout", "oos"}
    assert len(gates) == 10
    assert selection["transmission_score"].notna().all()
