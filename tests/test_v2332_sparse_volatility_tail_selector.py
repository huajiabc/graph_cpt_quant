import pandas as pd

from pressure_graph.reports.v2332_sparse_volatility_tail_selector import (
    write_v2332_sparse_volatility_tail_selector,
)


def test_v2332_real_report_reconciles() -> None:
    paths = write_v2332_sparse_volatility_tail_selector()
    selection = pd.read_parquet(paths["selection"])
    rules = pd.read_csv(paths["rules"])
    gates = pd.read_csv(paths["gates"])
    assert len(selection) == 96
    assert len(rules) == 2
    assert len(gates) == 10
    assert selection["selected"].dtype == bool
