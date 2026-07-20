import pandas as pd

from pressure_graph.reports.v2321_alt_first_volatility_ignition_breakout import (
    write_v2321_alt_first_volatility_ignition_breakout,
)


def test_v2321_real_reveal_is_reproducible() -> None:
    paths = write_v2321_alt_first_volatility_ignition_breakout()
    summary = pd.read_csv(paths["summary"])
    random_summary = pd.read_csv(paths["random_summary"])
    assert summary.loc[summary["scope"].eq("all"), "events"].iloc[0] == 100
    assert random_summary["unmatched_events"].eq(0).all()
