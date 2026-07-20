import pandas as pd

from pressure_graph.reports.v2324_q90_broad_taker_confirmation import (
    write_v2324_q90_broad_taker_confirmation,
)


def test_v2324_real_reveal_is_reproducible() -> None:
    paths = write_v2324_q90_broad_taker_confirmation()
    summary = pd.read_csv(paths["summary"])
    random_summary = pd.read_csv(paths["random_summary"])
    assert summary.loc[summary["scope"].eq("all"), "events"].iloc[0] == 26
    assert random_summary.loc[
        random_summary["scope"].eq("all"), "unmatched_events"
    ].iloc[0] == 2
