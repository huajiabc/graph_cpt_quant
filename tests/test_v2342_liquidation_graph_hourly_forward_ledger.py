import pandas as pd

from pressure_graph.reports.v2342_liquidation_graph_hourly_forward_ledger import (
    available_v2342_decisions,
)


def test_v2342_available_decisions_start_at_first_complete_hour() -> None:
    decisions = available_v2342_decisions(
        pd.Timestamp("2026-07-17T04:25:50Z"),
        pd.Timestamp("2026-07-17T07:03:00Z"),
    )
    assert list(decisions) == list(
        pd.date_range("2026-07-17T05:00:00Z", periods=3, freq="h")
    )


def test_v2342_no_decision_before_first_complete_hour() -> None:
    decisions = available_v2342_decisions(
        pd.Timestamp("2026-07-17T04:25:50Z"),
        pd.Timestamp("2026-07-17T04:59:59Z"),
    )
    assert len(decisions) == 0
