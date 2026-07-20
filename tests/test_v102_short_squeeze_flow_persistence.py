from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v102_short_squeeze_flow_persistence import summarize_v102


def test_v102_summary_reports_selected_minus_other_lift() -> None:
    panel = pd.DataFrame(
        {
            "period": ["validation", "validation"],
            "OF1_CONFIRM_LONG": [True, False],
            "symbol": ["ETHUSDT", "ETHUSDT"],
            "entry_day": ["2026-05-01", "2026-05-01"],
            "net_return_240m_10bp": [0.03, 0.00],
            "net_return_240m_20bp": [0.02, -0.01],
            "net_return_240m_30bp": [0.01, -0.02],
            "hedged_net_return_240m_40bp": [0.005, -0.02],
        }
    )

    row = summarize_v102(panel).query("scope == 'validation'").iloc[0]

    assert row["mean_raw_net20"] == 0.02
    assert row["selected_minus_other_net20"] == 0.03
    assert row["mean_hedged_net40"] == 0.005
