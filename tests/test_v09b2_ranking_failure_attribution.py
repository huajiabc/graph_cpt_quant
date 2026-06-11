from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v09b2 import write_v09b2_ranking_failure_attribution


def test_v09b2_builds_failure_attribution_from_v09b_summary(tmp_path) -> None:
    source = tmp_path / "v09b"
    report = tmp_path / "v09b2"
    source.mkdir()
    pd.DataFrame(
        [
            {
                "pool": "P0_CIC1_ONLY",
                "ranking": "cluster_impulse_density_high",
                "cost_single_side_bps": 20,
                "max_positions": 5,
                "selected_trades": 3,
                "skipped_trades": 2,
                "net_expectancy": 0.01,
                "skipped_avg_net": 0.03,
                "random_median_net": 0.008,
                "random_p75_net": 0.009,
                "random_p90_net": 0.02,
                "month_cap35_net": 0.006,
            }
        ]
    ).to_csv(source / "ranking_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "pool": "P0_CIC1_ONLY",
                "ranking": "cluster_impulse_density_high",
                "max_positions": 5,
                "skip_reason": "portfolio_full",
                "skipped_trades": 2,
                "skipped_avg_net20": 0.03,
                "skipped_good_trade_rate": 1.0,
            }
        ]
    ).to_csv(source / "skipped_trade_attribution.csv", index=False)
    pd.DataFrame(
        [
            {
                "pool": "P0_CIC1_ONLY",
                "feature": "cluster_impulse_density",
                "bucket": "q4_high",
                "trades": 2,
                "net20": 0.02,
                "good_trade_rate": 1.0,
            },
            {
                "pool": "P0_CIC1_ONLY",
                "feature": "cluster_impulse_density",
                "bucket": "q3",
                "trades": 2,
                "net20": 0.04,
                "good_trade_rate": 1.0,
            },
        ]
    ).to_csv(source / "feature_rank_bucket_summary.csv", index=False)

    outputs = write_v09b2_ranking_failure_attribution(source, report)

    assert outputs["conflict_set_analysis"].exists()
    conflict = pd.read_csv(outputs["conflict_set_analysis"])
    assert conflict.iloc[0]["selected_minus_skipped"] == pytest.approx(-0.02)
    skip = pd.read_csv(outputs["skip_reason_attribution"])
    assert skip.iloc[0]["skip_reason"] == "portfolio_full"
    shape = pd.read_csv(outputs["cluster_density_shape"])
    assert shape.iloc[0]["interpretation"] == "nonlinear_or_crowding_possible"
    notes = outputs["candidate_notes"].read_text(encoding="utf-8")
    assert "No CIC/MIR1 parameters were changed" in notes
