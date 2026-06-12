from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09e2 import (
    V09E2Config,
    _add_upside_vacuum_features,
    _ranking_summary,
    build_pool_trade_source,
    build_unique_replay_targets,
    load_capacity_context_rows,
)


def _write_capacity_files(root) -> None:
    base = pd.Timestamp("2025-08-22T14:30:00Z")
    selected = pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "pool": "P2_CIC1_CIC2_COMBINED",
                "max_positions": 5,
                "candidate": "CIC1_beta_extreme",
                "symbol": "AAAUSDT",
                "entry_time": base,
                "exit_time": base + pd.Timedelta(hours=1),
                "net_return": 0.02,
                "net_return_10bp": 0.022,
                "net_return_20bp": 0.020,
                "signal_id": "s1",
                "trade_id": "t1",
            },
            {
                "exchange": "bybit",
                "pool": "P0_CIC1_ONLY",
                "max_positions": 8,
                "candidate": "CIC1_beta_extreme",
                "symbol": "CCCUSDT",
                "entry_time": base + pd.Timedelta(minutes=15),
                "exit_time": base + pd.Timedelta(hours=1),
                "net_return": 0.01,
                "net_return_10bp": 0.012,
                "net_return_20bp": 0.010,
                "signal_id": "s3",
                "trade_id": "t3",
            },
        ]
    )
    skipped = pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "pool": "P2_CIC1_CIC2_COMBINED",
                "max_positions": 5,
                "candidate": "CIC2_beta_broad",
                "symbol": "BBBUSDT",
                "entry_time": base,
                "exit_time": base + pd.Timedelta(hours=1),
                "net_return": 0.03,
                "net_return_10bp": 0.032,
                "net_return_20bp": 0.030,
                "signal_id": "s2",
                "trade_id": "t2",
                "skip_reason": "portfolio_full",
            }
        ]
    )
    selected.to_csv(root / "portfolio_timeline.csv", index=False)
    skipped.to_csv(root / "portfolio_skipped_candidates.csv", index=False)


def test_v09e2_builds_fair_window_targets(tmp_path) -> None:
    _write_capacity_files(tmp_path)
    cfg = V09E2Config(capacity_root=tmp_path)

    context = load_capacity_context_rows(cfg)
    targets = build_unique_replay_targets(context)
    source = build_pool_trade_source(context)

    assert len(context) == 3
    assert targets["is_skipped_target"].sum() == 1
    assert set(source["analysis_pool"]) == {"P2_CIC1_CIC2_COMBINED", "P0_CIC1_ONLY"}
    assert set(source["candidate"]) == {"CIC1_FILTERED_MIR1", "CIC2_FILTERED_MIR1"}


def test_v09e2_ranking_prefers_thin_ask_depth_when_available() -> None:
    base = pd.Timestamp("2025-08-22T14:30:00Z")
    data = pd.DataFrame(
        [
            {
                "analysis_pool": "P2_CIC1_CIC2_COMBINED",
                "candidate": "CIC1_FILTERED_MIR1",
                "symbol": "AAAUSDT",
                "entry_time": base + pd.Timedelta(minutes=idx),
                "exit_time": base + pd.Timedelta(hours=1, minutes=idx),
                "net_return_10bp": ret + 0.002,
                "net_return_20bp": ret,
                "coverage_status": "covered",
                "strict_asof_covered": True,
                "sensitivity_covered": True,
                "ask_depth_10bp": ask,
                "ask_depth_25bp": ask,
                "ask_depth_50bp": ask * 2,
                "bid_depth_25bp": 10_000,
                "spread_bps": 1.0,
                "downside_liquidity_risk_25bp": 0.0,
                "cluster_impulse_density": 0.5,
            }
            for idx, (ask, ret) in enumerate([(100, 0.05), (200, 0.04), (10_000, -0.02), (12_000, -0.01)])
        ]
    )
    ranked = _add_upside_vacuum_features(data)
    summary, _, _ = _ranking_summary(ranked, modes=("strict_asof_only",))
    row = summary[
        summary["pool"].eq("P2_CIC1_CIC2_COMBINED")
        & summary["ranking_rule"].eq("R2_ask_depth_25bp_low_first")
        & summary["max_positions"].eq(3)
    ].iloc[0]

    assert row["selected_net20"] > row["skipped_net20"]
