from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v94_forward_monitoring import build_forward_funnel, build_sample_progress


def test_forward_funnel_deduplicates_candidate_rows_into_opportunities() -> None:
    signals = pd.DataFrame(
        [
            {
                "signal_id": "s1",
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "candidate": "CIC1_FILTERED_MIR1",
                "feature_time": "2026-07-11T10:00:00Z",
                "timely_forward_observation": True,
                "skip_reason": "entry_market_gate_off:cic1",
                "status": "skipped",
            },
            {
                "signal_id": "s2",
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "candidate": "CIC2_FILTERED_MIR1",
                "feature_time": "2026-07-11T10:00:00Z",
                "timely_forward_observation": True,
                "skip_reason": "",
                "pullback_time": "2026-07-11T10:15:00Z",
                "entry_time": "2026-07-11T10:30:00Z",
                "exit_time": "2026-07-11T11:30:00Z",
                "portfolio_accepted": True,
                "status": "exited",
            },
        ]
    )
    funnel, skips = build_forward_funnel(signals)
    counts = funnel.set_index("stage")["unique_opportunities"].to_dict()
    assert counts == {
        "observed": 1,
        "market_gate_passed": 1,
        "pullback_seen": 1,
        "entry_created": 1,
        "portfolio_accepted": 1,
        "trade_completed": 1,
    }
    assert int(skips["signal_rows"].sum()) == 2


def test_sample_progress_uses_only_timely_completed_rows() -> None:
    base = {
        "trade_id": "t1",
        "exit_time": "2026-07-11T11:00:00Z",
        "timely_forward_observation": True,
    }
    checkpoints = pd.DataFrame(
        [
            {**base, "portfolio_id": "P2_MAX8_BASELINE"},
            {**base, "trade_id": "t2", "portfolio_id": "P2_MAX8_CP60", "exit_reason": "checkpoint_60_exit"},
            {
                **base,
                "trade_id": "t3",
                "portfolio_id": "P2_MAX8_CP60_PROTECT_A_CAP2",
                "protection_applied": True,
            },
        ]
    )
    risk = pd.DataFrame(
        [{**base, "risk_shadow_arm": "P2_VOL", "position_size": 0.5}]
    )
    risk_skipped = pd.DataFrame(
        [
            {
                **base,
                "risk_shadow_arm": "P2_CORR",
                "skip_reason": "correlation_cluster_cap",
            }
        ]
    )
    token = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "candidate": "CIC1_FILTERED_MIR1",
                "net20_later": 0.01,
                "timely_forward_observation": True,
                "token_prior_24h": True,
            }
        ]
    )
    progress = build_sample_progress(
        checkpoint_trades=checkpoints,
        overflow_trades=pd.DataFrame(),
        risk_shadow_trades=risk,
        risk_shadow_skipped=risk_skipped,
        token_context=token,
        primary_portfolio_id="P2_MAX8_BASELINE",
    ).set_index("metric")
    assert progress.loc["P2 initial review", "observed"] == 1
    assert progress.loc["CP60 initial review", "observed"] == 1
    assert progress.loc["Protect_A initial review", "observed"] == 1
    assert progress.loc["P2_VOL constrained decisions", "observed"] == 1
    assert progress.loc["P2_CORR constrained decisions", "observed"] == 1
    assert progress.loc["Token prior-24h trades", "observed"] == 1
