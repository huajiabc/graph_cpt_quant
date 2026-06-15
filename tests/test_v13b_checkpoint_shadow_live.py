from __future__ import annotations

import pandas as pd

from pressure_graph.paper_live.v07d2 import (
    CHECKPOINT_PORTFOLIO_IDS,
    CP60_PROTECT_A_BETA_HIGH_THRESHOLD,
    V22BConfig,
    checkpoint_shadow_live,
    _pre_entry_router_counterfactual_live,
)


def _trade(symbol: str, entry: pd.Timestamp, *, net20: float = -0.02) -> dict[str, object]:
    return {
        "trade_id": f"CIC1:{symbol}:{entry.isoformat()}",
        "signal_id": f"CIC1:{symbol}:{entry.isoformat()}",
        "candidate": "CIC1_FILTERED_MIR1",
        "baseline_kind": "",
        "exchange": "bybit",
        "symbol": symbol,
        "local_volume_shock_time": entry - pd.Timedelta(minutes=45),
        "entry_time": entry,
        "entry_price": 100.0,
        "exit_time": entry + pd.Timedelta(hours=4),
        "exit_price": 100.0 * (1.0 + net20 + 0.004),
        "exit_reason": "max_hold",
        "gross_return": net20 + 0.004,
        "net_return_10bp": net20 + 0.002,
        "net_return_20bp": net20,
        "net_return_30bp": net20 - 0.002,
        "net_return_50bp": net20 - 0.006,
        "cluster_impulse_density_at_entry": 0.5,
        "c2_beta_extension_score": 100.0,
    }


def _bars(symbol: str, *, price: float = 99.0) -> list[dict[str, object]]:
    rows = []
    for idx in range(22):
        feature_time = pd.Timestamp("2026-06-01T00:00:00Z") + pd.Timedelta(minutes=15 * idx)
        rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "bar_open_time": feature_time - pd.Timedelta(minutes=15),
                "feature_time": feature_time,
                "open": price,
                "close": price,
            }
        )
    return rows


def test_cp60_shadow_releases_slot_for_later_candidate() -> None:
    entry = pd.Timestamp("2026-06-01T00:00:00Z")
    later_entry = pd.Timestamp("2026-06-01T01:10:00Z")
    trades = pd.DataFrame(
        [_trade(f"S{idx}USDT", entry, net20=-0.02) for idx in range(8)]
        + [_trade("S8USDT", later_entry, net20=0.08)]
    )
    prepared = pd.DataFrame(
        [row for idx in range(8) for row in _bars(f"S{idx}USDT", price=99.0)]
        + _bars("S8USDT", price=110.0)
    )

    status, ledger, skipped, _, summary, slot = checkpoint_shadow_live(trades, prepared)

    baseline_id = CHECKPOINT_PORTFOLIO_IDS["S0"]
    cp_id = CHECKPOINT_PORTFOLIO_IDS["S2"]
    baseline_selected = ledger[ledger["portfolio_id"].eq(baseline_id)]
    baseline_skipped = skipped[skipped["portfolio_id"].eq(baseline_id)]
    cp_selected = ledger[ledger["portfolio_id"].eq(cp_id)]

    assert len(baseline_selected) == 8
    assert baseline_skipped["symbol"].tolist() == ["S8USDT"]
    assert "S8USDT" in cp_selected["symbol"].tolist()
    assert int(status[status["portfolio_id"].eq(cp_id)]["checkpoint_exits"].iloc[0]) == 8

    cp20 = summary[
        summary["portfolio_id"].eq(cp_id)
        & summary["cost_single_side_bps"].eq(20)
    ].iloc[0]
    base20 = summary[
        summary["portfolio_id"].eq(baseline_id)
        & summary["cost_single_side_bps"].eq(20)
    ].iloc[0]
    assert cp20["portfolio_net"] > base20["portfolio_net"]
    assert slot["new_trade_entered_due_to_release"].fillna(False).astype(bool).any()


def test_protect_a_cap2_limits_protected_exits_per_burst() -> None:
    entry = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            {
                **_trade(f"S{idx}USDT", entry + pd.Timedelta(minutes=5 * idx), net20=0.04),
                "c2_beta_extension_score": CP60_PROTECT_A_BETA_HIGH_THRESHOLD + 0.01,
            }
            for idx in range(3)
        ]
    )
    prepared = pd.DataFrame(
        [row for idx in range(3) for row in _bars(f"S{idx}USDT", price=99.0)]
    )

    status, ledger, _, _, summary, _ = checkpoint_shadow_live(trades, prepared)

    s4_id = CHECKPOINT_PORTFOLIO_IDS["S4"]
    s4 = ledger[ledger["portfolio_id"].eq(s4_id)].copy()
    assert len(s4) == 3
    assert int(s4["protected_by_beta_high"].fillna(False).astype(bool).sum()) == 2
    assert int(s4["checkpoint_triggered"].fillna(False).astype(bool).sum()) == 1
    assert pd.to_numeric(s4["protected_burst_count_after"], errors="coerce").dropna().max() == 2

    s4_status = status[status["portfolio_id"].eq(s4_id)].iloc[0]
    assert int(s4_status["protected_exits"]) == 2
    assert int(s4_status["protection_cap"]) == 2

    s4_20 = summary[
        summary["portfolio_id"].eq(s4_id)
        & summary["cost_single_side_bps"].eq(20)
    ].iloc[0]
    assert int(s4_20["protected_exits"]) == 2


def test_pre_entry_router_counterfactual_live_scores_without_changing_actions(tmp_path) -> None:
    rows = []
    base_time = pd.Timestamp("2026-01-01T00:00:00Z")
    for idx in range(50):
        label = "no_trade" if idx % 2 else "core_trade"
        rows.append(
            {
                "trade_key": f"train-{idx}",
                "symbol": f"T{idx}USDT",
                "candidate": "CIC1_beta_extreme" if idx % 2 else "CIC2_beta_broad",
                "entry_time": (base_time + pd.Timedelta(days=idx)).isoformat(),
                "entry_month": (base_time + pd.Timedelta(days=idx)).strftime("%Y-%m"),
                "period": "search",
                "meta_router_training_split": "search",
                "state_cluster_id": "SDG00",
                "cic_type": "CIC1" if idx % 2 else "CIC2",
                "btc_state": "BTC_up",
                "market_impulse_density": 0.1 if label == "no_trade" else 0.8,
                "cluster_density": 0.1 if label == "no_trade" else 0.7,
                "beta_strength": 98.0,
                "local_shock_strength": 2.0,
                "ret_4h": 0.05,
                "ret_4h_percentile": 98.0,
                "symbol_volatility_percentile": 50.0,
                "burst_count_so_far": 1 if label == "no_trade" else 12,
                "minutes_since_burst_start": 0.0,
                "same_timestamp_peer_count": 1 if label == "no_trade" else 10,
                "walkforward_state_novelty": 0.0,
                "novelty_bucket": "q1",
                "net20": -0.02 if label == "no_trade" else 0.02,
                "pre_entry_action_label": label,
                "post_entry_checkpoint_label": "not_cp60_eligible",
                "protect_a_label": "not_protect_candidate",
                "capacity_overflow_label": "overflow_neutral",
                "utility_core_trade": 0.0,
                "utility_no_trade": 0.0,
                "utility_reduce_size_50": 0.0,
                "utility_cp60_exit_if_triggered": 0.0,
                "utility_o6_overflow_025": 0.0,
                "utility_o6_overflow_050": 0.0,
                "checkpoint_exit_delta_vs_keep": 0.0,
                "protect_keep_delta_vs_cp60": 0.0,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "meta_router_feature_matrix.csv", index=False)

    live_entry = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            {
                **_trade("AUSDT", live_entry, net20=-0.02),
                "candidate": "CIC1_FILTERED_MIR1",
                "volume_impulse_density_at_entry": 0.1,
                "cluster_impulse_density_at_entry": 0.1,
                "beta_extension_score_at_signal": 98.0,
                "local_volume_shock_strength_at_signal": 2.0,
                "btc_state_at_entry": "BTC_up",
            },
            {
                **_trade("BUSDT", live_entry + pd.Timedelta(minutes=15), net20=0.03),
                "candidate": "CIC2_FILTERED_MIR1",
                "volume_impulse_density_at_entry": 0.8,
                "cluster_impulse_density_at_entry": 0.7,
                "beta_extension_score_at_signal": 98.0,
                "local_volume_shock_strength_at_signal": 2.0,
                "btc_state_at_entry": "BTC_up",
            },
        ]
    )

    out = _pre_entry_router_counterfactual_live(
        trades,
        train_root=tmp_path,
        cfg=V22BConfig(min_train_months=1, min_train_events=40, logistic_steps=40),
    )

    assert len(out) == 2
    assert out["router_score_status"].eq("scored_prior_only_logistic").all()
    assert {"would_skip_t70", "would_skip_t75", "would_skip_t80"}.issubset(out.columns)
    assert out["actual_action"].eq("p2_candidate_observed").all()
