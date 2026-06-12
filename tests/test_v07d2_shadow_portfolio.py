from __future__ import annotations

import pandas as pd

from pressure_graph.config.v07a2 import load_v07a2_config
from pressure_graph.paper_live import v07d2
from pressure_graph.paper_live.v07d2 import add_v07d2_live_columns, shadow_portfolio_live


def _trade(
    symbol: str,
    *,
    candidate: str = "CIC1_FILTERED_MIR1",
    signal_time: pd.Timestamp,
    score: float,
    net20: float,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "baseline_kind": "",
        "exchange": "bybit",
        "symbol": symbol,
        "local_volume_shock_time": signal_time,
        "entry_time": signal_time + pd.Timedelta(minutes=45),
        "exit_time": signal_time + pd.Timedelta(hours=2),
        "exit_reason": "tp" if net20 > 0 else "sl",
        "cluster_impulse_density_at_entry": score,
        "net_return_10bp": net20 + 0.002,
        "net_return_20bp": net20,
        "net_return_30bp": net20 - 0.002,
        "net_return_50bp": net20 - 0.006,
    }


def test_v07d2_relaxed_shadow_gate_columns(monkeypatch) -> None:
    monkeypatch.setattr(v07d2, "_add_cluster_context", lambda frame: frame.copy())
    config = load_v07a2_config("configs/v0_7d2_cic_mir1_paper_live.yaml")
    frame = pd.DataFrame(
        [
            {
                "exchange": "bybit",
                "symbol": "AAAUSDT",
                "bar_open_time": pd.Timestamp("2026-06-01T00:00:00Z"),
                "feature_time": pd.Timestamp("2026-06-01T00:15:00Z"),
                "warmup_complete": True,
                "volume_z_4h": 3.0,
                "ret_4h": 0.02,
                "ret_4h_percentile": 92.0,
                "volume_impulse_density": 0.09,
                "btc_market_state": "BTC_chop",
            },
            {
                "exchange": "bybit",
                "symbol": "BBBUSDT",
                "bar_open_time": pd.Timestamp("2026-06-01T00:00:00Z"),
                "feature_time": pd.Timestamp("2026-06-01T00:15:00Z"),
                "warmup_complete": True,
                "volume_z_4h": 3.0,
                "ret_4h": 0.02,
                "ret_4h_percentile": 92.0,
                "volume_impulse_density": 0.10,
                "btc_market_state": "BTC_chop",
            },
        ]
    )

    out = add_v07d2_live_columns(frame, config)

    assert not bool(out.loc[0, "c2_relax_beta90"])
    assert bool(out.loc[0, "c2_relax_density08_beta90"])
    assert bool(out.loc[1, "c2_relax_beta90"])
    assert bool(out.loc[1, "c2_relax_density08_beta90"])


def test_v07d2_shadow_portfolio_records_selected_and_skipped_counterfactuals() -> None:
    signal_time = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            _trade(f"S{idx}USDT", signal_time=signal_time, score=float(idx), net20=0.01 * idx)
            for idx in range(6)
        ]
    )

    status, selected, skipped, _, summary = shadow_portfolio_live(trades)

    p0_status = status[status["portfolio_id"].eq("P0_CIC1_CLUSTER_RANK_MAX5")].iloc[0]
    assert p0_status["selected_trades"] == 5
    assert p0_status["skipped_candidates"] == 1
    p0_selected = selected[selected["portfolio_id"].eq("P0_CIC1_CLUSTER_RANK_MAX5")]
    p0_skipped = skipped[skipped["portfolio_id"].eq("P0_CIC1_CLUSTER_RANK_MAX5")]
    assert "S5USDT" in p0_selected["symbol"].tolist()
    assert p0_skipped["symbol"].tolist() == ["S0USDT"]
    focal = summary[
        summary["portfolio_id"].eq("P0_CIC1_CLUSTER_RANK_MAX5")
        & summary["cost_single_side_bps"].eq(20)
    ].iloc[0]
    assert focal["selected_net"] > focal["skipped_net"]


def test_v07d2_combined_shadow_dedupes_cic2_when_cic1_exists() -> None:
    signal_time = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            _trade(
                "AAAUSDT",
                candidate="CIC1_FILTERED_MIR1",
                signal_time=signal_time,
                score=0.5,
                net20=0.03,
            ),
            _trade(
                "AAAUSDT",
                candidate="CIC2_FILTERED_MIR1",
                signal_time=signal_time,
                score=0.9,
                net20=-0.02,
            ),
        ]
    )

    status, selected, _, _, _ = shadow_portfolio_live(trades)

    p2_status = status[status["portfolio_id"].eq("P2_CIC_COMBINED_CLUSTER_RANK_MAX5")].iloc[0]
    assert p2_status["candidate_count"] == 1
    p2_selected = selected[selected["portfolio_id"].eq("P2_CIC_COMBINED_CLUSTER_RANK_MAX5")]
    assert p2_selected["candidate"].tolist() == ["CIC1_FILTERED_MIR1"]


def test_v07d2_combined_basket_max8_is_first_come_shadow() -> None:
    signal_time = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            _trade(f"S{idx}USDT", signal_time=signal_time, score=100.0 - idx, net20=0.01)
            for idx in range(10)
        ]
    )

    status, selected, skipped, _, _ = shadow_portfolio_live(trades)

    basket_status = status[status["portfolio_id"].eq("P2_CIC_COMBINED_BASKET_MAX8")].iloc[0]
    assert basket_status["ranking"] == "first_come_basket"
    assert basket_status["selected_trades"] == 8
    assert basket_status["skipped_candidates"] == 2
    basket_selected = selected[selected["portfolio_id"].eq("P2_CIC_COMBINED_BASKET_MAX8")]
    basket_skipped = skipped[skipped["portfolio_id"].eq("P2_CIC_COMBINED_BASKET_MAX8")]
    assert basket_selected["symbol"].tolist() == [f"S{idx}USDT" for idx in range(8)]
    assert basket_skipped["skip_reason"].tolist() == ["portfolio_full", "portfolio_full"]


def test_v07d2_late_burst_overflow_selects_small_overflow_slots() -> None:
    signal_time = pd.Timestamp("2026-06-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            _trade(f"S{idx}USDT", signal_time=signal_time + pd.Timedelta(minutes=5 * idx), score=0.0, net20=0.02)
            for idx in range(10)
        ]
    )

    status, selected, skipped, _, _ = shadow_portfolio_live(trades)

    overflow_status = status[status["portfolio_id"].eq("P2_MAX8_PLUS_O6_LATE_BURST_OVERFLOW")].iloc[0]
    assert overflow_status["selected_trades"] == 10
    assert overflow_status["core_trades"] == 8
    assert overflow_status["overflow_trades"] == 2
    assert overflow_status["skipped_candidates"] == 0

    ledger = selected[selected["portfolio_id"].eq("P2_MAX8_PLUS_O6_LATE_BURST_OVERFLOW")]
    overflow = ledger[ledger["is_overflow"].astype(bool)]
    assert overflow["burst_count_so_far"].min() >= 9
    assert overflow["position_size"].tolist() == [0.5, 0.5]
    assert skipped[skipped["portfolio_id"].eq("P2_MAX8_PLUS_O6_LATE_BURST_OVERFLOW")].empty
