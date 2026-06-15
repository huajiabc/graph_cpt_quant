from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10c_burst_phase_allocation import _add_asof_burst_phase
from pressure_graph.reports.v13a_checkpoint_robustness import (
    CheckpointSpec,
    PortfolioSpec,
    _checkpoint_o6_integration,
    _checkpoint_time_sensitivity,
    _checkpoint_by_cic_type,
    _prepare_checkpoint_sample,
    _run_spec,
)


def _base_pool_and_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    base_time = pd.Timestamp("2026-01-01T00:00:00Z")
    trade_rows = []
    price_rows = []
    for idx in range(12):
        early = idx < 8
        entry = base_time + pd.Timedelta(minutes=5 * idx if early else 70 + 5 * (idx - 8))
        signal = entry - pd.Timedelta(minutes=30)
        symbol = f"S{idx:02d}USDT"
        net = 0.004 if early else 0.05
        trade_rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
                "base_signal_id": f"{signal.isoformat()}|{symbol}",
                "signal_time": signal,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "entry_price": 100.0,
                "gross_return": net + 0.004,
                "funding_cost": 0.0,
                "cost_single_side_bps": 20.0,
                "net_return": net,
                "month": "2026-01",
            }
        )
        for minutes in (30, 60, 90, 120):
            close = 99.0 if early else 105.0
            price_rows.append(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "feature_time": entry + pd.Timedelta(minutes=minutes),
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                }
            )
    pool = _add_asof_burst_phase(pd.DataFrame(trade_rows), "1h")
    return pool, pd.DataFrame(price_rows)


def test_checkpoint_time_sensitivity_releases_capacity() -> None:
    pool, prices = _base_pool_and_prices()

    summary = _checkpoint_time_sensitivity(pool, prices)

    assert set(summary["checkpoint_minutes"]) == {30, 60, 90, 120}
    assert summary["early_exit_trades"].max() > 0
    assert summary["portfolio_net20"].max() > 0
    assert summary["portfolio_net20"].max() > summary["portfolio_net20"].min()


def test_checkpoint_o6_integration_beats_core_baseline() -> None:
    pool, prices = _base_pool_and_prices()

    integration, ledger, _ = _checkpoint_o6_integration(pool, prices)

    s0 = integration[integration["portfolio_id"].eq("S0_P2_MAX8_BASELINE")].iloc[0]
    s1 = integration[integration["portfolio_id"].eq("S1_P2_MAX8_CHECKPOINT_60M")].iloc[0]
    s2 = integration[integration["portfolio_id"].eq("S2_P2_MAX8_PLUS_O6")].iloc[0]
    assert s1["portfolio_net20"] > s0["portfolio_net20"]
    assert s2["overflow_trades"] > 0
    assert not ledger.empty


def test_positive_threshold_exits_more_than_zero_threshold() -> None:
    pool, prices = _base_pool_and_prices()
    zero = PortfolioSpec(
        "zero",
        True,
        False,
        CheckpointSpec("zero", 60, 0.0, "net_lte_threshold", 20.0),
    )
    plus = PortfolioSpec(
        "plus",
        True,
        False,
        CheckpointSpec("plus", 60, 0.005, "net_lte_threshold", 20.0),
    )
    sample = _prepare_checkpoint_sample(pool, prices, zero.checkpoint)
    zero_row, _, _ = _run_spec(sample, zero)
    plus_row, _, _ = _run_spec(sample, plus)

    assert plus_row["early_exit_trades"] >= zero_row["early_exit_trades"]


def test_checkpoint_by_cic_type_reports_false_exit_and_avoidance() -> None:
    pool, prices = _base_pool_and_prices()
    integration, ledger, _ = _checkpoint_o6_integration(pool, prices)
    assert not integration.empty
    baseline = ledger[ledger["portfolio_id"].eq("S0_P2_MAX8_BASELINE")]
    checkpoint = ledger[ledger["portfolio_id"].eq("S1_P2_MAX8_CHECKPOINT_60M")]

    summary = _checkpoint_by_cic_type(baseline, checkpoint)

    assert {"CIC1", "CIC2"}.issubset(set(summary["cic_type"]))
    assert (summary["checkpoint_exits"] > 0).any()
    assert "false_exit_rate" in summary.columns
    assert "avoided_loss_sum" in summary.columns
