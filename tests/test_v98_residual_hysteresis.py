from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v98_residual_hysteresis import (
    V98Config,
    _select_ranked,
    build_v98_portfolio_ledger,
    prepare_v98_dataset,
)


def _row(
    timestamp: str,
    symbol: str,
    ret_4h: float,
    future_return: float,
    btc_ret_4h: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": symbol,
        "feature_time": pd.Timestamp(timestamp),
        "universe_dynamic_monthly_top30": True,
        "warmup_complete": True,
        "future_ret_4h": future_return,
        "turnover": 1000.0,
        "btc_ret_1h": btc_ret_4h / 4,
        "btc_ret_4h": btc_ret_4h,
        "btc_volatility_4h": 0.03,
        "btc_volatility_percentile": 50.0,
        "ret_4h": ret_4h,
    }
    for col in [
        "ret_15m",
        "ret_1h",
        "volatility_1h",
        "volatility_4h",
        "volume_z_1h",
        "volume_z_4h",
        "ret_4h_percentile",
        "volume_1h_percentile",
        "volume_4h_percentile",
        "funding_z",
        "funding_percentile",
        "oi_value_delta_z_1h",
        "oi_value_delta_z_4h",
        "oi_value_delta_1h_percentile",
        "oi_value_delta_4h_percentile",
    ]:
        row[col] = ret_4h
    return row


def test_prepare_v98_constructs_asof_beta_neutral_target() -> None:
    rows = []
    observations = [0.01, 0.02, -0.01, 0.03, 0.02, -0.02, 0.01, 0.04, -0.01]
    for hour, btc_return in enumerate(observations):
        timestamp = f"2025-07-01 {hour:02d}:00Z"
        rows.append(_row(timestamp, "BTCUSDT", btc_return, 0.02, btc_return))
        rows.append(_row(timestamp, "AAAUSDT", 2 * btc_return, 0.05, btc_return))
    cfg = V98Config(
        sample_start="2025-07-01",
        sample_end="2025-07-02",
        beta_window_hours=4,
        beta_min_obs=3,
        min_cross_section=2,
    )
    data = prepare_v98_dataset(pd.DataFrame(rows), cfg)

    anchor = data[data["feature_time"].eq(pd.Timestamp("2025-07-01 00:00Z"))]
    assert anchor.empty
    last_available = data.sort_values("feature_time").groupby("symbol").tail(1)
    assert last_available.set_index("symbol").loc["AAAUSDT", "beta_30d"] == pytest.approx(2.0)
    assert last_available["target_residual_relative"].sum() == pytest.approx(0.0)


def test_rank_band_retains_incumbent_inside_exit_band() -> None:
    selected, ages = _select_ranked(
        ["A", "B", "C", "D"],
        {},
        top_k=2,
        exit_size=4,
        min_hold=1,
    )
    assert selected == ["A", "B"]

    selected, ages = _select_ranked(
        ["C", "A", "B", "D"],
        ages,
        top_k=2,
        exit_size=4,
        min_hold=1,
    )
    assert selected == ["A", "B"]
    assert ages == {"A": 2, "B": 2}


def test_long_short_ledger_normalizes_spread_and_charges_initial_entry() -> None:
    rows = []
    for score in range(1, 11):
        rows.append(
            {
                "model": "ridge_residual",
                "feature_time": pd.Timestamp("2026-01-01 00:00Z"),
                "entry_month": "2026-01",
                "period": "development",
                "symbol": f"S{score:02d}",
                "score": float(score),
                "future_residual": score / 100.0,
                "target_residual_relative": (score - 5.5) / 100.0,
                "beta_30d": 1.0,
            }
        )
    ledger, _ = build_v98_portfolio_ledger(pd.DataFrame(rows), V98Config())
    result = ledger[
        ledger["book"].eq("long_short_5x5")
        & ledger["policy"].eq("refresh")
    ].iloc[0]

    assert result["gross_excess"] == pytest.approx(0.025)
    assert result["turnover"] == pytest.approx(1.0)
    assert result["net_excess_20"] == pytest.approx(0.023)
