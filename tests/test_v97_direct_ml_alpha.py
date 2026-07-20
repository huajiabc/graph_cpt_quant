from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v97_direct_ml_alpha import (
    V97Config,
    build_portfolio_ledger,
    prepare_v97_dataset,
)


def _source_row(timestamp: str, symbol: str, ret: float, score: float) -> dict[str, object]:
    row: dict[str, object] = {
        "symbol": symbol,
        "feature_time": pd.Timestamp(timestamp),
        "universe_dynamic_monthly_top30": True,
        "warmup_complete": True,
        "future_ret_4h": ret,
        "turnover": 1000.0,
        "btc_ret_1h": 0.01,
        "btc_ret_4h": 0.02,
        "btc_volatility_4h": 0.03,
        "btc_volatility_percentile": 0.5,
    }
    for col in [
        "ret_15m",
        "ret_1h",
        "ret_4h",
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
        row[col] = score
    return row


def test_prepare_dataset_uses_nonoverlapping_dynamic_cross_sections() -> None:
    rows = [
        _source_row("2025-07-01 04:00Z", "AAA", 0.03, 1.0),
        _source_row("2025-07-01 04:00Z", "BBB", 0.01, 2.0),
        _source_row("2025-07-01 05:00Z", "AAA", 0.50, 9.0),
    ]
    data = prepare_v97_dataset(pd.DataFrame(rows), V97Config(min_cross_section=2))

    assert len(data) == 2
    assert data["feature_time"].nunique() == 1
    assert data["target_relative"].sum() == pytest.approx(0.0)
    assert data.sort_values("symbol")["xrank_ret_4h"].tolist() == [0.0, 0.5]


def test_portfolio_turnover_tracks_replaced_names() -> None:
    rows = []
    for timestamp, scores in [
        ("2026-01-01 00:00Z", {"A": 4, "B": 3, "C": 2}),
        ("2026-01-01 04:00Z", {"A": 4, "C": 3, "B": 2}),
    ]:
        for symbol, score in scores.items():
            rows.append(
                {
                    "model": "ridge",
                    "feature_time": pd.Timestamp(timestamp),
                    "entry_month": "2026-01",
                    "period": "development",
                    "symbol": symbol,
                    "score": score,
                    "future_return": 0.01 if symbol == "A" else 0.0,
                    "target_relative": 0.01 if symbol == "A" else -0.005,
                }
            )
    ledger, _ = build_portfolio_ledger(pd.DataFrame(rows), top_k=2, costs_bps=(20,))

    assert ledger["turnover"].tolist() == [1.0, 0.5]
    assert ledger.iloc[1]["cost_20"] == pytest.approx(0.001)
