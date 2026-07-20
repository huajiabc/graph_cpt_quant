from __future__ import annotations

import pandas as pd
import pytest

from pressure_graph.reports.v99_sparse_cost_leverage import (
    V99Config,
    _compound_metrics,
    _select_sparse_side,
    simulate_sparse_policy,
)


def test_sparse_side_retains_positive_incumbent_and_requires_replacement_edge() -> None:
    previous = {"A", "B"}
    scores = {"A": 0.0001, "B": 0.0002, "C": 0.0010, "D": -0.0010}
    selected = _select_sparse_side(
        scores,
        previous,
        "long",
        entry_hurdle=0.0005,
        replacement_hurdle=0.0020,
        top_k=2,
    )
    assert selected == previous

    scores["C"] = 0.0023
    selected = _select_sparse_side(
        scores,
        previous,
        "long",
        entry_hurdle=0.0005,
        replacement_hurdle=0.0020,
        top_k=2,
    )
    assert selected == {"B", "C"}


def test_sparse_ledger_leaves_unfilled_slots_in_cash() -> None:
    rows = []
    for score, symbol in [(0.001, "A"), (0.0001, "B"), (-0.001, "C")]:
        rows.append(
            {
                "model": "ridge_residual",
                "feature_time": pd.Timestamp("2026-03-01 00:00Z"),
                "entry_month": "2026-03",
                "period": "validation",
                "symbol": symbol,
                "score": score,
                "target_residual_relative": score,
            }
        )
    ledger = simulate_sparse_policy(
        pd.DataFrame(rows),
        "ridge_residual",
        "long_sparse",
        hurdle_bps=5.0,
        cfg=V99Config(top_k=5),
    )
    result = ledger.iloc[0]

    assert result["long_count"] == 1
    assert result["gross_exposure"] == pytest.approx(0.2)
    assert result["turnover"] == pytest.approx(0.2)
    assert result["gross_excess"] == pytest.approx(0.0002)
    assert result["net_excess_20"] == pytest.approx(-0.0002)


def test_compound_metrics_marks_ruin_without_clipping_tail() -> None:
    metrics = _compound_metrics(pd.Series([0.10, -1.05, 0.50]), 365.0)

    assert metrics["ruin"] is True
    assert metrics["equity_multiple"] == 0.0
    assert metrics["max_drawdown"] == pytest.approx(-1.0)
    assert metrics["worst_period"] == pytest.approx(-1.05)
