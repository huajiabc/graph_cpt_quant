from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09b import RANKING_RULES, RANKING_SCORE_COLUMNS, select_portfolio


def test_portfolio_ranking_prefers_higher_score_same_fill_time() -> None:
    entry = pd.Timestamp("2026-01-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=1),
                "net_return": -0.02,
                "score_col": 1.0,
            },
            {
                "symbol": "BBBUSDT",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=1),
                "net_return": 0.05,
                "score_col": 5.0,
            },
        ]
    )

    selected, skipped = select_portfolio(trades, score_col="score_col", max_positions=1)

    assert selected["symbol"].tolist() == ["BBBUSDT"]
    assert skipped["symbol"].tolist() == ["AAAUSDT"]
    assert skipped["skip_reason"].tolist() == ["portfolio_full"]


def test_portfolio_ranking_blocks_same_symbol_until_exit() -> None:
    entry = pd.Timestamp("2026-01-01T00:00:00Z")
    trades = pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=2),
                "net_return": 0.03,
                "score_col": 5.0,
            },
            {
                "symbol": "AAAUSDT",
                "entry_time": entry + pd.Timedelta(minutes=15),
                "exit_time": entry + pd.Timedelta(hours=3),
                "net_return": 0.04,
                "score_col": 9.0,
            },
        ]
    )

    selected, skipped = select_portfolio(trades, score_col="score_col", max_positions=3)

    assert selected["symbol"].tolist() == ["AAAUSDT"]
    assert skipped["skip_reason"].tolist() == ["symbol_already_active"]


def test_all_declared_ranking_rules_have_score_columns() -> None:
    assert set(RANKING_RULES) == set(RANKING_SCORE_COLUMNS)
    assert all(value.startswith("rank_") for value in RANKING_SCORE_COLUMNS.values())
