from pathlib import Path

import numpy as np
import pandas as pd

from pressure_graph.paper_live.cm2 import (
    CM2LiveConfig,
    build_tg1_decision_input,
    build_tg1_weights,
)


def _cfg(tmp_path: Path) -> CM2LiveConfig:
    return CM2LiveConfig(
        base_config=Path("configs/v0_3.yaml"),
        live_root=tmp_path / "live",
        membership_path=tmp_path / "membership.csv",
        tg1_seed_portfolio_path=tmp_path / "seed.parquet",
        fss3_report_root=tmp_path / "fss3",
        report_root=tmp_path / "report",
    )


def test_tg1_hold_band_reproduces_saved_last_transition(tmp_path: Path) -> None:
    panel = pd.read_parquet(
        "reports/v13_2_tg1_forward_temporal_extension/weekly_symbol_panel.parquet"
    )
    panel["entry_time"] = pd.to_datetime(panel["entry_time"], utc=True)
    portfolio = pd.read_parquet(
        "reports/v13_2_tg1_forward_temporal_extension/weekly_portfolio.parquet"
    )
    portfolio["entry_time"] = pd.to_datetime(portfolio["entry_time"], utc=True)
    current = portfolio["entry_time"].max()
    previous = current - pd.Timedelta(days=7)
    previous_symbols = (
        portfolio.loc[
            portfolio["entry_time"].eq(previous), "selected_symbols"
        ]
        .iloc[0]
        .split("|")
    )
    expected = (
        portfolio.loc[
            portfolio["entry_time"].eq(current), "selected_symbols"
        ]
        .iloc[0]
        .split("|")
    )
    selected, weights, details = build_tg1_weights(
        panel[panel["entry_time"].eq(current)],
        previous_symbols,
        _cfg(tmp_path),
    )
    assert selected == expected
    # The saved research row adds a terminal 1.0 close because that sample ends
    # on this week. A continuous forward state correctly records only the one
    # name replacement: one ninth out plus one ninth in.
    assert np.isclose(details["rebalance_turnover"], 2 / 9)
    assert np.isclose(sum(weights.values()), 1.0)


def test_tg1_decision_input_excludes_decision_time_funding(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    decision = pd.Timestamp("2026-07-20", tz="UTC")
    symbols = [f"S{index}" for index in range(10)]
    membership = pd.DataFrame(
        {
            "month_start": pd.Timestamp("2026-07-01", tz="UTC"),
            "symbol": symbols,
        }
    )
    prices = pd.DataFrame(
        {
            "symbol": symbols,
            "feature_time": decision,
            "close": np.arange(10) + 10.0,
        }
    )
    binance_prices = prices.rename(
        columns={"close": "binance_close"}
    )
    funding_rows = []
    for index, symbol in enumerate(symbols):
        funding_rows.extend(
            [
                {
                    "symbol": symbol,
                    "funding_time": decision - pd.Timedelta(hours=8),
                    "funding_rate_settled": 0.001 + index * 0.0001,
                },
                {
                    "symbol": symbol,
                    "funding_time": decision,
                    "funding_rate_settled": 100.0,
                },
            ]
        )
    bybit = pd.DataFrame(funding_rows)
    binance = bybit.copy()
    binance.loc[
        binance["funding_time"].lt(decision), "funding_rate_settled"
    ] += 0.01
    local, metadata = build_tg1_decision_input(
        bybit,
        binance,
        prices,
        binance_prices,
        membership,
        decision,
        cfg,
    )
    assert metadata["reasons"] == ()
    assert len(local) == 10
    assert np.allclose(local["score_30d"], 0.01)


def test_cm2_config_has_no_order_or_leverage_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.scope == "live_shadow"
    assert cfg.push_policy == "record_only"
    assert not cfg.real_orders_allowed
    assert not cfg.leverage_allowed
    assert cfg.fss3_weight == 0.80
    assert cfg.tg1_weight == 0.20
