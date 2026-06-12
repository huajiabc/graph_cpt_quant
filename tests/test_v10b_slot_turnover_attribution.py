from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v10b_slot_turnover_attribution import (
    _lookup_mark,
    _oracle_replacement_gap,
    _simulate_baseline,
    _simulate_replacement,
)


def _rows() -> pd.DataFrame:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    specs = [
        ("AAAUSDT", "CIC2_beta_broad", 0, 0.001),
        ("BBBUSDT", "CIC2_beta_broad", 15, -0.010),
        ("CCCUSDT", "CIC1_beta_extreme", 120, 0.050),
        ("DDDUSDT", "CIC1_beta_extreme", 135, 0.040),
    ]
    for idx, (symbol, candidate, minute, net) in enumerate(specs):
        entry = base + pd.Timedelta(minutes=minute)
        rows.append(
            {
                "row_id": idx,
                "exchange": "bybit",
                "symbol": symbol,
                "candidate": candidate,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "entry_price": 100.0,
                "exit_price": 100.0 * (1.0 + net),
                "cost_single_side_bps": 20.0,
                "net_return": net,
                "month": "2026-01",
                "holding_minutes": 240.0,
                "mfe_12h": max(net, 0.0),
                "mae_12h": min(net, 0.0),
                "rank_beta_extreme_strength": float(idx),
                "rank_local_volume_shock_strength": float(idx),
            }
        )
    return pd.DataFrame(rows)


def test_slot_opportunity_cost_records_open_positions_when_full() -> None:
    trades = _rows()

    selected, skipped, opportunity = _simulate_baseline(
        trades,
        max_positions=2,
        mark_table={},
        capture_opportunity=True,
    )

    assert len(selected) == 2
    assert len(skipped) == 2
    assert not opportunity.empty
    assert "opportunity_cost_vs_worst_open" in opportunity.columns
    assert opportunity["open_position_role"].eq("worst_open_position").any()


def test_oracle_and_rule_replacement_create_replacement_ledger() -> None:
    trades = _rows()

    oracle, oracle_ledger = _oracle_replacement_gap(trades, {})
    chosen, skipped, ledger = _simulate_replacement(
        trades,
        max_positions=2,
        rule="R1_replace_stagnant_loser",
        mark_table={},
    )

    assert not oracle.empty
    assert not oracle_ledger.empty
    assert not chosen.empty
    assert not skipped.empty
    assert ledger["ledger_exit_reason"].astype(str).str.contains("R1_replace_stagnant_loser").any()


def test_lookup_mark_uses_latest_asof_utc_timestamp() -> None:
    table = {
        "AAAUSDT": pd.DataFrame(
            {
                "mark_time": pd.to_datetime(
                    ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
                    utc=True,
                ),
                "mark_price": [100.0, 101.0],
            }
        )
    }

    price, source = _lookup_mark(table, "AAAUSDT", pd.Timestamp("2026-01-01T00:16:00Z"))

    assert price == 101.0
    assert source == "feature_close_asof"
