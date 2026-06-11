from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v09e import write_v09e_orderbook_capacity_ranking


def _trade(idx: int, entry: str, net20: float) -> dict[str, object]:
    return {
        "trade_id": f"t{idx}",
        "signal_id": f"s{idx}",
        "candidate": "CIC1_FILTERED_MIR1" if idx % 2 == 0 else "CIC2_FILTERED_MIR1",
        "candidate_role": "primary",
        "symbol": "AAAUSDT",
        "entry_time": entry,
        "exit_time": pd.Timestamp(entry) + pd.Timedelta(hours=2),
        "net_return_20bp": net20,
    }


def _feature(symbol: str, snapshot_time: str, spread: float, imbalance: float) -> dict[str, object]:
    row = {
        "exchange": "bybit",
        "symbol": symbol,
        "snapshot_time": snapshot_time,
        "exchange_ts": snapshot_time,
        "update_id": 1,
        "seq": 1,
        "best_bid": 99.0,
        "best_ask": 100.0,
        "mid": 99.5,
        "spread_bps": spread,
        "bid_levels_available": 200,
        "ask_levels_available": 200,
        "levels_available": 400,
        "top5_bid_notional": 1000.0,
        "top5_ask_notional": 500.0,
        "top5_imbalance": 0.333,
        "ask_wall_20bp_ratio": 0.2,
        "bid_wall_20bp_ratio": 0.1,
        "buy_impact_10k": 1.0,
        "buy_impact_50k": 2.0,
        "buy_impact_100k": 3.0,
        "sell_impact_10k": 1.0,
        "sell_impact_50k": 2.0,
        "sell_impact_100k": 3.0,
        "upside_vacuum_25bp": 0.5,
        "downside_liquidity_risk_25bp": -0.5,
        "entry_book_quality_score": 0.5,
    }
    for bps in [5, 10, 20, 25, 50]:
        row[f"bid_depth_{bps}bp"] = 1000.0
        row[f"ask_depth_{bps}bp"] = 500.0
        row[f"imbalance_{bps}bp"] = imbalance
    return row


def test_v09e_does_not_use_future_orderbook_snapshot(tmp_path) -> None:
    source = tmp_path / "source"
    orderbook = tmp_path / "orderbook"
    report = tmp_path / "report"
    source.mkdir()
    (orderbook / "features").mkdir(parents=True)
    pd.DataFrame([_trade(0, "2026-06-11 00:00:00Z", 0.01)]).to_csv(
        source / "orderflow_shadow_trades.csv", index=False
    )
    pd.DataFrame([_feature("AAAUSDT", "2026-06-11 00:01:00Z", 1.0, 0.5)]).to_parquet(
        orderbook / "features" / "AAAUSDT.parquet", index=False
    )

    outputs = write_v09e_orderbook_capacity_ranking(source, orderbook, report)

    coverage = pd.read_csv(outputs["orderbook_feature_coverage"])
    assert int(coverage["orderbook_covered_trades"].sum()) == 0
    audit = pd.read_csv(outputs["orderbook_coverage_audit"])
    assert set(audit["coverage_status"]) == {"future_only_snapshot"}
    notes = outputs["candidate_notes"].read_text(encoding="utf-8")
    assert "insufficient_orderbook_coverage" in notes


def test_v09e_writes_ranking_when_pre_entry_snapshot_exists(tmp_path) -> None:
    source = tmp_path / "source"
    orderbook = tmp_path / "orderbook"
    report = tmp_path / "report"
    source.mkdir()
    (orderbook / "features").mkdir(parents=True)
    trades = [_trade(i, f"2026-06-11 0{i}:00:00Z", 0.02 if i % 2 == 0 else -0.01) for i in range(6)]
    pd.DataFrame(trades).to_csv(source / "orderflow_shadow_trades.csv", index=False)
    features = [
        _feature(
            "AAAUSDT",
            str(pd.Timestamp(f"2026-06-11 0{i}:00:00Z") - pd.Timedelta(minutes=5)),
            spread=1.0 + i,
            imbalance=0.1 * i,
        )
        for i in range(6)
    ]
    pd.DataFrame(features).to_parquet(orderbook / "features" / "AAAUSDT.parquet", index=False)

    outputs = write_v09e_orderbook_capacity_ranking(source, orderbook, report, max_staleness_minutes=15)

    ranking = pd.read_csv(outputs["orderbook_ranking_summary"])
    assert {"R1_spread_low", "R7_book_quality_composite"}.issubset(set(ranking["ranking_rule"]))
    assert (ranking["selected_trades"] > 0).any()
    audit = pd.read_csv(outputs["orderbook_coverage_audit"])
    assert set(audit["coverage_status"]) == {"covered"}
