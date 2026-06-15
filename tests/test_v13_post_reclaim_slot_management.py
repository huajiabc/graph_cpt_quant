from __future__ import annotations

import pandas as pd

from pressure_graph.reports.v07d1 import _signal_id
from pressure_graph.reports.v13_post_reclaim_slot_management import (
    V13Config,
    write_v13_post_reclaim_slot_management_from_trades,
)


def _trades_prices_orderflow() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    trade_rows = []
    price_rows = []
    orderflow_rows = []
    for idx in range(12):
        early = idx < 8
        entry = base + pd.Timedelta(minutes=5 * idx if early else 70 + 5 * (idx - 8))
        signal = entry - pd.Timedelta(minutes=30)
        symbol = f"S{idx:02d}USDT"
        candidate = "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad"
        net = 0.004 if early else 0.05
        checkpoint_close = 99.0 if early else 105.0
        trade_rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "candidate": candidate,
                "candidate_role": "primary_extreme" if candidate.startswith("CIC1") else "secondary_broad",
                "base_signal_id": f"{signal.isoformat()}|{symbol}",
                "signal_time": signal,
                "entry_time": entry,
                "exit_time": entry + pd.Timedelta(hours=4),
                "entry_price": 100.0,
                "exit_price": 100.0 * (1.0 + net + 0.004),
                "cost_single_side_bps": 20.0,
                "net_return": net,
                "month": "2026-01",
                "rank_first_come_first_served": 0.0,
            }
        )
        price_rows.append(
            {
                "exchange": "bybit",
                "symbol": symbol,
                "feature_time": entry + pd.Timedelta(hours=1),
                "close": checkpoint_close,
            }
        )
    trades = pd.DataFrame(trade_rows)
    trades["signal_id"] = _signal_id(trades)
    for row in trades.itertuples(index=False):
        weak = str(row.symbol)[1:3] < "08"
        orderflow_rows.append(
            {
                "signal_id": row.signal_id,
                "mapping_status": "mapped",
                "post_entry_1h_covered": True,
                "post_entry_1h_coverage_ratio": 1.0,
                "post_entry_1h_turnover": 1_000.0,
                "post_entry_1h_buy_sell_imbalance": -0.2 if weak else 0.4,
                "post_entry_1h_taker_buy_ratio": 0.42 if weak else 0.65,
                "post_entry_1h_cvd_delta_turnover": -200.0 if weak else 400.0,
            }
        )
    return trades, pd.DataFrame(price_rows), pd.DataFrame(orderflow_rows)


def test_checkpoint_rule_can_release_weak_slots_for_later_candidates(tmp_path) -> None:
    trades, prices, orderflow = _trades_prices_orderflow()
    orderflow_path = tmp_path / "event_orderflow.parquet"
    orderflow.to_parquet(orderflow_path, index=False)

    outputs = write_v13_post_reclaim_slot_management_from_trades(
        trades,
        prices,
        V13Config(report_root=tmp_path / "report", event_orderflow_path=orderflow_path),
    )

    summary = pd.read_csv(outputs["checkpoint_rule_summary"])
    baseline = summary[
        summary["rule_id"].eq("baseline_hold") & summary["max_positions"].eq(5)
    ].iloc[0]
    checkpoint = summary[
        summary["rule_id"].eq("exit_if_checkpoint_net_lte_0") & summary["max_positions"].eq(5)
    ].iloc[0]

    assert checkpoint["early_exit_trades"] > 0
    assert checkpoint["newly_selected_vs_baseline"] > 0
    assert checkpoint["portfolio_net20"] > baseline["portfolio_net20"]
