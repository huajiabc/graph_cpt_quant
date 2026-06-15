from __future__ import annotations

import numpy as np
import pandas as pd

from pressure_graph.reports.v07d1 import _signal_id
from pressure_graph.reports.v12_reclaim_microstructure_quality import (
    V12Config,
    write_v12_reclaim_microstructure_quality_from_trades,
)


def _sample_trades() -> pd.DataFrame:
    base = pd.Timestamp("2025-07-01T00:00:00Z")
    rows = []
    idx = 0
    for month in range(6):
        month_base = base + pd.DateOffset(months=month)
        for burst in range(2):
            burst_base = month_base + pd.Timedelta(days=burst * 3)
            for slot in range(6):
                signal_time = burst_base + pd.Timedelta(minutes=slot * 5)
                entry_time = signal_time + pd.Timedelta(minutes=30)
                symbol = f"S{slot:02d}USDT"
                candidate = "CIC1_beta_extreme" if slot % 2 == 0 else "CIC2_beta_broad"
                rows.append(
                    {
                        "row_id": idx,
                        "exchange": "bybit",
                        "symbol": symbol,
                        "candidate": candidate,
                        "base_signal_id": f"{signal_time.isoformat()}|{symbol}",
                        "signal_time": signal_time,
                        "entry_time": entry_time,
                        "exit_time": entry_time + pd.Timedelta(hours=4),
                        "cost_single_side_bps": 20.0,
                        "net_return": -0.012 + slot * 0.006 + burst * 0.001,
                        "month": signal_time.strftime("%Y-%m"),
                        "exit_reason": "tp" if slot >= 4 else "timeout",
                    }
                )
                idx += 1
    out = pd.DataFrame(rows)
    out["signal_id"] = _signal_id(out)
    return out


def _sample_orderflow(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in trades.itertuples(index=False):
        slot = int(str(row.symbol)[1:3])
        strength = 0.30 + slot * 0.10
        payload = {
            "signal_id": row.signal_id,
            "exchange": row.exchange,
            "symbol": row.symbol,
            "candidate": row.candidate,
            "signal_time": row.signal_time,
            "entry_time": row.entry_time,
            "mapping_status": "mapped",
        }
        for window in (
            "shock_bar",
            "pullback_window",
            "reclaim_bar",
            "pre_entry_all",
            "entry_bar",
            "post_entry_1h",
        ):
            payload[f"{window}_covered"] = True
            payload[f"{window}_coverage_ratio"] = 1.0
            payload[f"{window}_turnover"] = 1_000.0 + slot * 100.0
            payload[f"{window}_trade_count"] = 20 + slot
            payload[f"{window}_taker_buy_ratio"] = strength if "reclaim" in window else 0.45 + slot * 0.03
            payload[f"{window}_buy_sell_imbalance"] = (strength - 0.5) if "reclaim" in window else -0.10 + slot * 0.04
            payload[f"{window}_cvd_delta_turnover"] = payload[f"{window}_buy_sell_imbalance"] * payload[f"{window}_turnover"]
            payload[f"{window}_large_buy_turnover"] = strength * 200.0
            payload[f"{window}_large_sell_turnover"] = (1.0 - strength) * 120.0
        rows.append(payload)
    return pd.DataFrame(rows)


def test_v12_reclaim_microstructure_report_detects_within_burst_quality(tmp_path) -> None:
    trades = _sample_trades()
    orderflow_path = tmp_path / "event_orderflow.parquet"
    _sample_orderflow(trades).to_parquet(orderflow_path, index=False)

    outputs = write_v12_reclaim_microstructure_quality_from_trades(
        trades,
        V12Config(report_root=tmp_path / "report", event_orderflow_path=orderflow_path),
    )

    coverage = pd.read_csv(outputs["micro_coverage_summary"])
    features = pd.read_csv(outputs["micro_feature_summary"])
    pairwise = pd.read_csv(outputs["micro_pairwise_winrate"])
    post = pd.read_csv(outputs["post_reclaim_followthrough_diagnostic"])

    assert coverage["pre_entry_covered_rate"].min() == 1.0
    reclaim_ratio = features[features["feature"].eq("micro_reclaim_taker_buy_ratio")].iloc[0]
    assert reclaim_ratio["q5_minus_q1_net20"] > 0.02
    assert reclaim_ratio["within_burst_mean_ic"] > 0.9
    reclaim_pairwise = pairwise[pairwise["feature"].eq("micro_reclaim_taker_buy_ratio")].iloc[0]
    assert reclaim_pairwise["higher_feature_win_rate"] > 0.95
    assert bool(reclaim_pairwise["passes_55pct_threshold"])
    assert np.isfinite(post["global_spearman_ic"]).any()
