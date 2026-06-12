from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pressure_graph.orderflow_history import PRE_ENTRY_WINDOWS
from pressure_graph.reports.v07d1 import _signal_id
from pressure_graph.reports.v11_orderflow_burst_ranking import (
    ALL_RANK_COLUMNS,
    RANK_FEATURES,
    V11Config,
    write_v11_orderflow_burst_ranking,
)

COSTS = (5.0, 10.0, 20.0, 30.0, 50.0)


def _synthetic_trades() -> pd.DataFrame:
    rows = []
    trade_idx = 0
    for month in range(1, 7):
        for burst in range(2):
            burst_entry = pd.Timestamp(f"2026-{month:02d}-10 04:00:00", tz="UTC") + pd.Timedelta(
                hours=8 * burst
            )
            for slot in range(12):
                trade_idx += 1
                symbol = f"SYM{trade_idx:03d}USDT"
                signal_time = burst_entry - pd.Timedelta(minutes=45)
                entry_time = burst_entry + pd.Timedelta(minutes=15 * (slot % 2))
                exit_time = entry_time + pd.Timedelta(hours=4)
                gross = 0.001 * (slot - 5)
                rows.append(
                    {
                        "exchange": "bybit",
                        "symbol": symbol,
                        "candidate": "CIC1_beta_extreme" if slot % 2 == 0 else "CIC2_beta_broad",
                        "candidate_role": "primary_extreme",
                        "signal_time": signal_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "gross_return": gross,
                        "funding_cost": 0.0,
                        "holding_minutes": 240.0,
                    }
                )
    base = pd.DataFrame(rows)
    frames = []
    for cost in COSTS:
        frame = base.copy()
        frame["cost_single_side_bps"] = cost
        frame["net_return"] = frame["gross_return"] - 2.0 * cost / 10_000.0
        frames.append(frame)
    trades = pd.concat(frames, ignore_index=True)
    trades["base_signal_id"] = (
        trades["exchange"].astype(str)
        + "|"
        + trades["symbol"].astype(str)
        + "|"
        + trades["signal_time"].astype(str)
    )
    trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")
    trades["rank_first_come_first_served"] = 0.0
    return trades


def _synthetic_orderflow(trades: pd.DataFrame) -> pd.DataFrame:
    events = trades[trades["cost_single_side_bps"].eq(20.0)].copy()
    events["signal_id"] = _signal_id(events)
    events = events.drop_duplicates("signal_id").reset_index(drop=True)
    rng = np.random.default_rng(7)
    rows = []
    for _, event in events.iterrows():
        net = float(event["net_return"])
        row: dict[str, object] = {
            "signal_id": event["signal_id"],
            "binance_symbol": event["symbol"],
            "mapping_status": "mapped",
        }
        for window in ("shock_bar", "pullback_window", "reclaim_bar", "pre_entry_all", "entry_bar", "post_entry_1h"):
            row[f"{window}_covered"] = True
            row[f"{window}_coverage_ratio"] = 1.0
            row[f"{window}_source_quality"] = "complete"
            row[f"{window}_turnover"] = 1_000.0
            row[f"{window}_trade_count"] = 100
            row[f"{window}_large_buy_turnover"] = 50.0
            row[f"{window}_large_sell_turnover"] = 50.0
            # Perfectly informative reclaim feature; others are noise.
            if window == "reclaim_bar":
                row[f"{window}_taker_buy_ratio"] = 0.5 + net * 10.0
                row[f"{window}_buy_sell_imbalance"] = net * 20.0
            else:
                row[f"{window}_taker_buy_ratio"] = float(rng.uniform(0.4, 0.6))
                row[f"{window}_buy_sell_imbalance"] = float(rng.uniform(-0.1, 0.1))
            row[f"{window}_cvd_delta_turnover"] = row[f"{window}_buy_sell_imbalance"]
            row[f"{window}_cvd_delta_volume"] = row[f"{window}_buy_sell_imbalance"]
        rows.append(row)
    return pd.DataFrame(rows)


def test_rank_features_only_use_pre_entry_windows():
    for column, (window, _stat) in RANK_FEATURES.items():
        assert window in PRE_ENTRY_WINDOWS, column
    for column in ALL_RANK_COLUMNS:
        assert "entry_bar" not in column
        assert "post_entry" not in column


def test_v11_report_end_to_end(tmp_path):
    trades = _synthetic_trades()
    orderflow = _synthetic_orderflow(trades)
    orderflow_path = tmp_path / "cic_event_orderflow.parquet"
    orderflow.to_parquet(orderflow_path, index=False)
    cfg = V11Config(report_root=tmp_path / "report", event_orderflow_path=orderflow_path)
    outputs = write_v11_orderflow_burst_ranking(trades, cfg)
    for path in outputs.values():
        assert path.exists(), path

    ic = pd.read_csv(outputs["feature_ic"])
    reclaim_ic = ic[ic["feature"].eq("of_reclaim_taker_buy_ratio")]["within_burst_mean_ic"].iloc[0]
    assert reclaim_ic > 0.9

    replays = pd.read_csv(outputs["counterfactual_replays"])
    focal = replays[
        replays["cost_single_side_bps"].eq(20.0) & replays["max_positions"].astype(str).eq("5")
    ]
    baseline = float(focal[focal["rule"].eq("R0_first_come")]["portfolio_net20"].iloc[0])
    informed = float(
        focal[focal["rule"].eq("R_of_reclaim_taker_buy_ratio")]["portfolio_net20"].iloc[0]
    )
    assert informed > baseline

    winrate = pd.read_csv(outputs["pairwise_winrate"])
    reclaim_wr = winrate[winrate["feature"].eq("of_reclaim_taker_buy_ratio")][
        "higher_feature_win_rate"
    ].iloc[0]
    assert reclaim_wr > 0.95

    walk_forward = pd.read_csv(outputs["walkforward_replay"])
    assert not walk_forward.empty
    wf = walk_forward[
        walk_forward["rule"].eq("R_of_learned_walk_forward")
        & walk_forward["max_positions"].astype(str).eq("5")
    ]
    wf_baseline = walk_forward[
        walk_forward["rule"].eq("R0_first_come_scored_period")
        & walk_forward["max_positions"].astype(str).eq("5")
    ]
    assert float(wf["portfolio_net20"].iloc[0]) > float(wf_baseline["portfolio_net20"].iloc[0])


def test_v11_missing_orderflow_file_raises(tmp_path):
    trades = _synthetic_trades()
    cfg = V11Config(
        report_root=tmp_path / "report",
        event_orderflow_path=tmp_path / "missing.parquet",
    )
    with pytest.raises(FileNotFoundError):
        write_v11_orderflow_burst_ranking(trades, cfg)
