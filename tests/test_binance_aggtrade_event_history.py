import zipfile

import numpy as np
import pandas as pd

from pressure_graph.binance_aggtrade_event_history import (
    build_extreme_overshoot_tasks,
    load_aggtrades_zip_windows,
    normalize_rest_aggtrades,
    summarize_aggtrade_window,
)


def test_normalize_rest_aggtrades_maps_buyer_maker_to_taker_sell() -> None:
    payload = [
        {"a": 1, "p": "100", "q": "2", "T": 1_767_225_600_000, "m": True},
        {"a": 2, "p": "101", "q": "1", "T": 1_767_225_601_000, "m": False},
    ]
    frame = normalize_rest_aggtrades(payload)
    assert frame["buyer_maker"].tolist() == [True, False]
    assert frame["agg_trade_id"].tolist() == [1, 2]


def test_summary_detects_aligned_flow_exhaustion() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    times = [
        start + pd.Timedelta(minutes=1),
        start + pd.Timedelta(minutes=6),
        start + pd.Timedelta(minutes=11),
    ]
    trades = pd.DataFrame(
        {
            "agg_trade_id": [1, 2, 3],
            "price": [100.0, 101.0, 102.0],
            "quantity": [10.0, 10.0, 10.0],
            "timestamp": times,
            "buyer_maker": [False, False, True],
        }
    )
    summary = summarize_aggtrade_window(
        trades, start, start + pd.Timedelta(minutes=15), source_sign=1.0
    )
    assert summary["early_aligned_imbalance"] == 1.0
    assert summary["late_aligned_imbalance"] == -1.0
    assert summary["aligned_flow_exhaustion"] == 2.0
    assert summary["late_flow_opposes_source"]
    assert np.isclose(summary["aligned_price_return"], 0.02)


def test_summary_accepts_normalized_archive_schema() -> None:
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    trades = pd.DataFrame(
        {
            "price": [100.0, 101.0],
            "size": [2.0, 3.0],
            "turnover": [200.0, 303.0],
            "timestamp": [
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=14),
            ],
            "side": ["buy", "sell"],
        }
    )
    summary = summarize_aggtrade_window(
        trades, start, start + pd.Timedelta(minutes=15), source_sign=1.0
    )
    assert summary["trade_count"] == 2
    assert summary["buy_turnover"] == 200.0
    assert summary["sell_turnover"] == 303.0


def test_task_builder_freezes_extreme_community_trade_events() -> None:
    frame = pd.DataFrame(
        {
            "source_scope": ["COMMUNITY_COHERENT_INDEX_SHOCK"] * 2,
            "family": ["TRADE_VS_MARK_OVERSHOOT_FADE"] * 2,
            "source_setting": ["z2.0", "z2.0"],
            "receiver_z_threshold": [2.0, 1.5],
            "receiver_count": [3, 3],
            "receivers": ["A|B|C", "D|E|F"],
            "source_event_id": ["E1", "E2"],
            "feature_time": pd.to_datetime(
                ["2026-01-01 00:15Z", "2026-01-01 00:30Z"]
            ),
            "period": ["validation", "validation"],
            "community_id": ["C1", "C1"],
            "source_sign": [1.0, -1.0],
        }
    )
    tasks = build_extreme_overshoot_tasks(frame)
    assert len(tasks) == 3
    assert set(tasks["symbol"]) == {"A", "B", "C"}
    assert tasks["window_end"].sub(tasks["window_start"]).eq(
        pd.Timedelta(minutes=15)
    ).all()


def test_streaming_zip_loader_keeps_only_requested_window(tmp_path) -> None:
    path = tmp_path / "trades.zip"
    rows = [
        "1,100,1,1,1,1767225600000,true",
        "2,101,1,2,2,1767225660000,false",
        "3,102,1,3,3,1767226500000,true",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("trades.csv", "\n".join(rows))
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    loaded = load_aggtrades_zip_windows(
        path, [(start, start + pd.Timedelta(minutes=2))], chunksize=1
    )
    assert len(loaded) == 2
    assert loaded["execId"].tolist() == ["1", "2"]
