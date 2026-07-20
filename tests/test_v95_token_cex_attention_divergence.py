from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v95_token_cex_attention_divergence import (
    TAD1,
    V95Config,
    _add_candidate_fields,
    _deoverlap_events,
    write_v95_token_cex_attention_divergence,
)


def test_deoverlap_and_frozen_candidate_predicate() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["a", "b", "c"],
            "cex_symbol": ["AAAUSDT"] * 3,
            "event_available_time": pd.to_datetime(
                ["2026-01-01 00:00Z", "2026-01-01 12:00Z", "2026-01-02 00:00Z"], utc=True
            ),
        }
    )
    kept = _deoverlap_events(events, 24)
    assert kept["event_id"].tolist() == ["a", "c"]

    frame = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-01-01 00:15Z", "2026-01-01 00:30Z"], utc=True),
            "ret_15m": [0.001, -0.001],
            "ret_1h": [0.002, -0.002],
            "ret_4h": [0.005, 0.005],
            "volume_z_1h": [0.5, 0.5],
            "future_ret_12h": [0.02, -0.01],
        }
    )
    scored = _add_candidate_fields(frame, V95Config(bootstrap_trials=10))
    assert scored[TAD1].tolist() == [True, False]
    assert scored["net20_12h"].round(6).tolist() == [0.016, -0.014]


def _feature_rows() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2025-12-20", "2025-12-31", freq="D", tz="UTC"):
        rows.extend(
            [
                {
                    "exchange": "bybit",
                    "symbol": "AAAUSDT",
                    "feature_time": day,
                    "turnover": 2_000_000.0,
                    "warmup_complete": True,
                    "ret_15m": 0.0,
                    "ret_1h": 0.0,
                    "ret_4h": 0.0,
                    "volume_z_1h": 0.0,
                    "future_ret_12h": 0.0,
                },
                {
                    "exchange": "bybit",
                    "symbol": "BBBUSDT",
                    "feature_time": day,
                    "turnover": 1_000_000.0,
                    "warmup_complete": True,
                    "ret_15m": 0.0,
                    "ret_1h": 0.0,
                    "ret_4h": 0.0,
                    "volume_z_1h": 0.0,
                    "future_ret_12h": 0.0,
                },
            ]
        )
    rows.append(
        {
            "exchange": "bybit",
            "symbol": "AAAUSDT",
            "feature_time": pd.Timestamp("2026-01-01 01:15Z"),
            "turnover": 2_000_000.0,
            "warmup_complete": True,
            "ret_15m": 0.001,
            "ret_1h": 0.002,
            "ret_4h": 0.005,
            "volume_z_1h": 0.5,
            "future_ret_12h": 0.02,
        }
    )
    return pd.DataFrame(rows)


def test_v95_writes_a_strictly_timed_top50_ledger(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    _feature_rows().to_parquet(feature_path, index=False)
    event_path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "cex_symbol": "AAAUSDT",
                "chain": "eth",
                "event_time": "2026-01-01 00:00:00+00:00",
                "event_available_time": "2026-01-01 01:05:00+00:00",
                "source": "dexpaprika_pool_ohlcv_1h",
                "mapping_confidence": "B",
            }
        ]
    ).to_csv(event_path, index=False)
    cfg = V95Config(
        report_root=tmp_path / "report",
        feature_path=feature_path,
        event_path=event_path,
        p2_trade_path=tmp_path / "missing.csv",
        universe_top_n=1,
        bootstrap_trials=10,
        random_trials=2,
        random_token_trials=2,
    )
    outputs = write_v95_token_cex_attention_divergence(cfg)
    ledger = pd.read_csv(outputs["event_entry_ledger"])
    assert len(ledger) == 1
    assert bool(ledger[TAD1].iloc[0])
    assert float(ledger["entry_delay_minutes"].iloc[0]) == 10.0
    assert outputs["decision_table"].exists()
