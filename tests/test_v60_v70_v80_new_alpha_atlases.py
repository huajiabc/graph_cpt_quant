from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v60_onchain_attention_graph import V60Config, write_v60_onchain_attention_graph
from pressure_graph.reports.v70_narrative_sector_rotation import V70Config, write_v70_narrative_sector_rotation
from pressure_graph.reports.v80_catalyst_event_alpha import V80Config, write_v80_catalyst_event_alpha


def _feature_rows(symbol: str, future: float, impulse_at: set[int] | None = None) -> list[dict[str, object]]:
    impulse_at = impulse_at or set()
    times = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    rows = []
    for idx, ts in enumerate(times):
        rows.append(
            {
                "symbol": symbol,
                "feature_time": ts,
                "warmup_complete": True,
                "universe_dynamic_monthly_top30": True,
                "ret_15m": 0.003 if idx in impulse_at else 0.0002,
                "ret_1h": 0.008 if idx in impulse_at else 0.001,
                "ret_4h": 0.01,
                "ret_4h_percentile": 50.0,
                "volume_z_1h": 2.2 if idx in impulse_at else 0.4,
                "volume_z_4h": 1.0,
                "future_ret_4h": future / 2.0,
                "future_ret_12h": future,
                "btc_market_state": "btc_up",
            }
        )
    return rows


def _write_feature_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "features.parquet"
    rows = []
    rows.extend(_feature_rows("SOLUSDT", 0.02, {10, 11}))
    rows.extend(_feature_rows("WIFUSDT", 0.03, {15}))
    rows.extend(_feature_rows("DOGEUSDT", -0.01, set()))
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_v60_onchain_attention_graph_with_optional_events(tmp_path: Path) -> None:
    feature_path = _write_feature_fixture(tmp_path)
    event_path = tmp_path / "onchain_attention_events.csv"
    pd.DataFrame(
        [
            {
                "event_time": "2026-01-01 02:30:00+00:00",
                "symbol": "SOLUSDT",
                "event_type": "dex_volume_spike",
                "attention_score": 3.0,
                "source": "fixture",
            }
        ]
    ).to_csv(event_path, index=False)
    inst = tmp_path / "instruments.parquet"
    pd.DataFrame({"symbol": ["SOLUSDT", "WIFUSDT"], "baseCoin": ["SOL", "WIF"], "launch_time": pd.to_datetime(["2025-01-01", "2025-01-01"], utc=True)}).to_parquet(inst)

    outputs = write_v60_onchain_attention_graph(
        V60Config(report_root=tmp_path / "reports" / "v60", feature_path=feature_path, event_path=event_path, bybit_instruments_path=inst)
    )

    assert outputs["onchain_event_schema"].exists()
    follow = pd.read_csv(outputs["cex_followthrough_atlas"])
    assert "dex_volume_spike" in set(follow["bucket"])


def test_v70_narrative_sector_rotation_outputs_controls(tmp_path: Path) -> None:
    feature_path = _write_feature_fixture(tmp_path)
    inst = tmp_path / "binance_instruments.parquet"
    pd.DataFrame(
        {
            "symbol": ["SOLUSDT", "WIFUSDT", "DOGEUSDT"],
            "underlyingSubType": [["Layer-1"], ["Meme"], ["Meme"]],
        }
    ).to_parquet(inst)
    outputs = write_v70_narrative_sector_rotation(
        V70Config(report_root=tmp_path / "reports" / "v70", feature_path=feature_path, binance_instruments_path=inst, trade_cache_path=tmp_path / "missing.csv")
    )

    assert outputs["sector_leader_beta_summary"].exists()
    controls = pd.read_csv(outputs["random_sector_control"])
    assert {"real_narrative", "random_shuffled_narrative"}.issubset(set(controls["control"]))


def _write_kline(path: Path, symbol: str) -> None:
    times = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
    rows = []
    price = 1.0
    for idx, ts in enumerate(times):
        price *= 1.001 if idx < 96 else 0.999
        rows.append(
            {
                "bar_close_time": ts,
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "turnover": 1000.0,
            }
        )
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path / f"{symbol}.parquet", index=False)


def test_v80_catalyst_event_alpha_replays_listing(tmp_path: Path) -> None:
    inst = tmp_path / "instruments.parquet"
    pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "baseCoin": ["AAA"],
            "status": ["Trading"],
            "launch_time": pd.to_datetime(["2026-01-01 00:00:00+00:00"], utc=True),
        }
    ).to_parquet(inst, index=False)
    kline_root = tmp_path / "klines"
    _write_kline(kline_root, "AAAUSDT")

    outputs = write_v80_catalyst_event_alpha(
        V80Config(report_root=tmp_path / "reports" / "v80", bybit_instruments_path=inst, bybit_kline_root=kline_root, external_event_path=tmp_path / "missing.csv")
    )

    response = pd.read_csv(outputs["post_listing_response"])
    assert response["status"].iloc[0] == "ok"
    summary = pd.read_csv(outputs["listing_event_summary"])
    assert int(summary.loc[summary["status"].eq("ok"), "events"].iloc[0]) == 1

