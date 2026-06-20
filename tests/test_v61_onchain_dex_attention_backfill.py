from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config, write_v61_onchain_dex_attention_backfill


def _write_json_cache(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    chart = []
    for idx in range(80):
        ts = int((base + pd.Timedelta(days=idx)).timestamp())
        value = 1000.0 + idx * 2.0
        if idx in {45, 60}:
            value = 3000.0 + idx * 10.0
        chart.append([ts, value])
    for name in ["defillama_dexs_overview.json", "defillama_fees_overview.json"]:
        (cache / name).write_text(pd.Series({"totalDataChart": chart}).to_json(), encoding="utf-8")
    stable = []
    for idx in range(80):
        ts = int((base + pd.Timedelta(days=idx)).timestamp())
        value = 10_000.0 + idx * 10.0
        if idx == 50:
            value = 40_000.0
        stable.append({"date": str(ts), "totalCirculatingUSD": {"peggedUSD": value}})
    (cache / "defillama_stablecoincharts_all.json").write_text(pd.Series(stable).to_json(), encoding="utf-8")


def _write_features(path: Path) -> None:
    rows = []
    times = pd.date_range("2026-02-01", periods=500, freq="15min", tz="UTC")
    for symbol in ["AAAUSDT", "BBBUSDT", "BTCUSDT"]:
        for idx, ts in enumerate(times):
            rows.append(
                {
                    "symbol": symbol,
                    "feature_time": ts,
                    "warmup_complete": True,
                    "universe_dynamic_monthly_top30": True,
                    "ret_15m": 0.001,
                    "ret_1h": 0.006 if idx % 97 == 0 else 0.001,
                    "ret_4h": 0.01,
                    "volume_z_1h": 2.5 if idx % 97 == 0 else 0.2,
                    "volume_z_4h": 1.0,
                    "future_ret_4h": 0.01,
                    "future_ret_12h": 0.02,
                    "btc_market_state": "btc_up",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_trades(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": "2026-02-16 01:00:00+00:00",
                "net_return": 0.02,
            }
        ]
    ).to_csv(path, index=False)


def test_v61_onchain_dex_attention_backfill_uses_cached_sources(tmp_path: Path) -> None:
    cache = tmp_path / "defillama"
    _write_json_cache(cache)
    feature_path = tmp_path / "features.parquet"
    trade_path = tmp_path / "trades.csv"
    _write_features(feature_path)
    _write_trades(trade_path)

    outputs = write_v61_onchain_dex_attention_backfill(
        V61Config(
            report_root=tmp_path / "reports" / "v61",
            feature_path=feature_path,
            trade_cache_path=trade_path,
            cache_root=cache,
            allow_network=False,
            min_lookback_days=10,
            lookback_days=20,
        )
    )

    assert outputs["onchain_attention_events"].exists()
    events = pd.read_csv(outputs["onchain_attention_events"])
    assert not events.empty
    response = pd.read_csv(outputs["onchain_to_cex_response_curve"])
    assert "max_cex_volume_shock_rate" in response.columns
    fusion = pd.read_csv(outputs["onchain_cic_fusion_summary"])
    assert "bucket" in fusion.columns

