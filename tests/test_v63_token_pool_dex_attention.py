from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v63_token_pool_dex_attention import V63Config, write_v63_token_pool_dex_attention


def _write_features(path: Path) -> None:
    rows = []
    times = pd.date_range("2026-05-01", periods=600, freq="15min", tz="UTC")
    for idx, ts in enumerate(times):
        rows.append(
            {
                "symbol": "AAAUSDT",
                "feature_time": ts,
                "warmup_complete": True,
                "universe_dynamic_monthly_top30": True,
                "ret_15m": 0.001,
                "ret_1h": 0.006 if idx % 80 == 0 else 0.001,
                "ret_4h": 0.01,
                "volume_z_1h": 2.8 if idx % 80 == 0 else 0.2,
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
                "entry_time": "2026-05-06 06:30:00+00:00",
                "exit_time": "2026-05-06 10:30:00+00:00",
                "net_return": 0.02,
                "signal_id": "aaa-1",
            }
        ]
    ).to_csv(path, index=False)


def _write_ohlcv_cache(cache: Path) -> None:
    out = cache / "geckoterminal" / "ohlcv_hour"
    out.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2026-05-01", tz="UTC")
    bars = []
    for idx in range(200):
        ts = int((base + pd.Timedelta(hours=idx)).timestamp())
        volume = 1000.0 + idx
        if idx in {60, 120}:
            volume = 20_000.0
        bars.append([ts, 1.0, 1.1, 0.9, 1.0, volume])
    payload = {"data": {"attributes": {"ohlcv_list": bars}}}
    (out / "eth_0xpool_p0.json").write_text(pd.Series(payload).to_json(), encoding="utf-8")


def test_v63_token_pool_dex_attention_with_manual_mapping(tmp_path: Path) -> None:
    feature_path = tmp_path / "features.parquet"
    trade_path = tmp_path / "trades.csv"
    seed_path = tmp_path / "symbol_mapping_seed.csv"
    manual_path = tmp_path / "manual_mapping.csv"
    cache = tmp_path / "cache"
    _write_features(feature_path)
    _write_trades(trade_path)
    _write_ohlcv_cache(cache)

    pd.DataFrame([{"symbol": "AAAUSDT", "base_asset": "AAA"}]).to_csv(seed_path, index=False)
    pd.DataFrame(
        [
            {
                "cex_symbol": "AAAUSDT",
                "base_asset": "AAA",
                "chain": "eth",
                "token_address": "0xaaa",
                "pool_address": "0xpool",
                "pool_rank": 1,
                "pool_dex": "unit_test_dex",
                "pool_quote_token": "eth_0xquote",
                "pool_liquidity_usd": 1_000_000,
                "pool_24h_volume_usd": 100_000,
                "mapping_confidence": "A",
                "mapping_source": "unit_test_manual",
            }
        ]
    ).to_csv(manual_path, index=False)

    outputs = write_v63_token_pool_dex_attention(
        V63Config(
            report_root=tmp_path / "reports" / "v63",
            mapping_seed_path=seed_path,
            manual_mapping_path=manual_path,
            cache_root=cache,
            v61=V61Config(feature_path=feature_path, trade_cache_path=trade_path, allow_network=False),
            allow_network=False,
            max_symbols=1,
            ohlcv_pages=1,
            min_lookback_bars=10,
            lookback_bars=20,
        )
    )

    mapping = pd.read_csv(outputs["token_pool_mapping"])
    assert mapping.loc[0, "mapping_confidence"] == "A"
    events = pd.read_csv(outputs["token_pool_attention_events"])
    assert not events.empty
    response = pd.read_csv(outputs["token_to_cex_response_curve"])
    assert "cex_volume_shock_rate" in response.columns
    fusion = pd.read_csv(outputs["token_cic_fusion_summary"])
    assert "module" in fusion.columns

