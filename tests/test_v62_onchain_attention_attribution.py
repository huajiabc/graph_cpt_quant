from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v62_onchain_attention_attribution import V62Config, write_v62_onchain_attention_attribution


def _write_json_cache(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2026-01-01", tz="UTC")
    chart = []
    for idx in range(90):
        ts = int((base + pd.Timedelta(days=idx)).timestamp())
        value = 1000.0 + idx * 2.0
        if idx in {45, 60}:
            value = 3500.0 + idx * 10.0
        chart.append([ts, value])
    for name in ["defillama_dexs_overview.json", "defillama_fees_overview.json"]:
        (cache / name).write_text(pd.Series({"totalDataChart": chart}).to_json(), encoding="utf-8")

    stable = []
    for idx in range(90):
        ts = int((base + pd.Timedelta(days=idx)).timestamp())
        value = 10_000.0 + idx * 10.0
        if idx == 50:
            value = 45_000.0
        stable.append({"date": str(ts), "totalCirculatingUSD": {"peggedUSD": value}})
    (cache / "defillama_stablecoincharts_all.json").write_text(pd.Series(stable).to_json(), encoding="utf-8")


def _write_features(path: Path) -> None:
    rows = []
    times = pd.date_range("2026-02-01", periods=2400, freq="15min", tz="UTC")
    for symbol in ["AAAUSDT", "BBBUSDT", "BTCUSDT"]:
        for idx, ts in enumerate(times):
            shock = idx % 113 == 0
            rows.append(
                {
                    "symbol": symbol,
                    "feature_time": ts,
                    "warmup_complete": True,
                    "universe_dynamic_monthly_top30": True,
                    "ret_15m": 0.001,
                    "ret_1h": 0.006 if shock else 0.001,
                    "ret_4h": 0.01,
                    "volume_z_1h": 2.8 if shock else 0.2,
                    "volume_z_4h": 1.0,
                    "future_ret_4h": 0.01,
                    "future_ret_12h": 0.02,
                    "btc_market_state": "btc_up" if idx % 2 == 0 else "btc_chop",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_trades(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": "2026-02-16 05:00:00+00:00",
                "exit_time": "2026-02-16 09:00:00+00:00",
                "net_return": 0.02,
                "signal_id": "t1",
            },
            {
                "symbol": "BBBUSDT",
                "candidate": "CIC2_beta_broad",
                "entry_time": "2026-03-03 06:00:00+00:00",
                "exit_time": "2026-03-03 10:00:00+00:00",
                "net_return": -0.01,
                "signal_id": "t2",
            },
        ]
    ).to_csv(path, index=False)


def test_v62_onchain_attention_attribution_outputs_controls(tmp_path: Path) -> None:
    cache = tmp_path / "defillama"
    _write_json_cache(cache)
    feature_path = tmp_path / "features.parquet"
    trade_path = tmp_path / "trades.csv"
    _write_features(feature_path)
    _write_trades(trade_path)

    v61 = V61Config(
        report_root=tmp_path / "reports" / "v61",
        feature_path=feature_path,
        trade_cache_path=trade_path,
        cache_root=cache,
        allow_network=False,
        min_lookback_days=10,
        lookback_days=20,
    )
    outputs = write_v62_onchain_attention_attribution(
        V62Config(
            report_root=tmp_path / "reports" / "v62",
            v61=v61,
            matched_random_trials=5,
        )
    )

    assert outputs["asof_policy_summary"].exists()
    asof = pd.read_csv(outputs["asof_policy_summary"])
    assert set(asof["asof_policy"]) == {"same_day_naive", "next_day_conservative"}
    assert asof.loc[asof["asof_policy"].eq("next_day_conservative"), "event_available_lag_hours"].iloc[0] == 28

    matched = pd.read_csv(outputs["event_day_matched_random"])
    assert "event_percentile" in matched.columns
    assert "dedup_all" in set(matched["scope"])

    interaction = pd.read_csv(outputs["onchain_cic_interaction"])
    assert {"P2_all", "CIC1", "CIC2"}.issubset(set(interaction["module"]))

