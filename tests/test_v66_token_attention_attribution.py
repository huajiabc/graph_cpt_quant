from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v66_token_attention_attribution import V66Config, write_v66_token_attention_attribution


def test_v66_token_attention_attribution_builds_controls(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    events_path = tmp_path / "events.csv"
    market_path = tmp_path / "market_days.csv"
    trades_path = tmp_path / "trades.csv"

    pd.DataFrame(
        [
            {"cex_symbol": "AAAUSDT", "chain": "eth", "mapping_confidence": "B", "token_address": "0xaaa", "pool_address": "0xpoolaaa"},
            {"cex_symbol": "BBBUSDT", "chain": "eth", "mapping_confidence": "B", "token_address": "0xbbb", "pool_address": "0xpoolbbb"},
        ]
    ).to_csv(mapping_path, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "aaa-1",
                "cex_symbol": "AAAUSDT",
                "event_time": "2026-05-02 05:00:00+00:00",
                "event_available_time": "2026-05-02 06:05:00+00:00",
                "event_type": "token_pool_volume_spike",
                "source": "unit",
                "zscore": 3.0,
                "percentile": 0.99,
            },
            {
                "event_id": "bbb-1",
                "cex_symbol": "BBBUSDT",
                "event_time": "2026-05-03 05:00:00+00:00",
                "event_available_time": "2026-05-03 06:05:00+00:00",
                "event_type": "token_pool_volume_spike",
                "source": "unit",
                "zscore": 2.5,
                "percentile": 0.97,
            },
        ]
    ).to_csv(events_path, index=False)
    pd.DataFrame([{"event_date": "2026-05-02 00:00:00+00:00"}]).to_csv(market_path, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": "2026-05-02 07:00:00+00:00",
                "exit_time": "2026-05-02 10:00:00+00:00",
                "net_return": 0.03,
                "signal_id": "a",
            },
            {
                "symbol": "BBBUSDT",
                "candidate": "CIC2_beta_broad",
                "entry_time": "2026-05-04 07:00:00+00:00",
                "exit_time": "2026-05-04 10:00:00+00:00",
                "net_return": -0.01,
                "signal_id": "b",
            },
        ]
    ).to_csv(trades_path, index=False)

    outputs = write_v66_token_attention_attribution(
        V66Config(
            report_root=tmp_path / "reports" / "v66",
            token_events_path=events_path,
            token_mapping_path=mapping_path,
            market_attention_days_path=market_path,
            v61=V61Config(trade_cache_path=trades_path, allow_network=False),
            random_trials=3,
            random_seed=1,
        )
    )

    fusion = pd.read_csv(outputs["token_cic_o6_fusion_summary"])
    p2_24h = fusion[(fusion["module"].eq("P2_all")) & (fusion["lookback_window"].eq("24h"))].iloc[0]
    assert p2_24h["prior_trades"] == 1
    assert p2_24h["no_prior_trades"] == 1

    controls = pd.read_csv(outputs["random_control_summary"])
    assert {"same_token_random_time", "same_day_random_token", "same_chain_random_token"}.issubset(set(controls["control"]))

    market = pd.read_csv(outputs["market_level_control_summary"])
    assert "token_prior_market_day" in set(market["bucket"])
