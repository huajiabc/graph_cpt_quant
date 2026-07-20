from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v66_token_attention_attribution import V66Config
from pressure_graph.reports.v67_token_attention_forward_context import (
    V67Config,
    build_token_attention_context_for_trades,
    write_v67_token_attention_forward_context,
)


def test_v67_token_attention_forward_context_is_counterfactual_only(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    events_path = tmp_path / "events.csv"
    market_path = tmp_path / "market_days.csv"
    trades_path = tmp_path / "trades.csv"
    ohlcv_path = tmp_path / "ohlcv.csv"

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
                "source": "unit_source",
                "zscore": 3.0,
                "percentile": 0.99,
            }
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
    pd.DataFrame(
        [
            {"cex_symbol": "AAAUSDT", "time_close": "2026-05-02 06:00:00+00:00"},
            {"cex_symbol": "BBBUSDT", "time_close": "2026-05-04 06:00:00+00:00"},
        ]
    ).to_csv(ohlcv_path, index=False)

    cfg = V67Config(
        report_root=tmp_path / "reports" / "v67",
        v66_report_root=tmp_path / "reports" / "missing_v66",
        v66=V66Config(
            token_events_path=events_path,
            token_mapping_path=mapping_path,
            market_attention_days_path=market_path,
            v61=V61Config(trade_cache_path=trades_path, allow_network=False),
            random_trials=3,
            random_seed=7,
        ),
        token_ohlcv_path=ohlcv_path,
    )
    outputs = write_v67_token_attention_forward_context(cfg)

    ledger = pd.read_csv(outputs["forward_context_ledger"])
    assert len(ledger) == 2
    aaa = ledger[ledger["symbol"].eq("AAAUSDT")].iloc[0]
    assert bool(aaa["token_prior_24h"]) is True
    assert aaa["token_event_types_24h"] == "token_pool_volume_spike"
    assert aaa["token_event_sources_24h"] == "unit_source"
    assert aaa["recommended_use"] == "forward_counterfactual_diagnostic_only"
    assert not bool(aaa["live_action_allowed"])
    assert bool(aaa["token_mapping_covered"])
    assert not bool(aaa["token_dataset_stale_at_entry"])
    assert bool(aaa["token_event_asof_passed_24h"])
    assert aaa["token_event_publication_latency_minutes_24h"] == 65.0
    assert not bool(aaa["token_placebo_7d_prior_24h"])

    decisions = pd.read_csv(outputs["decision_table"])
    assert not decisions.empty
    assert not decisions["live_action_allowed"].astype(bool).any()
    assert not decisions["shadow_portfolio_allowed"].astype(bool).any()

    spec = pd.read_csv(outputs["live_field_spec"])
    assert {"token_prior_24h", "token_event_age_minutes_24h", "token_attention_live_action_allowed", "token_dataset_stale_at_entry", "token_placebo_7d_prior_24h"}.issubset(
        set(spec["field"])
    )

    live = build_token_attention_context_for_trades(
        pd.DataFrame(
            [
                {
                    "trade_id": "live-a",
                    "signal_id": "live-signal-a",
                    "symbol": "AAAUSDT",
                    "candidate": "CIC1_FILTERED_MIR1",
                    "signal_time": "2026-05-02 06:30:00+00:00",
                    "entry_time": "2026-05-02 07:00:00+00:00",
                    "net_return_20bp": 0.02,
                },
                {
                    "trade_id": "live-unmapped",
                    "signal_id": "live-signal-unmapped",
                    "symbol": "ZZZUSDT",
                    "candidate": "CIC2_FILTERED_MIR1",
                    "signal_time": "2026-05-02 06:30:00+00:00",
                    "entry_time": "2026-05-02 07:00:00+00:00",
                    "net_return_20bp": -0.01,
                },
            ]
        ),
        cfg,
    )
    assert len(live) == 2
    live_a = live[live["trade_id"].eq("live-a")].iloc[0]
    unmapped = live[live["trade_id"].eq("live-unmapped")].iloc[0]
    assert bool(live_a["token_prior_24h"]) is True
    assert not bool(unmapped["token_mapping_covered"])
    assert bool(unmapped["token_dataset_stale_at_entry"])
