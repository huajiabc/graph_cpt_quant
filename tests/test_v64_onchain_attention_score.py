from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v64_onchain_attention_score import V64Config, write_v64_onchain_attention_score


def test_v64_onchain_attention_score_builds_trade_ledger(tmp_path: Path) -> None:
    market_path = tmp_path / "market_days.csv"
    token_path = tmp_path / "token_events.csv"
    mapping_path = tmp_path / "mapping.csv"
    trades_path = tmp_path / "trades.csv"

    pd.DataFrame(
        [
            {
                "attention_day_id": "global|2026-05-01",
                "event_available_time": "2026-05-02 04:00:00+00:00",
                "event_date": "2026-05-01 00:00:00+00:00",
                "attention_intensity": 5.0,
                "primary_event_type": "protocol_fee_spike",
            },
            {
                "attention_day_id": "global|2026-05-02",
                "event_available_time": "2026-05-03 04:00:00+00:00",
                "event_date": "2026-05-02 00:00:00+00:00",
                "attention_intensity": 1.0,
                "primary_event_type": "chain_dex_volume_spike",
            },
        ]
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "AAA|pool|2026-05-02T05:00:00Z",
                "cex_symbol": "AAAUSDT",
                "event_available_time": "2026-05-02 06:05:00+00:00",
                "event_time": "2026-05-02 05:00:00+00:00",
                "event_type": "token_pool_volume_spike",
                "zscore": 3.0,
                "percentile": 0.99,
            }
        ]
    ).to_csv(token_path, index=False)
    pd.DataFrame(
        [{"cex_symbol": "AAAUSDT", "mapping_confidence": "A"}, {"cex_symbol": "BBBUSDT", "mapping_confidence": "D"}]
    ).to_csv(mapping_path, index=False)
    pd.DataFrame(
        [
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme",
                "entry_time": "2026-05-02 07:00:00+00:00",
                "exit_time": "2026-05-02 11:00:00+00:00",
                "net_return": 0.02,
                "signal_id": "a",
            },
            {
                "symbol": "BBBUSDT",
                "candidate": "CIC2_beta_broad",
                "entry_time": "2026-05-04 07:00:00+00:00",
                "exit_time": "2026-05-04 11:00:00+00:00",
                "net_return": -0.01,
                "signal_id": "b",
            },
        ]
    ).to_csv(trades_path, index=False)

    outputs = write_v64_onchain_attention_score(
        V64Config(
            report_root=tmp_path / "reports" / "v64",
            market_attention_days_path=market_path,
            token_events_path=token_path,
            token_mapping_path=mapping_path,
            v61=V61Config(trade_cache_path=trades_path, allow_network=False),
            random_trials=5,
        )
    )

    ledger = pd.read_csv(outputs["scored_trade_ledger"])
    assert {"market_attention_score", "token_attention_score", "combined_attention_score"}.issubset(ledger.columns)
    assert ledger["combined_attention_score"].max() > 0
    coverage = pd.read_csv(outputs["score_coverage"])
    assert "token_mapping_A_B" in set(coverage["dataset"])
    random_control = pd.read_csv(outputs["score_random_control"])
    assert "actual_percentile" in random_control.columns

