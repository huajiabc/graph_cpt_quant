from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config
from pressure_graph.reports.v65_token_pool_coverage_expansion import V65Config, write_v65_token_pool_coverage_expansion


def test_v65_token_pool_coverage_expansion_outputs_priority_and_overlap(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    token_events_path = tmp_path / "token_events.csv"
    trades_path = tmp_path / "trades.csv"

    pd.DataFrame(
        [
            {
                "cex_symbol": "AAAUSDT",
                "base_asset": "AAA",
                "chain": "eth",
                "token_address": "0xaaa",
                "pool_address": "0xpoolaaa",
                "pool_dex": "unit",
                "pool_quote_token": "eth_0xquote",
                "pool_liquidity_usd": 1_000_000,
                "pool_24h_volume_usd": 250_000,
                "mapping_confidence": "A",
                "mapping_source": "unit_test_manual",
                "coingecko_id": "aaa",
            },
            {
                "cex_symbol": "BBBUSDT",
                "base_asset": "BBB",
                "chain": "",
                "token_address": "",
                "pool_address": "",
                "mapping_confidence": "D",
                "mapping_source": "unit_test_missing",
                "coingecko_id": "bbb",
            },
        ]
    ).to_csv(mapping_path, index=False)
    pd.DataFrame(
        [
            {
                "event_id": "aaa-event",
                "cex_symbol": "AAAUSDT",
                "event_time": "2026-05-02 05:00:00+00:00",
                "event_available_time": "2026-05-02 05:05:00+00:00",
                "event_type": "token_pool_volume_spike",
                "zscore": 3.0,
                "percentile": 0.99,
            }
        ]
    ).to_csv(token_events_path, index=False)

    rows = []
    base = pd.Timestamp("2026-05-02 06:00:00", tz="UTC")
    for idx in range(10):
        rows.append(
            {
                "symbol": "AAAUSDT",
                "candidate": "CIC1_beta_extreme" if idx % 2 == 0 else "CIC2_beta_broad",
                "entry_time": base + pd.Timedelta(minutes=idx),
                "exit_time": base + pd.Timedelta(hours=3, minutes=idx),
                "net_return": 0.01,
                "signal_id": f"aaa-{idx}",
            }
        )
    rows.append(
        {
            "symbol": "BBBUSDT",
            "candidate": "CIC2_beta_broad",
            "entry_time": "2026-05-03 06:00:00+00:00",
            "exit_time": "2026-05-03 09:00:00+00:00",
            "net_return": -0.02,
            "signal_id": "bbb-1",
        }
    )
    pd.DataFrame(rows).to_csv(trades_path, index=False)

    outputs = write_v65_token_pool_coverage_expansion(
        V65Config(
            report_root=tmp_path / "reports" / "v65",
            token_mapping_path=mapping_path,
            token_events_path=token_events_path,
            v61=V61Config(trade_cache_path=trades_path, allow_network=False),
        )
    )

    coverage = pd.read_csv(outputs["trade_weighted_mapping_coverage"])
    assert {"p2_trades", "o6_trades", "dex_relevance_score", "mapping_status"}.issubset(coverage.columns)
    aaa = coverage[coverage["symbol"].eq("AAAUSDT")].iloc[0]
    assert aaa["mapping_status"] == "mapped_A_B"
    assert aaa["p2_trades"] == 10
    assert aaa["o6_trades"] >= 1

    missing = pd.read_csv(outputs["missing_mapping_priority"])
    assert "BBBUSDT" in set(missing["symbol"])
    assert {"manual_review_needed", "coverage_gap_reason", "priority_score"}.issubset(missing.columns)

    overlap = pd.read_csv(outputs["token_event_overlap_audit"])
    aaa_overlap = overlap[overlap["symbol"].eq("AAAUSDT")].iloc[0]
    assert aaa_overlap["token_prior_24h"] == 10
    assert aaa_overlap["o6_prior_count"] >= 1

    targets = pd.read_csv(outputs["coverage_targets"])
    assert "mapped_A_B_trade_coverage" in set(targets["metric"])
