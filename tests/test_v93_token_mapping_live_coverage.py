from __future__ import annotations

from pathlib import Path

import pandas as pd

from pressure_graph.reports.v93_token_mapping_live_coverage import (
    TokenMappingExpansionConfig,
    discover_symbol_mapping,
    write_token_mapping_live_coverage,
)


def _getter(url: str) -> dict:
    if "/search?" in url:
        return {
            "tokens": [
                {"id": "native-sui", "name": "Sui", "symbol": "SUI", "chain": "sui", "status": "visible"},
                {"id": "wrapped-sui", "name": "Wrapped Sui", "symbol": "SUI", "chain": "eth", "status": "visible"},
            ]
        }
    if "/sui/" in url:
        return {
            "pools": [
                {
                    "id": "sui-pool",
                    "dex_id": "cetus",
                    "volume_usd": 400_000,
                    "tokens": [{"id": "native-sui"}, {"id": "usdc"}],
                }
            ]
        }
    return {
        "pools": [
            {
                "id": "eth-pool",
                "dex_id": "uniswap",
                "volume_usd": 50_000,
                "tokens": [{"id": "wrapped-sui"}, {"id": "weth"}],
            }
        ]
    }


def test_discovery_promotes_only_dominant_exact_symbol() -> None:
    result, candidates = discover_symbol_mapping("SUIUSDT", TokenMappingExpansionConfig(), _getter)
    assert result["mapping_confidence"] == "B"
    assert result["pool_address"] == "sui-pool"
    assert result["token_dominance_ratio"] > 3.0
    assert len(candidates) == 1
    assert candidates[0]["selected_for_mapping"]


def test_discovery_does_not_promote_unregistered_wrapped_asset() -> None:
    result, candidates = discover_symbol_mapping("BCHUSDT", TokenMappingExpansionConfig(), _getter)
    assert result["mapping_confidence"] == "C"
    assert result["discovery_status"] == "canonical_network_not_registered"
    assert candidates == []


def test_live_coverage_preserves_existing_and_promotes_new_mapping(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame(
        [
            {
                "cex_symbol": "AAAUSDT",
                "base_asset": "AAA",
                "chain": "eth",
                "token_address": "aaa",
                "pool_address": "aaa-pool",
                "mapping_confidence": "B",
            },
            {"cex_symbol": "SUIUSDT", "base_asset": "SUI", "mapping_confidence": "D"},
        ]
    ).to_csv(mapping_path, index=False)
    cfg = TokenMappingExpansionConfig(mapping_path=mapping_path, report_root=tmp_path / "report")
    outputs = write_token_mapping_live_coverage(
        cfg,
        symbols=["AAAUSDT", "SUIUSDT"],
        promote=True,
        getter=_getter,
    )
    updated = pd.read_csv(mapping_path)
    assert set(updated["cex_symbol"]) == {"AAAUSDT", "SUIUSDT"}
    assert updated.set_index("cex_symbol").loc["SUIUSDT", "mapping_confidence"] == "B"
    coverage = pd.read_csv(outputs["coverage"])
    assert coverage["mapping_covered_A_B"].all()
