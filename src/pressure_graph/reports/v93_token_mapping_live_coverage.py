from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


DEXPAPRIKA_SEARCH = "https://api.dexpaprika.com/search?query={query}"
DEXPAPRIKA_TOKEN_POOLS = "https://api.dexpaprika.com/networks/{network}/tokens/{token}/pools"
AUTO_MAPPING_SOURCE = "dexpaprika_exact_symbol_dominant_pool"
CANONICAL_NETWORKS_BY_ASSET = {
    "ADA": {"cardano"},
    "APT": {"aptos"},
    "AVAX": {"avalanche"},
    "BNB": {"bsc"},
    "ETH": {"ethereum"},
    "FARTCOIN": {"solana"},
    "HBAR": {"hedera"},
    "HYPE": {"hyperevm"},
    "JTO": {"solana"},
    "NEAR": {"near"},
    "PENGU": {"solana"},
    "RENDER": {"solana"},
    "SOL": {"solana"},
    "SUI": {"sui"},
    "TON": {"ton"},
    "TRUMP": {"solana"},
    "VIRTUAL": {"base"},
    "VVV": {"base"},
    "WIF": {"solana"},
    "WLD": {"ethereum", "optimism"},
    "XAUT": {"ethereum"},
    "XRP": {"xrpl"},
}


@dataclass(frozen=True)
class TokenMappingExpansionConfig:
    mapping_path: Path = Path("reports/v6_3_token_pool_dex_attention/token_pool_mapping.csv")
    live_feature_path: Path = Path("data/live_v07d2/processed/v0_7d2_live_features.parquet")
    report_root: Path = Path("reports/v6_5_token_pool_coverage_expansion")
    minimum_pool_volume_usd: float = 25_000.0
    minimum_token_dominance_ratio: float = 3.0
    request_timeout_seconds: int = 45


def _clean_base(symbol: str) -> str:
    text = str(symbol).upper().removesuffix("USDT")
    for prefix in ("1000000", "100000", "10000", "1000"):
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            return text[len(prefix) :]
    return text


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _http_json(url: str, timeout: int = 45) -> Any:
    request = Request(url, headers={"User-Agent": "graph_quant/0.1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _pool_candidate(token: dict[str, Any], pools: list[dict[str, Any]]) -> dict[str, object] | None:
    token_id = str(token.get("id", ""))
    exact_pools = []
    for pool in pools:
        pool_tokens = pool.get("tokens", []) if isinstance(pool, dict) else []
        if token_id not in {str(item.get("id", "")) for item in pool_tokens}:
            continue
        exact_pools.append(pool)
    if not exact_pools:
        return None
    top = max(exact_pools, key=lambda item: _float(item.get("volume_usd")) if np.isfinite(_float(item.get("volume_usd"))) else -1.0)
    other_tokens = [item for item in top.get("tokens", []) if str(item.get("id", "")) != token_id]
    quote_token = str(other_tokens[0].get("id", "")) if other_tokens else ""
    return {
        "chain": str(token.get("chain", "")),
        "token_address": token_id,
        "token_name": str(token.get("name", "")),
        "pool_address": str(top.get("id", "")),
        "pool_dex": str(top.get("dex_id", "")),
        "pool_quote_token": quote_token,
        "pool_liquidity_usd": np.nan,
        "pool_24h_volume_usd": _float(top.get("volume_usd")),
    }


def discover_symbol_mapping(
    cex_symbol: str,
    cfg: TokenMappingExpansionConfig,
    getter: Callable[[str], Any] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    getter = getter or (lambda url: _http_json(url, cfg.request_timeout_seconds))
    base = _clean_base(cex_symbol)
    result: dict[str, object] = {
        "cex_symbol": cex_symbol,
        "base_asset": base,
        "mapping_confidence": "D",
        "mapping_source": "dexpaprika_no_eligible_exact_symbol",
        "discovery_status": "no_eligible_exact_symbol",
    }
    if len(base) < 3:
        result.update(mapping_source="dexpaprika_short_symbol_not_auto_promoted", discovery_status="ambiguous_short_symbol")
        return result, []
    canonical_networks = CANONICAL_NETWORKS_BY_ASSET.get(base)
    if not canonical_networks:
        result.update(
            mapping_confidence="C",
            mapping_source="dexpaprika_canonical_network_not_registered",
            discovery_status="canonical_network_not_registered",
        )
        return result, []
    search = getter(DEXPAPRIKA_SEARCH.format(query=quote(base)))
    exact_tokens = [
        token
        for token in search.get("tokens", [])
        if str(token.get("symbol", "")).upper() == base and str(token.get("status", "visible")) == "visible"
        and str(token.get("chain", "")) in canonical_networks
    ][:8] if isinstance(search, dict) else []

    def fetch_candidate(token: dict[str, Any]) -> dict[str, object] | None:
        network = str(token.get("chain", ""))
        token_id = str(token.get("id", ""))
        if not network or not token_id:
            return None
        pools_url = DEXPAPRIKA_TOKEN_POOLS.format(network=quote(network), token=quote(token_id, safe=":"))
        try:
            payload = getter(pools_url)
        except Exception:  # noqa: BLE001 - one chain candidate must not abort coverage discovery
            return None
        candidate = _pool_candidate(token, payload.get("pools", []) if isinstance(payload, dict) else [])
        return {"cex_symbol": cex_symbol, "base_asset": base, **candidate} if candidate is not None else None

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(exact_tokens)))) as executor:
        fetched = list(executor.map(fetch_candidate, exact_tokens))
    candidates = [candidate for candidate in fetched if candidate is not None]
    candidates.sort(key=lambda item: _float(item.get("pool_24h_volume_usd")), reverse=True)
    if not candidates:
        return result, []
    top_volume = _float(candidates[0].get("pool_24h_volume_usd"))
    second_volume = _float(candidates[1].get("pool_24h_volume_usd")) if len(candidates) > 1 else 0.0
    dominance = top_volume / max(second_volume, 1.0) if np.isfinite(top_volume) else 0.0
    status = "promoted_B"
    confidence = "B"
    source = AUTO_MAPPING_SOURCE
    if not np.isfinite(top_volume) or top_volume < cfg.minimum_pool_volume_usd:
        status, confidence, source = "insufficient_pool_volume", "C", "dexpaprika_exact_symbol_low_volume_pool"
    elif dominance < cfg.minimum_token_dominance_ratio:
        status, confidence, source = "ambiguous_exact_symbol", "C", "dexpaprika_exact_symbol_ambiguous_chain"
    result.update(
        candidates[0],
        pool_rank=1,
        mapping_confidence=confidence,
        mapping_source=source,
        discovery_status=status,
        token_candidate_count=len(candidates),
        token_dominance_ratio=dominance,
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["candidate_rank"] = rank
        candidate["token_dominance_ratio"] = dominance
        candidate["selected_for_mapping"] = rank == 1 and confidence == "B"
    return result, candidates


def _universe(cfg: TokenMappingExpansionConfig, symbols: list[str] | None) -> list[str]:
    if symbols:
        return sorted({str(symbol).upper() for symbol in symbols if str(symbol)})
    if not cfg.live_feature_path.exists():
        return []
    return sorted(pd.read_parquet(cfg.live_feature_path, columns=["symbol"])["symbol"].dropna().astype(str).str.upper().unique())


def write_token_mapping_live_coverage(
    cfg: TokenMappingExpansionConfig = TokenMappingExpansionConfig(),
    *,
    symbols: list[str] | None = None,
    promote: bool = False,
    getter: Callable[[str], Any] | None = None,
) -> dict[str, Path]:
    mapping = pd.read_csv(cfg.mapping_path, low_memory=False) if cfg.mapping_path.exists() else pd.DataFrame()
    if not mapping.empty and {"base_asset", "chain", "mapping_source", "mapping_confidence"}.issubset(mapping.columns):
        auto = mapping["mapping_source"].astype(str).eq(AUTO_MAPPING_SOURCE)
        canonical = pd.Series(False, index=mapping.index)
        for idx in mapping.index[auto]:
            base = _clean_base(str(mapping.at[idx, "cex_symbol"]))
            canonical.at[idx] = str(mapping.at[idx, "chain"]) in CANONICAL_NETWORKS_BY_ASSET.get(base, set())
        rejected = auto & ~canonical
        mapping.loc[rejected, "mapping_confidence"] = "C"
        mapping.loc[rejected, "mapping_source"] = "dexpaprika_auto_mapping_rejected_noncanonical"
    universe = _universe(cfg, symbols)
    index = mapping.drop_duplicates("cex_symbol").set_index("cex_symbol") if not mapping.empty else pd.DataFrame()
    discoveries: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for symbol in universe:
        current = index.loc[symbol] if not index.empty and symbol in index.index else pd.Series(dtype=object)
        if str(current.get("mapping_confidence", "")) in {"A", "B"} and str(current.get("pool_address", "")) not in {"", "nan"}:
            continue
        try:
            discovery, candidates = discover_symbol_mapping(symbol, cfg, getter)
        except Exception as exc:  # noqa: BLE001 - record API failures without corrupting the existing mapping
            discovery = {
                "cex_symbol": symbol,
                "base_asset": _clean_base(symbol),
                "mapping_confidence": "D",
                "mapping_source": "dexpaprika_discovery_error",
                "discovery_status": f"error:{type(exc).__name__}",
            }
            candidates = []
        discoveries.append(discovery)
        candidate_rows.extend(candidates)
        print(
            f"v93 mapping {symbol} status={discovery.get('discovery_status', '')} "
            f"confidence={discovery.get('mapping_confidence', '')}",
            flush=True,
        )
    discovery_frame = pd.DataFrame(discoveries)
    merged = mapping.copy()
    if promote and not discovery_frame.empty:
        promoted = discovery_frame[discovery_frame["mapping_confidence"].astype(str).eq("B")].copy()
        if not promoted.empty:
            merged = merged[~merged.get("cex_symbol", pd.Series(dtype=str)).astype(str).isin(set(promoted["cex_symbol"].astype(str)))]
            merged = pd.concat([merged, promoted], ignore_index=True, sort=False)
    mapped_symbols = set(
        merged.loc[
            merged.get("mapping_confidence", pd.Series(dtype=str)).astype(str).isin(["A", "B"])
            & merged.get("pool_address", pd.Series(dtype=str)).fillna("").astype(str).ne(""),
            "cex_symbol",
        ].astype(str)
    ) if not merged.empty else set()
    coverage = pd.DataFrame(
        {
            "cex_symbol": universe,
            "mapping_covered_A_B": [symbol in mapped_symbols for symbol in universe],
        }
    )
    if not discovery_frame.empty:
        coverage = coverage.merge(
            discovery_frame[["cex_symbol", "discovery_status", "mapping_confidence"]].rename(
                columns={"mapping_confidence": "discovered_confidence"}
            ),
            on="cex_symbol",
            how="left",
        )
    root = ensure_dir(cfg.report_root)
    outputs = {
        "mapping": cfg.mapping_path,
        "discovery": root / "live_universe_mapping_discovery.csv",
        "candidates": root / "live_universe_mapping_candidates.csv",
        "coverage": root / "live_universe_mapping_coverage.csv",
        "status": root / "live_universe_mapping_status.md",
    }
    ensure_dir(cfg.mapping_path.parent)
    if promote:
        merged.to_csv(cfg.mapping_path, index=False)
    discovery_frame.to_csv(outputs["discovery"], index=False)
    pd.DataFrame(candidate_rows).to_csv(outputs["candidates"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    covered = int(coverage["mapping_covered_A_B"].sum()) if not coverage.empty else 0
    outputs["status"].write_text(
        "\n".join(
            [
                "# Live Token Mapping Coverage",
                "",
                f"- universe_symbols: {len(universe)}",
                f"- covered_A_B: {covered}",
                f"- coverage_ratio: {covered / len(universe):.2%}" if universe else "- coverage_ratio: n/a",
                f"- promoted_this_run: {int(discovery_frame['mapping_confidence'].eq('B').sum()) if not discovery_frame.empty else 0}",
                "- automatic promotion requires an exact symbol, at least $25k pool volume, and 3x dominance over the next token candidate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


__all__ = [
    "TokenMappingExpansionConfig",
    "discover_symbol_mapping",
    "write_token_mapping_live_coverage",
]
