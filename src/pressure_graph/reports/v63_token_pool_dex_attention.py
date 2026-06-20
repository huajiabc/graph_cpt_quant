"""v6.3 Token / Pool-Level DEX Attention.

First token-specific on-chain pass.  The report builds a conservative
CEX-symbol -> token -> GeckoTerminal top-pool mapping, backfills top-pool
hourly OHLCV where available, creates token-level DEX volume-spike events,
and audits whether those events lead same-symbol CEX/CIC activity.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v09b import _max_contribution, _month_cap_expectancy
from pressure_graph.reports.v60_onchain_attention_graph import _base_from_symbol
from pressure_graph.reports.v61_onchain_dex_attention_backfill import V61Config, _read_features, _read_p2_trades


REPORT_ROOT = Path("reports/v6_3_token_pool_dex_attention")
DEFAULT_MAPPING_SEED = Path("reports/v6_0_onchain_attention_graph/symbol_mapping_seed.csv")
DEFAULT_MANUAL_MAPPING = Path("configs/onchain_token_pool_mapping.csv")
DEFAULT_CACHE_ROOT = Path("data/external/token_dex_attention")

COINGECKO_SEARCH_URL = "https://api.coingecko.com/api/v3/search?query={query}"
COINGECKO_COIN_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}"
GECKO_POOLS_URL = "https://api.geckoterminal.com/api/v2/networks/{network}/tokens/{address}/pools"
GECKO_OHLCV_URL = (
    "https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/hour"
    "?aggregate=1&limit={limit}{before}"
)

PLATFORM_TO_GECKO = {
    "ethereum": "eth",
    "arbitrum-one": "arbitrum",
    "base": "base",
    "binance-smart-chain": "bsc",
    "polygon-pos": "polygon_pos",
    "optimistic-ethereum": "optimism",
    "avalanche": "avax",
    "solana": "solana",
}
PLATFORM_PRIORITY = (
    "ethereum",
    "solana",
    "base",
    "arbitrum-one",
    "binance-smart-chain",
    "polygon-pos",
    "optimistic-ethereum",
    "avalanche",
)
HORIZONS = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48),
}


@dataclass(frozen=True)
class V63Config:
    report_root: Path = REPORT_ROOT
    mapping_seed_path: Path = DEFAULT_MAPPING_SEED
    manual_mapping_path: Path = DEFAULT_MANUAL_MAPPING
    cache_root: Path = DEFAULT_CACHE_ROOT
    v61: V61Config = V61Config()
    max_symbols: int = 80
    allow_network: bool = True
    request_delay_seconds: float = 0.35
    ohlcv_limit: int = 1000
    ohlcv_pages: int = 3
    z_threshold: float = 2.0
    pct_threshold: float = 0.95
    lookback_bars: int = 120
    min_lookback_bars: int = 48
    random_seed: int = 630


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _clean_base(base: str) -> str:
    text = str(base).upper().strip()
    for prefix in ("1000000", "100000", "10000", "1000"):
        if text.startswith(prefix) and len(text) > len(prefix) + 1:
            return text[len(prefix) :]
    return text


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:160]


def _read_json_cache(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_json(url: str, path: Path, cfg: V63Config) -> Any:
    if path.exists():
        return _read_json_cache(path)
    if not cfg.allow_network:
        return {}
    ensure_dir(path.parent)
    response = requests.get(url, timeout=30, headers={"User-Agent": "graph_quant/0.1"})
    if response.status_code == 429:
        time.sleep(max(cfg.request_delay_seconds * 4.0, 2.0))
        response = requests.get(url, timeout=30, headers={"User-Agent": "graph_quant/0.1"})
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    if cfg.request_delay_seconds > 0:
        time.sleep(cfg.request_delay_seconds)
    return response.json()


def _load_seed_symbols(cfg: V63Config) -> pd.DataFrame:
    if cfg.mapping_seed_path.exists():
        seed = pd.read_csv(cfg.mapping_seed_path)
    else:
        features = _read_features(cfg.v61)
        symbols = sorted(features["symbol"].dropna().astype(str).unique()) if not features.empty else []
        seed = pd.DataFrame({"symbol": symbols, "base_asset": [_base_from_symbol(s) for s in symbols]})
    if seed.empty:
        return pd.DataFrame(columns=["cex_symbol", "base_asset", "query_asset", "seed_rank"])
    seed = seed.rename(columns={"symbol": "cex_symbol"}).copy()
    if "base_asset" not in seed.columns:
        seed["base_asset"] = seed["cex_symbol"].map(_base_from_symbol)
    seed["query_asset"] = seed["base_asset"].map(_clean_base)

    trades = _read_p2_trades(cfg.v61.trade_cache_path)
    if not trades.empty and "dynamic_all_rank" in trades.columns:
        ranks = trades.groupby("symbol", as_index=False)["dynamic_all_rank"].median().rename(columns={"symbol": "cex_symbol", "dynamic_all_rank": "seed_rank"})
        seed = seed.merge(ranks, on="cex_symbol", how="left")
    if "seed_rank" not in seed.columns:
        seed["seed_rank"] = np.nan
    seed["seed_rank_sort"] = pd.to_numeric(seed["seed_rank"], errors="coerce").fillna(9999)
    seed = seed.sort_values(["seed_rank_sort", "cex_symbol"]).head(cfg.max_symbols)
    return seed[["cex_symbol", "base_asset", "query_asset", "seed_rank"]].reset_index(drop=True)


def _manual_mapping(cfg: V63Config) -> pd.DataFrame:
    if not cfg.manual_mapping_path.exists():
        return pd.DataFrame()
    out = pd.read_csv(cfg.manual_mapping_path)
    if out.empty:
        return out
    rename = {"symbol": "cex_symbol"}
    out = out.rename(columns=rename)
    return out


def _pick_coingecko_coin(search: dict[str, Any], query_asset: str) -> dict[str, Any] | None:
    coins = search.get("coins", []) if isinstance(search, dict) else []
    exact = [coin for coin in coins if str(coin.get("symbol", "")).upper() == query_asset.upper()]
    if not exact:
        return None
    exact = sorted(exact, key=lambda coin: coin.get("market_cap_rank") if coin.get("market_cap_rank") is not None else 999_999)
    return exact[0]


def _pick_platform(platforms: dict[str, Any]) -> tuple[str, str, str]:
    for platform in PLATFORM_PRIORITY:
        address = str(platforms.get(platform, "") or "").strip()
        if address and platform in PLATFORM_TO_GECKO:
            return platform, PLATFORM_TO_GECKO[platform], address
    return "", "", ""


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else np.nan


def _volume_h24(attrs: dict[str, Any]) -> float:
    volume = attrs.get("volume_usd", {})
    if isinstance(volume, dict):
        return _parse_float(volume.get("h24"))
    return np.nan


def _pool_row(pool: dict[str, Any]) -> dict[str, Any]:
    attrs = pool.get("attributes", {}) if isinstance(pool, dict) else {}
    rel = pool.get("relationships", {}) if isinstance(pool, dict) else {}
    dex = rel.get("dex", {}).get("data", {}).get("id", "")
    quote = rel.get("quote_token", {}).get("data", {}).get("id", "")
    return {
        "pool_address": attrs.get("address", ""),
        "pool_name": attrs.get("name", ""),
        "pool_dex": dex,
        "pool_quote_token": quote,
        "pool_liquidity_usd": _parse_float(attrs.get("reserve_in_usd")),
        "pool_24h_volume_usd": _volume_h24(attrs),
    }


def _fetch_pool_mapping(seed: pd.DataFrame, cfg: V63Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual = _manual_mapping(cfg)
    manual_symbols = set(manual.get("cex_symbol", pd.Series(dtype=str)).astype(str)) if not manual.empty else set()
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    if not manual.empty:
        for row in manual.to_dict("records"):
            payload = {**row}
            payload.setdefault("mapping_confidence", "A")
            payload.setdefault("mapping_source", "manual")
            payload.setdefault("pool_rank", 1)
            rows.append(payload)
    for seed_row in seed.itertuples(index=False):
        symbol = str(seed_row.cex_symbol)
        if symbol in manual_symbols:
            continue
        query = str(seed_row.query_asset)
        base = str(seed_row.base_asset)
        search_path = cfg.cache_root / "coingecko" / f"search_{_safe_name(query.lower())}.json"
        coin_id = ""
        coin_symbol = ""
        try:
            search = _get_json(COINGECKO_SEARCH_URL.format(query=query), search_path, cfg)
            coin = _pick_coingecko_coin(search, query)
            coin_id = str(coin.get("id", "")) if coin else ""
            coin_symbol = str(coin.get("symbol", "")).upper() if coin else ""
        except Exception as exc:  # noqa: BLE001
            coverage.append({"cex_symbol": symbol, "stage": "coingecko_search", "status": f"error:{type(exc).__name__}"})
            coin = None
        if not coin_id:
            rows.append(
                {
                    "cex_symbol": symbol,
                    "base_asset": base,
                    "chain": "",
                    "token_address": "",
                    "pool_address": "",
                    "pool_rank": np.nan,
                    "pool_dex": "",
                    "pool_quote_token": "",
                    "pool_liquidity_usd": np.nan,
                    "pool_24h_volume_usd": np.nan,
                    "mapping_confidence": "D",
                    "mapping_source": "coingecko_search_no_exact_symbol",
                    "coingecko_id": "",
                    "coingecko_symbol": "",
                }
            )
            continue
        coin_path = cfg.cache_root / "coingecko" / f"coin_{_safe_name(coin_id)}.json"
        try:
            detail = _get_json(COINGECKO_COIN_URL.format(coin_id=coin_id), coin_path, cfg)
            platform, network, address = _pick_platform(detail.get("platforms", {}) if isinstance(detail, dict) else {})
        except Exception as exc:  # noqa: BLE001
            coverage.append({"cex_symbol": symbol, "stage": "coingecko_coin", "status": f"error:{type(exc).__name__}"})
            platform, network, address = "", "", ""
        if not address or not network:
            rows.append(
                {
                    "cex_symbol": symbol,
                    "base_asset": base,
                    "chain": "",
                    "token_address": "",
                    "pool_address": "",
                    "pool_rank": np.nan,
                    "pool_dex": "",
                    "pool_quote_token": "",
                    "pool_liquidity_usd": np.nan,
                    "pool_24h_volume_usd": np.nan,
                    "mapping_confidence": "D",
                    "mapping_source": "coingecko_no_supported_platform_contract",
                    "coingecko_id": coin_id,
                    "coingecko_symbol": coin_symbol,
                }
            )
            continue
        pools_path = cfg.cache_root / "geckoterminal" / f"pools_{network}_{_safe_name(address)}.json"
        pool_rows: list[dict[str, Any]] = []
        try:
            pools = _get_json(GECKO_POOLS_URL.format(network=network, address=address), pools_path, cfg)
            pool_rows = [_pool_row(pool) for pool in pools.get("data", [])] if isinstance(pools, dict) else []
        except Exception as exc:  # noqa: BLE001
            coverage.append({"cex_symbol": symbol, "stage": "geckoterminal_pools", "status": f"error:{type(exc).__name__}"})
        pool_frame = pd.DataFrame(pool_rows)
        if pool_frame.empty:
            rows.append(
                {
                    "cex_symbol": symbol,
                    "base_asset": base,
                    "chain": network,
                    "token_address": address,
                    "pool_address": "",
                    "pool_rank": np.nan,
                    "pool_dex": "",
                    "pool_quote_token": "",
                    "pool_liquidity_usd": np.nan,
                    "pool_24h_volume_usd": np.nan,
                    "mapping_confidence": "C",
                    "mapping_source": "coingecko_contract_no_geckoterminal_pool",
                    "coingecko_id": coin_id,
                    "coingecko_symbol": coin_symbol,
                    "coingecko_platform": platform,
                }
            )
            continue
        pool_frame["pool_rank_metric"] = _num(pool_frame, "pool_liquidity_usd").fillna(0) + _num(pool_frame, "pool_24h_volume_usd").fillna(0)
        top = pool_frame.sort_values("pool_rank_metric", ascending=False).iloc[0].to_dict()
        rows.append(
            {
                "cex_symbol": symbol,
                "base_asset": base,
                "chain": network,
                "token_address": address,
                "pool_address": top.get("pool_address", ""),
                "pool_rank": 1,
                "pool_dex": top.get("pool_dex", ""),
                "pool_quote_token": top.get("pool_quote_token", ""),
                "pool_liquidity_usd": top.get("pool_liquidity_usd", np.nan),
                "pool_24h_volume_usd": top.get("pool_24h_volume_usd", np.nan),
                "mapping_confidence": "B",
                "mapping_source": "coingecko_exact_symbol_geckoterminal_top_pool",
                "coingecko_id": coin_id,
                "coingecko_symbol": coin_symbol,
                "coingecko_platform": platform,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(coverage)


def _fetch_ohlcv(mapping: pd.DataFrame, cfg: V63Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    usable = mapping[mapping["mapping_confidence"].astype(str).isin(["A", "B"])].copy() if not mapping.empty else pd.DataFrame()
    for item in usable.itertuples(index=False):
        symbol = str(item.cex_symbol)
        network = str(item.chain)
        pool = str(item.pool_address)
        if not network or not pool:
            continue
        before = ""
        page_rows: list[list[Any]] = []
        status = "ok"
        for page in range(cfg.ohlcv_pages):
            path = cfg.cache_root / "geckoterminal" / "ohlcv_hour" / f"{network}_{_safe_name(pool)}_p{page}.json"
            url = GECKO_OHLCV_URL.format(network=network, pool_address=pool, limit=cfg.ohlcv_limit, before=before)
            try:
                data = _get_json(url, path, cfg)
                ohlcv = data.get("data", {}).get("attributes", {}).get("ohlcv_list", []) if isinstance(data, dict) else []
            except Exception as exc:  # noqa: BLE001
                status = f"error:{type(exc).__name__}"
                break
            if not ohlcv:
                break
            page_rows.extend(ohlcv)
            oldest_ts = min(int(row[0]) for row in ohlcv if row)
            before = f"&before_timestamp={oldest_ts}"
        for row in page_rows:
            if len(row) < 6:
                continue
            rows.append(
                {
                    "cex_symbol": symbol,
                    "chain": network,
                    "token_address": str(item.token_address),
                    "pool_address": pool,
                    "pool_time": pd.to_datetime(int(row[0]), unit="s", utc=True, errors="coerce"),
                    "open": _parse_float(row[1]),
                    "high": _parse_float(row[2]),
                    "low": _parse_float(row[3]),
                    "close": _parse_float(row[4]),
                    "pool_volume_usd": _parse_float(row[5]),
                }
            )
        coverage.append({"cex_symbol": symbol, "chain": network, "pool_address": pool, "ohlcv_rows": len(page_rows), "status": status})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.dropna(subset=["pool_time"]).drop_duplicates(["cex_symbol", "pool_address", "pool_time"]).sort_values(["cex_symbol", "pool_time"])
    return frame, pd.DataFrame(coverage)


def _rolling_percentile(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    def pct(x: np.ndarray) -> float:
        if len(x) <= 1 or not np.isfinite(x[-1]):
            return np.nan
        prior = x[:-1]
        prior = prior[np.isfinite(prior)]
        return float((prior <= x[-1]).mean()) if len(prior) else np.nan

    return values.rolling(window + 1, min_periods=min_periods + 1).apply(pct, raw=True)


def _build_token_events(ohlcv: pd.DataFrame, mapping: pd.DataFrame, cfg: V63Config) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    map_idx = mapping.drop_duplicates("cex_symbol").set_index("cex_symbol") if not mapping.empty else pd.DataFrame()
    for symbol, group in ohlcv.groupby("cex_symbol", sort=False):
        local = group.sort_values("pool_time").copy()
        volume = _num(local, "pool_volume_usd")
        mean = volume.shift(1).rolling(cfg.lookback_bars, min_periods=cfg.min_lookback_bars).mean()
        std = volume.shift(1).rolling(cfg.lookback_bars, min_periods=cfg.min_lookback_bars).std(ddof=0).replace(0, np.nan)
        local["zscore"] = (volume - mean) / std
        local["percentile"] = _rolling_percentile(volume, cfg.lookback_bars, cfg.min_lookback_bars)
        mask = local["zscore"].ge(cfg.z_threshold) | local["percentile"].ge(cfg.pct_threshold)
        meta = map_idx.loc[symbol] if not map_idx.empty and symbol in map_idx.index else {}
        for row in local[mask.fillna(False)].itertuples(index=False):
            event_time = pd.Timestamp(getattr(row, "pool_time"))
            rows.append(
                {
                    "event_id": f"{symbol}|{getattr(row, 'pool_address')}|{event_time.isoformat()}|token_pool_volume_spike",
                    "cex_symbol": symbol,
                    "base_asset": meta.get("base_asset", _base_from_symbol(symbol)) if isinstance(meta, pd.Series) else _base_from_symbol(symbol),
                    "chain": getattr(row, "chain"),
                    "token_address": getattr(row, "token_address"),
                    "pool_address": getattr(row, "pool_address"),
                    "event_time": event_time,
                    "event_available_time": event_time + pd.Timedelta(hours=1, minutes=5),
                    "event_type": "token_pool_volume_spike",
                    "raw_value": float(getattr(row, "pool_volume_usd")),
                    "zscore": float(getattr(row, "zscore")) if pd.notna(getattr(row, "zscore")) else np.nan,
                    "percentile": float(getattr(row, "percentile")) if pd.notna(getattr(row, "percentile")) else np.nan,
                    "lookback_window": cfg.lookback_bars,
                    "mapping_confidence": meta.get("mapping_confidence", "") if isinstance(meta, pd.Series) else "",
                    "source": "geckoterminal_pool_ohlcv",
                }
            )
    return pd.DataFrame(rows).sort_values(["event_time", "cex_symbol"]).reset_index(drop=True)


def _prepare_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    out = features.copy()
    out["feature_time"] = pd.to_datetime(out["feature_time"], utc=True, errors="coerce")
    return out.dropna(subset=["symbol", "feature_time"]).sort_values(["symbol", "feature_time"]).reset_index(drop=True)


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["net20"] = _num(out, "net20") if "net20" in out.columns else _num(out, "net_return")
    out["month"] = out["entry_time"].dt.strftime("%Y-%m")
    return out.dropna(subset=["symbol", "entry_time", "net20"])


def _sample_metrics(sample: pd.DataFrame) -> dict[str, Any]:
    if sample.empty:
        return {"trades": 0, "net20": np.nan, "hit_rate": np.nan, "month_cap35_net20": np.nan, "max_symbol_contribution": np.nan}
    local = sample.copy()
    local["net_return"] = _num(local, "net20")
    if "month" not in local.columns:
        local["month"] = pd.to_datetime(local["entry_time"], utc=True, errors="coerce").dt.strftime("%Y-%m")
    return {
        "trades": int(len(local)),
        "net20": float(_num(local, "net20").mean()),
        "hit_rate": float(_num(local, "net20").gt(0).mean()),
        "month_cap35_net20": _month_cap_expectancy(local),
        "max_symbol_contribution": _max_contribution(local, "symbol"),
    }


def _event_response(events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if events.empty or features.empty:
        return pd.DataFrame([{"event_type": "no_token_events_or_features", "horizon": "", "events": 0}])
    rows = []
    for event_type, group in events.groupby("event_type", sort=False):
        for horizon_name, horizon in HORIZONS.items():
            shock = []
            impulse = []
            net = []
            cic_counts = []
            cic_net = []
            for event in group.itertuples(index=False):
                symbol = str(event.cex_symbol)
                start = pd.Timestamp(event.event_available_time)
                window = features[
                    features["symbol"].astype(str).eq(symbol)
                    & features["feature_time"].gt(start)
                    & features["feature_time"].le(start + horizon)
                ]
                if window.empty:
                    continue
                shock.append(float(window["cex_volume_shock"].max()))
                impulse.append(float(window["cex_price_impulse"].max()))
                net.append(float(_num(window, "net20_12h").mean()))
                if trades.empty or "symbol" not in trades.columns:
                    trade_window = pd.DataFrame()
                else:
                    trade_window = trades[
                        trades["symbol"].astype(str).eq(symbol)
                        & trades["entry_time"].gt(start)
                        & trades["entry_time"].le(start + horizon)
                    ]
                cic_counts.append(int(len(trade_window)))
                cic_net.append(float(_num(trade_window, "net20").mean()) if len(trade_window) else np.nan)
            rows.append(
                {
                    "event_type": event_type,
                    "horizon": horizon_name,
                    "events": int(len(group)),
                    "covered_events": int(len(shock)),
                    "cex_volume_shock_rate": float(np.nanmean(shock)) if shock else np.nan,
                    "cex_price_impulse_rate": float(np.nanmean(impulse)) if impulse else np.nan,
                    "future_net20": float(np.nanmean(net)) if net else np.nan,
                    "cic_trades_after_event": int(np.nansum(cic_counts)) if cic_counts else 0,
                    "cic_net20_after_event": _nanmean(cic_net),
                }
            )
    return pd.DataFrame(rows)


def _same_token_random_time(events: pd.DataFrame, ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: V63Config) -> pd.DataFrame:
    if events.empty or ohlcv.empty or features.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for event in events.itertuples(index=False):
        pool_times = ohlcv[ohlcv["cex_symbol"].astype(str).eq(str(event.cex_symbol))]["pool_time"].dropna()
        if pool_times.empty:
            continue
        t = pd.Timestamp(pool_times.iloc[int(rng.integers(0, len(pool_times)))]) + pd.Timedelta(hours=1, minutes=5)
        rows.append({**event._asdict(), "event_available_time": t, "event_time": t - pd.Timedelta(hours=1, minutes=5), "event_type": "same_token_random_time"})
    return _event_response(pd.DataFrame(rows), features, pd.DataFrame())


def _random_token_control(events: pd.DataFrame, features: pd.DataFrame, mapping: pd.DataFrame, cfg: V63Config, same_chain: bool) -> pd.DataFrame:
    if events.empty or features.empty or mapping.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_seed + (2 if same_chain else 1))
    usable = mapping[mapping["mapping_confidence"].astype(str).isin(["A", "B"])].copy()
    rows = []
    for event in events.itertuples(index=False):
        pool = usable[usable["cex_symbol"].astype(str).ne(str(event.cex_symbol))]
        if same_chain:
            pool = pool[pool["chain"].astype(str).eq(str(event.chain))]
        if pool.empty:
            continue
        picked = pool.iloc[int(rng.integers(0, len(pool)))]
        rows.append(
            {
                **event._asdict(),
                "cex_symbol": picked["cex_symbol"],
                "event_type": "same_chain_random_token" if same_chain else "same_day_random_token",
            }
        )
    return _event_response(pd.DataFrame(rows), features, pd.DataFrame())


def _market_level_days() -> pd.Series:
    path = Path("reports/v6_2_onchain_attention_attribution/onchain_attention_days.csv")
    if not path.exists():
        return pd.Series(dtype="datetime64[ns, UTC]")
    days = pd.read_csv(path, usecols=["event_date"])
    return pd.to_datetime(days["event_date"], utc=True, errors="coerce").dt.floor("D").dropna().drop_duplicates()


def _market_level_exclusion(events: pd.DataFrame, features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    market_days = set(_market_level_days())
    local = events.copy()
    local["event_date"] = pd.to_datetime(local["event_time"], utc=True, errors="coerce").dt.floor("D")
    kept = local[~local["event_date"].isin(market_days)].copy()
    response = _event_response(kept, features, trades)
    response.insert(0, "scope", "token_events_ex_market_level_days")
    response.insert(1, "kept_events", int(len(kept)))
    response.insert(2, "excluded_market_level_overlap_events", int(len(local) - len(kept)))
    return response


def _cic_fusion(events: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"module": "P2_all", "bucket": "no_trades", "trades": 0}])
    rows = []
    modules = [
        ("P2_all", trades),
        ("CIC1", trades[trades["candidate"].astype(str).eq("CIC1_beta_extreme")]),
        ("CIC2", trades[trades["candidate"].astype(str).eq("CIC2_beta_broad")]),
    ]
    if "burst_count_so_far" in trades.columns:
        modules.append(("O6_late9_candidates", trades[_num(trades, "burst_count_so_far").ge(9)]))
    for module, sample in modules:
        if sample.empty:
            rows.append({"module": module, "lookback_window": "", "bucket": "insufficient", "trades": 0})
            continue
        for lookback in [pd.Timedelta(hours=4), pd.Timedelta(hours=12), pd.Timedelta(hours=24), pd.Timedelta(hours=48)]:
            flags = []
            for trade in sample.itertuples(index=False):
                symbol = str(trade.symbol)
                entry = pd.Timestamp(trade.entry_time)
                prior = events[
                    events["cex_symbol"].astype(str).eq(symbol)
                    & events["event_available_time"].le(entry)
                    & events["event_available_time"].gt(entry - lookback)
                ]
                flags.append(bool(len(prior)))
            local = sample.copy()
            local["with_prior_token_event"] = flags
            for flag, group in local.groupby("with_prior_token_event", sort=False):
                metrics = _sample_metrics(group)
                rows.append(
                    {
                        "module": module,
                        "lookback_window": str(lookback),
                        "bucket": "with_prior_token_event" if flag else "without_prior_token_event",
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _coverage(mapping: pd.DataFrame, ohlcv: pd.DataFrame, events: pd.DataFrame, pool_coverage: pd.DataFrame, cfg: V63Config) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dataset": "mapping_seed", "rows": int(cfg.max_symbols), "status": "configured_top_n"},
            {"dataset": "token_pool_mapping", "rows": int(len(mapping)), "status": "ok" if len(mapping) else "empty"},
            {"dataset": "mapping_confidence_A_B", "rows": int(mapping["mapping_confidence"].astype(str).isin(["A", "B"]).sum()) if not mapping.empty else 0, "status": "usable_mapping"},
            {"dataset": "pool_ohlcv", "rows": int(len(ohlcv)), "status": "ok" if len(ohlcv) else "missing_or_empty"},
            {"dataset": "token_pool_attention_events", "rows": int(len(events)), "status": "ok" if len(events) else "empty"},
            {"dataset": "pool_ohlcv_symbols", "rows": int(ohlcv["cex_symbol"].nunique()) if not ohlcv.empty else 0, "status": "coverage"},
            {"dataset": "ohlcv_fetch_detail", "rows": int(len(pool_coverage)), "status": "ok"},
        ]
    )


def _write_notes(path: Path, mapping: pd.DataFrame, events: pd.DataFrame, response: pd.DataFrame, fusion: pd.DataFrame) -> None:
    usable = int(mapping["mapping_confidence"].astype(str).isin(["A", "B"]).sum()) if not mapping.empty else 0
    lines = [
        "# v6.3 Token / Pool-Level DEX Attention",
        "",
        "Status: token-level propagation audit only. No strategy, gate, selector, shadow, or real-live permission is changed.",
        "",
        f"Usable A/B mappings: {usable}.",
        f"Token/pool attention events: {int(len(events))}.",
        "Implemented event type: token_pool_volume_spike. Trade-count, buy-pressure, and liquidity-history events remain unavailable in this first pass unless stable historical endpoints are added.",
        "",
    ]
    if not response.empty and "cex_volume_shock_rate" in response.columns:
        valid = response.dropna(subset=["cex_volume_shock_rate"])
        if not valid.empty:
            best = valid.sort_values("cex_volume_shock_rate", ascending=False).iloc[0]
            lines.append(
                f"Best same-symbol CEX response: {best['event_type']} {best['horizon']} "
                f"volume_shock_rate={best['cex_volume_shock_rate']:.4%}, covered_events={int(best['covered_events'])}."
            )
    if not fusion.empty and "net20" in fusion.columns:
        valid = fusion[fusion.get("bucket", pd.Series(dtype=str)).astype(str).eq("with_prior_token_event")].dropna(subset=["net20"])
        if not valid.empty:
            best = valid.sort_values("net20", ascending=False).iloc[0]
            lines.append(f"Best CIC fusion bucket: {best['module']} {best['lookback_window']} net20={best['net20']:.4%}, trades={int(best['trades'])}.")
    lines.extend(
        [
            "",
            "Guardrails:",
            "- Mapping confidence A is reserved for manual verification; automated CoinGecko/GeckoTerminal matches are B at best.",
            "- v6.3 tests token-specific propagation, not market-level DeFiLlama attention.",
            "- Passing random-token and same-token-random-time controls is required before any shadow context is considered.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_v63_token_pool_dex_attention(cfg: V63Config | None = None) -> dict[str, Path]:
    cfg = cfg or V63Config()
    report_root = ensure_dir(cfg.report_root)
    seed = _load_seed_symbols(cfg)
    mapping, mapping_fetch_coverage = _fetch_pool_mapping(seed, cfg)
    ohlcv, pool_coverage = _fetch_ohlcv(mapping, cfg)
    events = _build_token_events(ohlcv, mapping, cfg)
    features = _prepare_features(_read_features(cfg.v61))
    trades = _prepare_trades(_read_p2_trades(cfg.v61.trade_cache_path))
    response = _event_response(events, features, trades)
    fusion = _cic_fusion(events, trades)
    same_token_random = _same_token_random_time(events, ohlcv, features, cfg)
    same_day_random_token = _random_token_control(events, features, mapping, cfg, same_chain=False)
    same_chain_random_token = _random_token_control(events, features, mapping, cfg, same_chain=True)
    ex_market = _market_level_exclusion(events, features, trades)
    coverage = _coverage(mapping, ohlcv, events, pool_coverage, cfg)
    if not mapping_fetch_coverage.empty:
        mapping_fetch_coverage.to_csv(report_root / "mapping_fetch_coverage.csv", index=False)

    outputs = {
        "token_pool_mapping": report_root / "token_pool_mapping.csv",
        "mapping_coverage": report_root / "mapping_coverage.csv",
        "token_pool_ohlcv_coverage": report_root / "token_pool_ohlcv_coverage.csv",
        "token_pool_attention_events": report_root / "token_pool_attention_events.csv",
        "token_to_cex_response_curve": report_root / "token_to_cex_response_curve.csv",
        "token_cic_fusion_summary": report_root / "token_cic_fusion_summary.csv",
        "same_token_random_time_control": report_root / "same_token_random_time_control.csv",
        "same_day_random_token_control": report_root / "same_day_random_token_control.csv",
        "same_chain_random_token_control": report_root / "same_chain_random_token_control.csv",
        "token_events_ex_market_level_days": report_root / "token_events_ex_market_level_days.csv",
        "candidate_notes": report_root / "candidate_notes.md",
    }
    mapping.to_csv(outputs["token_pool_mapping"], index=False)
    coverage.to_csv(outputs["mapping_coverage"], index=False)
    pool_coverage.to_csv(outputs["token_pool_ohlcv_coverage"], index=False)
    events.to_csv(outputs["token_pool_attention_events"], index=False)
    response.to_csv(outputs["token_to_cex_response_curve"], index=False)
    fusion.to_csv(outputs["token_cic_fusion_summary"], index=False)
    same_token_random.to_csv(outputs["same_token_random_time_control"], index=False)
    same_day_random_token.to_csv(outputs["same_day_random_token_control"], index=False)
    same_chain_random_token.to_csv(outputs["same_chain_random_token_control"], index=False)
    ex_market.to_csv(outputs["token_events_ex_market_level_days"], index=False)
    _write_notes(outputs["candidate_notes"], mapping, events, response, fusion)
    return outputs
