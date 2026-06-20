from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from pressure_graph.io import ensure_dir
from pressure_graph.reports.v63_token_pool_dex_attention import _rolling_percentile, _safe_name


REPORT_ROOT = Path("reports/v6_3_token_pool_dex_attention")
MAPPING_PATH = REPORT_ROOT / "token_pool_mapping.csv"
TOKEN_EVENTS_PATH = REPORT_ROOT / "token_pool_attention_events.csv"
TRADE_COVERAGE_PATH = Path("reports/v6_5_token_pool_coverage_expansion/trade_weighted_mapping_coverage.csv")
CACHE_ROOT = Path("data/external/token_dex_attention/dexpaprika/ohlcv_1h")

CHAIN_TO_DEXPAPRIKA = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "arbitrum": "arbitrum",
    "bsc": "bsc",
    "solana": "solana",
    "polygon_pos": "polygon",
    "polygon": "polygon",
    "base": "base",
    "optimism": "optimism",
}


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _get_json(url: str, path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ensure_dir(path.parent)
    response = requests.get(url, timeout=45, headers={"User-Agent": "graph_quant/0.1"})
    if response.status_code == 429:
        time.sleep(3.0)
        response = requests.get(url, timeout=45, headers={"User-Agent": "graph_quant/0.1"})
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    time.sleep(0.25)
    return response.json()


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int = 14) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + pd.Timedelta(days=days), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _load_target_mapping() -> pd.DataFrame:
    mapping = pd.read_csv(MAPPING_PATH, low_memory=False)
    mapping = mapping[mapping["mapping_confidence"].astype(str).isin(["A", "B"])].copy()
    if TRADE_COVERAGE_PATH.exists():
        coverage = pd.read_csv(TRADE_COVERAGE_PATH)
        traded = set(coverage.loc[_num(coverage, "p2_trades").fillna(0).gt(0), "symbol"].astype(str))
        mapping = mapping[mapping["cex_symbol"].astype(str).isin(traded)].copy()
    return mapping.dropna(subset=["pool_address"]).reset_index(drop=True)


def _fetch_ohlcv(mapping: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for item in mapping.itertuples(index=False):
        symbol = str(item.cex_symbol)
        chain = str(item.chain)
        network = CHAIN_TO_DEXPAPRIKA.get(chain, chain)
        pool = str(item.pool_address)
        if not network or not pool or pool == "nan":
            coverage.append({"cex_symbol": symbol, "chain": chain, "network": network, "pool_address": pool, "rows": 0, "status": "missing_pool"})
            continue
        total_rows = 0
        status = "ok"
        for chunk_start, chunk_end in _date_chunks(start_ts, end_ts):
            start_q = chunk_start.strftime("%Y-%m-%d")
            end_q = chunk_end.strftime("%Y-%m-%d")
            cache_path = CACHE_ROOT / network / f"{symbol}_{_safe_name(pool)}_{start_q}_{end_q}.json"
            url = (
                f"https://api.dexpaprika.com/networks/{network}/pools/{pool}/ohlcv"
                f"?start={start_q}&end={end_q}&interval=1h&limit=366"
            )
            try:
                payload = _get_json(url, cache_path)
            except Exception as exc:  # noqa: BLE001
                status = f"error:{type(exc).__name__}"
                continue
            if not isinstance(payload, list):
                status = "unexpected_payload"
                continue
            total_rows += len(payload)
            for row in payload:
                rows.append(
                    {
                        "cex_symbol": symbol,
                        "base_asset": getattr(item, "base_asset", ""),
                        "chain": chain,
                        "network": network,
                        "token_address": str(item.token_address),
                        "pool_address": pool,
                        "pool_time": pd.to_datetime(row.get("time_open"), utc=True, errors="coerce"),
                        "time_close": pd.to_datetime(row.get("time_close"), utc=True, errors="coerce"),
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row.get("close"),
                        "pool_volume_usd": row.get("volume"),
                        "mapping_confidence": str(item.mapping_confidence),
                    }
                )
        coverage.append({"cex_symbol": symbol, "chain": chain, "network": network, "pool_address": pool, "rows": total_rows, "status": status})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pool_time"] = pd.to_datetime(out["pool_time"], utc=True, errors="coerce")
        out["time_close"] = pd.to_datetime(out["time_close"], utc=True, errors="coerce")
        out = out.dropna(subset=["pool_time"]).drop_duplicates(["cex_symbol", "pool_address", "pool_time"]).sort_values(["cex_symbol", "pool_time"])
    return out, pd.DataFrame(coverage)


def _build_events(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for symbol, group in ohlcv.groupby("cex_symbol", sort=False):
        local = group.sort_values("pool_time").copy()
        volume = _num(local, "pool_volume_usd")
        lookback = 120
        min_periods = 48
        mean = volume.shift(1).rolling(lookback, min_periods=min_periods).mean()
        std = volume.shift(1).rolling(lookback, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
        local["zscore"] = (volume - mean) / std
        local["percentile"] = _rolling_percentile(volume, lookback, min_periods)
        mask = local["zscore"].ge(2.0) | local["percentile"].ge(0.95)
        for row in local[mask.fillna(False)].itertuples(index=False):
            event_time = pd.Timestamp(row.pool_time)
            available = pd.Timestamp(row.time_close) + pd.Timedelta(minutes=5)
            rows.append(
                {
                    "event_id": f"{symbol}|{row.pool_address}|{event_time.isoformat()}|dexpaprika_pool_volume_spike",
                    "cex_symbol": symbol,
                    "base_asset": getattr(row, "base_asset", ""),
                    "chain": getattr(row, "chain"),
                    "token_address": getattr(row, "token_address"),
                    "pool_address": getattr(row, "pool_address"),
                    "event_time": event_time,
                    "event_available_time": available,
                    "event_type": "token_pool_volume_spike",
                    "raw_value": float(row.pool_volume_usd),
                    "zscore": float(row.zscore) if pd.notna(row.zscore) else np.nan,
                    "percentile": float(row.percentile) if pd.notna(row.percentile) else np.nan,
                    "lookback_window": lookback,
                    "mapping_confidence": getattr(row, "mapping_confidence", ""),
                    "source": "dexpaprika_pool_ohlcv_1h",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    report_root = ensure_dir(REPORT_ROOT)
    mapping = _load_target_mapping()
    ohlcv, coverage = _fetch_ohlcv(mapping, start="2025-06-01", end="2025-12-01")
    events = _build_events(ohlcv)
    existing = pd.read_csv(TOKEN_EVENTS_PATH, low_memory=False) if TOKEN_EVENTS_PATH.exists() else pd.DataFrame()
    if not existing.empty and "source" in existing.columns:
        existing = existing[existing["source"].astype(str).ne("dexpaprika_pool_ohlcv_1h")]
    merged = pd.concat([existing, events], ignore_index=True) if not existing.empty else events
    if not merged.empty:
        merged = merged.drop_duplicates("event_id").sort_values(["event_time", "cex_symbol"])
    ohlcv.to_csv(report_root / "token_pool_dexpaprika_ohlcv_1h.csv", index=False)
    coverage.to_csv(report_root / "token_pool_dexpaprika_ohlcv_coverage.csv", index=False)
    events.to_csv(report_root / "token_pool_dexpaprika_attention_events.csv", index=False)
    merged.to_csv(TOKEN_EVENTS_PATH, index=False)
    print(f"target_mapped_symbols={len(mapping)}")
    print(f"dexpaprika_ohlcv_rows={len(ohlcv)}")
    print(f"dexpaprika_events={len(events)}")
    print(f"merged_token_events={len(merged)}")
    if not coverage.empty:
        print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
