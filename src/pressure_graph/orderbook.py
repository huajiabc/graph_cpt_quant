from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.clients.bybit import BybitClient
from pressure_graph.config import ExperimentConfig
from pressure_graph.io import ensure_dir, read_parquet, write_parquet
from pressure_graph.orderflow import DEMAND_QUEUE_PATH, LIVE_FEATURE_PATH, _coerce_timestamp_series


REPORT_ROOT = Path("reports/v0_8_5_orderbook_snapshot")
ORDERBOOK_ROOT = Path("data/orderbook/v0_8_5/bybit")

DEPTH_BPS = (5, 10, 20, 25, 50)
IMPACT_NOTIONALS = (10_000, 50_000, 100_000)

LEVEL_COLUMNS = [
    "exchange",
    "symbol",
    "snapshot_time",
    "exchange_ts",
    "update_id",
    "seq",
    "side",
    "level",
    "price",
    "size",
    "notional",
]

FEATURE_COLUMNS = [
    "exchange",
    "symbol",
    "snapshot_time",
    "exchange_ts",
    "update_id",
    "seq",
    "bid_levels_available",
    "ask_levels_available",
    "levels_available",
    "best_bid",
    "best_ask",
    "mid",
    "spread_bps",
    "bid_depth_5bp",
    "ask_depth_5bp",
    "imbalance_5bp",
    "bid_depth_10bp",
    "ask_depth_10bp",
    "imbalance_10bp",
    "bid_depth_20bp",
    "ask_depth_20bp",
    "imbalance_20bp",
    "bid_depth_25bp",
    "ask_depth_25bp",
    "imbalance_25bp",
    "bid_depth_50bp",
    "ask_depth_50bp",
    "imbalance_50bp",
    "buy_impact_10k",
    "buy_impact_50k",
    "buy_impact_100k",
    "sell_impact_10k",
    "sell_impact_50k",
    "sell_impact_100k",
    "upside_vacuum_25bp",
    "downside_liquidity_risk_25bp",
    "entry_book_quality_score",
    "top5_bid_notional",
    "top5_ask_notional",
    "top5_imbalance",
    "ask_wall_20bp_ratio",
    "bid_wall_20bp_ratio",
]


@dataclass(frozen=True)
class OrderbookRunConfig:
    report_root: Path = REPORT_ROOT
    orderbook_root: Path = ORDERBOOK_ROOT
    demand_queue_path: Path = DEMAND_QUEUE_PATH
    live_feature_path: Path = LIVE_FEATURE_PATH
    top_n: int = 50
    max_symbols: int | None = 10
    depth_limit: int = 200
    retain_days: int = 7
    core_reference_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _empty_levels() -> pd.DataFrame:
    return pd.DataFrame(columns=LEVEL_COLUMNS)


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _level_cache_path(root: Path, symbol: str) -> Path:
    return root / "snapshots" / f"{symbol}.parquet"


def _feature_cache_path(root: Path, symbol: str) -> Path:
    return root / "features" / f"{symbol}.parquet"


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_parquet(path)
    except Exception:
        return pd.DataFrame()


def select_orderbook_symbols(
    demand_queue_path: Path = DEMAND_QUEUE_PATH,
    live_feature_path: Path = LIVE_FEATURE_PATH,
    *,
    top_n: int = 50,
    max_symbols: int | None = 10,
    core_reference_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
) -> list[str]:
    demand_symbols: list[str] = []
    if demand_queue_path.exists():
        queue = read_parquet(demand_queue_path)
        if not queue.empty and "symbol" in queue.columns:
            data = queue.copy()
            data["window_end"] = _coerce_timestamp_series(data.get("window_end"))
            if "priority" not in data.columns:
                data["priority"] = "P3"
            if "status" not in data.columns:
                data["status"] = "pending"
            status_order = {"pending": 0, "partial": 1, "failed_network": 2, "done": 3}
            data["_status_order"] = data["status"].astype(str).map(status_order).fillna(2)
            data = data.sort_values(
                ["priority", "_status_order", "window_end", "symbol"],
                ascending=[True, True, False, True],
            )
            demand_symbols.extend(data["symbol"].dropna().astype(str).tolist())

    feature_symbols: list[str] = []
    if live_feature_path.exists():
        features = read_parquet(live_feature_path)
        if not features.empty and "symbol" in features.columns:
            rank_col = "dynamic_all_rank" if "dynamic_all_rank" in features.columns else "turnover_rank_30d"
            data = features.copy()
            data[rank_col] = pd.to_numeric(data.get(rank_col), errors="coerce")
            time_col = "feature_time" if "feature_time" in data.columns else "bar_close_time"
            if time_col in data.columns:
                data[time_col] = _coerce_timestamp_series(data[time_col])
                latest = data[time_col].max()
                data = data[data[time_col] >= latest - pd.Timedelta(days=1)]
            ranked = (
                data[data[rank_col] <= top_n]
                .groupby("symbol", as_index=False)[rank_col]
                .min()
                .sort_values([rank_col, "symbol"])
            )
            feature_symbols.extend(ranked["symbol"].dropna().astype(str).tolist())

    unique = list(dict.fromkeys(demand_symbols + list(core_reference_symbols) + feature_symbols))
    return unique[:max_symbols] if max_symbols else unique


def normalize_orderbook_snapshot(
    symbol: str,
    payload: dict[str, Any],
    *,
    received_at: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot_time = received_at or pd.Timestamp.now(tz="UTC")
    exchange_ts = pd.to_datetime(pd.to_numeric(payload.get("ts"), errors="coerce"), unit="ms", utc=True)
    update_id = payload.get("u")
    seq = payload.get("seq")
    rows: list[dict[str, object]] = []
    for side_name, side_key in [("bid", "b"), ("ask", "a")]:
        for level, pair in enumerate(payload.get(side_key, []) or [], start=1):
            if len(pair) < 2:
                continue
            price = float(pair[0])
            size = float(pair[1])
            rows.append(
                {
                    "exchange": "bybit",
                    "symbol": symbol,
                    "snapshot_time": snapshot_time,
                    "exchange_ts": exchange_ts,
                    "update_id": update_id,
                    "seq": seq,
                    "side": side_name,
                    "level": level,
                    "price": price,
                    "size": size,
                    "notional": price * size,
                }
            )
    levels = pd.DataFrame(rows, columns=LEVEL_COLUMNS)
    features = compute_orderbook_features(levels)
    return levels, features


def compute_orderbook_features(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return _empty_features()
    data = levels.copy()
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data["size"] = pd.to_numeric(data["size"], errors="coerce")
    data["notional"] = pd.to_numeric(data.get("notional", data["price"] * data["size"]), errors="coerce")
    bids = data[data["side"].eq("bid")].sort_values("price", ascending=False)
    asks = data[data["side"].eq("ask")].sort_values("price", ascending=True)
    if bids.empty or asks.empty:
        return _empty_features()
    best_bid = float(bids.iloc[0]["price"])
    best_ask = float(asks.iloc[0]["price"])
    mid = (best_bid + best_ask) / 2.0
    row: dict[str, object] = {
        "exchange": data.iloc[0].get("exchange", "bybit"),
        "symbol": data.iloc[0]["symbol"],
        "snapshot_time": data.iloc[0]["snapshot_time"],
        "exchange_ts": data.iloc[0].get("exchange_ts"),
        "update_id": data.iloc[0].get("update_id"),
        "seq": data.iloc[0].get("seq"),
        "bid_levels_available": int(len(bids)),
        "ask_levels_available": int(len(asks)),
        "levels_available": int(len(bids) + len(asks)),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_bps": (best_ask - best_bid) / mid * 10000 if mid else np.nan,
    }
    for bps in DEPTH_BPS:
        bid_cutoff = mid * (1 - bps / 10000)
        ask_cutoff = mid * (1 + bps / 10000)
        bid_depth = float(bids.loc[bids["price"] >= bid_cutoff, "notional"].sum())
        ask_depth = float(asks.loc[asks["price"] <= ask_cutoff, "notional"].sum())
        denom = bid_depth + ask_depth
        row[f"bid_depth_{bps}bp"] = bid_depth
        row[f"ask_depth_{bps}bp"] = ask_depth
        row[f"imbalance_{bps}bp"] = (bid_depth - ask_depth) / denom if denom else np.nan
    for notional in IMPACT_NOTIONALS:
        suffix = f"{int(notional / 1000)}k"
        row[f"buy_impact_{suffix}"] = _market_impact_bps(asks, mid, side="buy", target_notional=notional)
        row[f"sell_impact_{suffix}"] = _market_impact_bps(bids, mid, side="sell", target_notional=notional)
    bid25 = float(row.get("bid_depth_25bp", np.nan))
    ask25 = float(row.get("ask_depth_25bp", np.nan))
    row["upside_vacuum_25bp"] = np.log1p(bid25) - np.log1p(ask25) if np.isfinite(bid25) and np.isfinite(ask25) else np.nan
    row["downside_liquidity_risk_25bp"] = np.log1p(ask25) - np.log1p(bid25) if np.isfinite(bid25) and np.isfinite(ask25) else np.nan
    row["entry_book_quality_score"] = (
        float(row.get("imbalance_25bp", 0.0) or 0.0)
        + float(row.get("upside_vacuum_25bp", 0.0) or 0.0)
        - float(row.get("spread_bps", 0.0) or 0.0) / 10.0
        - max(float(row.get("downside_liquidity_risk_25bp", 0.0) or 0.0), 0.0)
    )
    top5_bid = float(bids.head(5)["notional"].sum())
    top5_ask = float(asks.head(5)["notional"].sum())
    row["top5_bid_notional"] = top5_bid
    row["top5_ask_notional"] = top5_ask
    row["top5_imbalance"] = (top5_bid - top5_ask) / (top5_bid + top5_ask) if top5_bid + top5_ask else np.nan
    row["ask_wall_20bp_ratio"] = _wall_ratio(asks, mid, side="ask", bps=20)
    row["bid_wall_20bp_ratio"] = _wall_ratio(bids, mid, side="bid", bps=20)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _market_impact_bps(levels: pd.DataFrame, mid: float, *, side: str, target_notional: float) -> float:
    if levels.empty or not mid or target_notional <= 0:
        return float("nan")
    remaining = float(target_notional)
    spent = 0.0
    base_qty = 0.0
    for row in levels.itertuples(index=False):
        price = float(getattr(row, "price"))
        available = float(getattr(row, "notional"))
        consume = min(available, remaining)
        if consume <= 0 or price <= 0:
            continue
        spent += consume
        base_qty += consume / price
        remaining -= consume
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or base_qty <= 0:
        return float("nan")
    vwap = spent / base_qty
    if side == "buy":
        return float((vwap - mid) / mid * 10000)
    return float((mid - vwap) / mid * 10000)


def _wall_ratio(levels: pd.DataFrame, mid: float, *, side: str, bps: int) -> float:
    if levels.empty or not mid:
        return float("nan")
    cutoff = mid * (1 + bps / 10000) if side == "ask" else mid * (1 - bps / 10000)
    sample = levels[levels["price"] <= cutoff] if side == "ask" else levels[levels["price"] >= cutoff]
    total = float(sample["notional"].sum())
    if not total:
        return float("nan")
    return float(sample["notional"].max() / total)


def update_orderbook_snapshot_cache(
    client: BybitClient,
    symbols: list[str],
    orderbook_root: Path,
    *,
    depth_limit: int = 200,
    retain_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    level_dir = ensure_dir(orderbook_root / "snapshots")
    feature_dir = ensure_dir(orderbook_root / "features")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=retain_days)
    status_rows: list[dict[str, object]] = []
    feature_rows: list[pd.DataFrame] = []
    fetched_at = pd.Timestamp.now(tz="UTC")
    for symbol in symbols:
        error = ""
        levels = _empty_levels()
        features = _empty_features()
        try:
            payload = client.orderbook(symbol, depth_limit)
            levels, features = normalize_orderbook_snapshot(symbol, payload, received_at=fetched_at)
        except Exception as exc:  # pragma: no cover - live network failure only
            error = str(exc)

        level_path = level_dir / f"{symbol}.parquet"
        feature_path = feature_dir / f"{symbol}.parquet"
        old_levels = _read_optional_parquet(level_path)
        old_features = _read_optional_parquet(feature_path)
        combined_levels = pd.concat([old_levels, levels], ignore_index=True) if not old_levels.empty else levels
        combined_features = (
            pd.concat([old_features, features], ignore_index=True) if not old_features.empty else features
        )
        if not combined_levels.empty:
            combined_levels["snapshot_time"] = _coerce_timestamp_series(combined_levels["snapshot_time"])
            combined_levels = combined_levels[combined_levels["snapshot_time"] >= cutoff].drop_duplicates(
                ["symbol", "snapshot_time", "side", "level"]
            )
        if not combined_features.empty:
            combined_features["snapshot_time"] = _coerce_timestamp_series(combined_features["snapshot_time"])
            combined_features = combined_features[combined_features["snapshot_time"] >= cutoff].drop_duplicates(
                ["symbol", "snapshot_time"]
            )
        write_parquet(combined_levels, level_path)
        write_parquet(combined_features, feature_path)
        if not features.empty:
            feature_rows.append(features)
        status_rows.append(
            {
                "symbol": symbol,
                "fetched_at": fetched_at,
                "level_rows": int(len(levels)),
                "feature_rows": int(len(features)),
                "cached_level_rows": int(len(combined_levels)),
                "cached_feature_rows": int(len(combined_features)),
                "best_bid": float(features["best_bid"].iloc[0]) if not features.empty else np.nan,
                "best_ask": float(features["best_ask"].iloc[0]) if not features.empty else np.nan,
                "spread_bps": float(features["spread_bps"].iloc[0]) if not features.empty else np.nan,
                "error": error,
            }
        )
    latest_features = pd.concat(feature_rows, ignore_index=True) if feature_rows else _empty_features()
    return pd.DataFrame(status_rows), latest_features


def write_v085_orderbook_snapshot(
    base_config: ExperimentConfig,
    run_config: OrderbookRunConfig | None = None,
    symbols: list[str] | None = None,
) -> dict[str, Path]:
    cfg = run_config or OrderbookRunConfig()
    report_root = ensure_dir(cfg.report_root)
    orderbook_root = ensure_dir(cfg.orderbook_root)
    selected = symbols or select_orderbook_symbols(
        cfg.demand_queue_path,
        cfg.live_feature_path,
        top_n=cfg.top_n,
        max_symbols=cfg.max_symbols,
        core_reference_symbols=cfg.core_reference_symbols,
    )
    client = BybitClient(
        str(base_config.exchanges.bybit.base_url),
        base_config.exchanges.bybit.category,
    )
    try:
        cache_status, features = update_orderbook_snapshot_cache(
            client,
            selected,
            orderbook_root,
            depth_limit=cfg.depth_limit,
            retain_days=cfg.retain_days,
        )
    finally:
        client.close()
    outputs = {
        "cache_status": report_root / "orderbook_cache_status.csv",
        "snapshot_features": report_root / "orderbook_snapshot_features.csv",
        "current_status": report_root / "current_status.md",
    }
    cache_status.to_csv(outputs["cache_status"], index=False)
    features.to_csv(outputs["snapshot_features"], index=False)
    _write_current_status(outputs["current_status"], selected, cache_status, features, cfg)
    return outputs


def _write_current_status(
    path: Path,
    symbols: list[str],
    cache_status: pd.DataFrame,
    features: pd.DataFrame,
    cfg: OrderbookRunConfig,
) -> None:
    errors = cache_status["error"].astype(str).ne("").sum() if "error" in cache_status.columns else 0
    lines = [
        "# v0.8.5 Orderbook Snapshot Status",
        "",
        "- mode: shadow_only",
        "- source: Bybit REST orderbook snapshot",
        "- trading_decision_impact: none",
        "- real_live_allowed: false",
        f"- selected_symbols: {len(symbols)}",
        f"- depth_limit: {cfg.depth_limit}",
        f"- retain_days: {cfg.retain_days}",
        f"- cache_errors: {int(errors)}",
        f"- snapshot_features: {len(features)}",
    ]
    if not features.empty:
        spread = pd.to_numeric(features["spread_bps"], errors="coerce")
        lines.extend(
            [
                f"- avg_spread_bps: {float(spread.mean()):.4f}",
                f"- median_spread_bps: {float(spread.median()):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- REST snapshots are suitable for live liquidity/execution-quality diagnostics.",
            "- They are not historical orderbook-path reconstruction and do not permit real-live execution.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "ORDERBOOK_ROOT",
    "REPORT_ROOT",
    "OrderbookRunConfig",
    "compute_orderbook_features",
    "normalize_orderbook_snapshot",
    "select_orderbook_symbols",
    "update_orderbook_snapshot_cache",
    "write_v085_orderbook_snapshot",
]
