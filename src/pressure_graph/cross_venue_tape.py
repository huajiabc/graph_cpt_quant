"""Cross-venue public-trade normalization, minute aggregation, and coverage QA."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pressure_graph.io import ensure_dir


TAPE_ROOT = Path("data/orderflow/v9_6_cross_venue")
REPORT_ROOT = Path("reports/v9_6_cross_venue_tape")
EXCHANGES = ("binance", "bybit")

BAR_COLUMNS = [
    "exchange",
    "symbol",
    "bar_open_time",
    "open",
    "high",
    "low",
    "close",
    "trade_count",
    "volume",
    "turnover",
    "buy_volume",
    "sell_volume",
    "buy_turnover",
    "sell_turnover",
    "taker_buy_ratio",
    "cvd_delta_volume",
    "cvd_delta_turnover",
    "buy_sell_imbalance",
    "first_event_time",
    "last_event_time",
    "first_event_id",
    "last_event_id",
    "stream_session_id",
    "ingest_time",
    "event_lag_seconds",
    "bar_complete",
]


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _trade(
    exchange: str,
    symbol: str,
    timestamp: object,
    event_id: object,
    price: object,
    size: object,
    side: object,
) -> dict[str, Any] | None:
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    px = pd.to_numeric(price, errors="coerce")
    qty = pd.to_numeric(size, errors="coerce")
    if pd.isna(ts) or not np.isfinite(px) or not np.isfinite(qty) or px <= 0 or qty < 0:
        return None
    side_text = str(side).capitalize()
    if side_text not in {"Buy", "Sell"}:
        return None
    return {
        "exchange": exchange,
        "symbol": str(symbol).upper(),
        "timestamp": ts,
        "event_id": str(event_id),
        "price": float(px),
        "size": float(qty),
        "turnover": float(px * qty),
        "side": side_text,
    }


def parse_binance_message(payload: str | dict[str, Any]) -> list[dict[str, Any]]:
    message = json.loads(payload) if isinstance(payload, str) else payload
    data = message.get("data", message) if isinstance(message, dict) else {}
    if not isinstance(data, dict) or data.get("e") != "aggTrade":
        return []
    # m=True means the buyer was maker, therefore the aggressor was a seller.
    row = _trade(
        "binance",
        data.get("s", ""),
        pd.to_datetime(data.get("T"), unit="ms", utc=True, errors="coerce"),
        data.get("a", ""),
        data.get("p"),
        data.get("q"),
        "Sell" if bool(data.get("m")) else "Buy",
    )
    return [row] if row else []


def parse_bybit_message(payload: str | dict[str, Any]) -> list[dict[str, Any]]:
    message = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(message, dict) or not str(message.get("topic", "")).startswith("publicTrade."):
        return []
    rows = []
    for item in message.get("data", []):
        row = _trade(
            "bybit",
            item.get("s", ""),
            pd.to_datetime(item.get("T"), unit="ms", utc=True, errors="coerce"),
            item.get("i", ""),
            item.get("p"),
            item.get("v"),
            item.get("S"),
        )
        if row:
            rows.append(row)
    return rows


@dataclass
class _BarState:
    exchange: str
    symbol: str
    bar_open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    trade_count: int
    volume: float
    turnover: float
    buy_volume: float
    sell_volume: float
    buy_turnover: float
    sell_turnover: float
    first_event_time: pd.Timestamp
    last_event_time: pd.Timestamp
    first_event_id: str
    last_event_id: str
    stream_session_id: str
    max_event_lag_seconds: float


class MinuteBarAccumulator:
    def __init__(self) -> None:
        self._bars: dict[tuple[str, str, pd.Timestamp, str], _BarState] = {}

    def add_trade(
        self,
        trade: dict[str, Any],
        session_id: str,
        received_at: object | None = None,
    ) -> None:
        timestamp = _utc(trade["timestamp"])
        receive_time = _utc(received_at) if received_at is not None else pd.Timestamp.now(tz="UTC")
        event_lag_seconds = max(float((receive_time - timestamp).total_seconds()), 0.0)
        minute = timestamp.floor("1min")
        exchange = str(trade["exchange"])
        symbol = str(trade["symbol"]).upper()
        key = (exchange, symbol, minute, session_id)
        price = float(trade["price"])
        size = float(trade["size"])
        turnover = float(trade.get("turnover", price * size))
        is_buy = str(trade["side"]).lower() == "buy"
        event_id = str(trade["event_id"])
        state = self._bars.get(key)
        if state is None:
            state = _BarState(
                exchange=exchange,
                symbol=symbol,
                bar_open_time=minute,
                open=price,
                high=price,
                low=price,
                close=price,
                trade_count=0,
                volume=0.0,
                turnover=0.0,
                buy_volume=0.0,
                sell_volume=0.0,
                buy_turnover=0.0,
                sell_turnover=0.0,
                first_event_time=timestamp,
                last_event_time=timestamp,
                first_event_id=event_id,
                last_event_id=event_id,
                stream_session_id=session_id,
                max_event_lag_seconds=event_lag_seconds,
            )
            self._bars[key] = state
        if timestamp < state.first_event_time:
            state.first_event_time = timestamp
            state.first_event_id = event_id
            state.open = price
        if timestamp >= state.last_event_time:
            state.last_event_time = timestamp
            state.last_event_id = event_id
            state.close = price
        state.high = max(state.high, price)
        state.low = min(state.low, price)
        state.trade_count += 1
        state.max_event_lag_seconds = max(state.max_event_lag_seconds, event_lag_seconds)
        state.volume += size
        state.turnover += turnover
        if is_buy:
            state.buy_volume += size
            state.buy_turnover += turnover
        else:
            state.sell_volume += size
            state.sell_turnover += turnover

    def drain(self, before: object, *, bar_complete: bool = True) -> pd.DataFrame:
        cutoff = _utc(before)
        keys = [key for key, state in self._bars.items() if state.bar_open_time < cutoff]
        ingest_time = pd.Timestamp.now(tz="UTC")
        rows = []
        for key in keys:
            state = self._bars.pop(key)
            imbalance = (
                (state.buy_turnover - state.sell_turnover) / state.turnover
                if state.turnover > 0
                else np.nan
            )
            rows.append(
                {
                    **state.__dict__,
                    "taker_buy_ratio": state.buy_turnover / state.turnover if state.turnover else np.nan,
                    "cvd_delta_volume": state.buy_volume - state.sell_volume,
                    "cvd_delta_turnover": state.buy_turnover - state.sell_turnover,
                    "buy_sell_imbalance": imbalance,
                    "ingest_time": ingest_time,
                    "event_lag_seconds": state.max_event_lag_seconds,
                    "bar_complete": bool(bar_complete),
                }
            )
        return pd.DataFrame(rows, columns=BAR_COLUMNS).sort_values(
            ["bar_open_time", "exchange", "symbol"]
        ) if rows else pd.DataFrame(columns=BAR_COLUMNS)


def write_bar_fragment(
    bars: pd.DataFrame,
    root: Path = TAPE_ROOT,
    fragment_id: str | None = None,
) -> Path | None:
    if bars.empty:
        return None
    data = bars.copy()
    data["bar_open_time"] = pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce")
    day = data["bar_open_time"].min().strftime("%Y-%m-%d")
    fragment = fragment_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
    path = ensure_dir(root / "bars_1m" / day) / f"{fragment}.parquet"
    data.to_parquet(path, index=False)
    return path


def read_bar_fragments(
    root: Path = TAPE_ROOT,
    start: object | None = None,
    end: object | None = None,
) -> pd.DataFrame:
    bars_root = root / "bars_1m"
    if not bars_root.exists():
        paths: list[Path] = []
    elif start is not None or end is not None:
        start_time = _utc(start) if start is not None else _utc(end) - pd.Timedelta(days=1)
        end_time = _utc(end) if end is not None else start_time + pd.Timedelta(days=1)
        days = pd.date_range(
            start_time.floor("D"), end_time.floor("D"), freq="D", tz="UTC"
        )
        paths = sorted(
            path
            for day in days
            for path in (bars_root / day.strftime("%Y-%m-%d")).glob("*.parquet")
        )
    else:
        paths = sorted(bars_root.glob("*/*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=BAR_COLUMNS)
    data = pd.concat(frames, ignore_index=True)
    data["bar_open_time"] = pd.to_datetime(data["bar_open_time"], utc=True, errors="coerce")
    if start is not None:
        data = data[data["bar_open_time"].ge(_utc(start))]
    if end is not None:
        data = data[data["bar_open_time"].lt(_utc(end))]
    return data.reset_index(drop=True)


def combine_bar_fragments(fragments: pd.DataFrame) -> pd.DataFrame:
    if fragments.empty:
        return pd.DataFrame(columns=[*BAR_COLUMNS, "fragment_count"])
    data = fragments.copy()
    for col in ["bar_open_time", "first_event_time", "last_event_time", "ingest_time"]:
        data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    keys = ["exchange", "symbol", "bar_open_time"]
    rows = []
    sum_cols = [
        "trade_count",
        "volume",
        "turnover",
        "buy_volume",
        "sell_volume",
        "buy_turnover",
        "sell_turnover",
    ]
    for key, group in data.groupby(keys, sort=False):
        first = group.sort_values("first_event_time").iloc[0]
        last = group.sort_values("last_event_time").iloc[-1]
        turnover = float(pd.to_numeric(group["turnover"], errors="coerce").sum())
        buy_turnover = float(pd.to_numeric(group["buy_turnover"], errors="coerce").sum())
        sell_turnover = float(pd.to_numeric(group["sell_turnover"], errors="coerce").sum())
        row = {
            "exchange": key[0],
            "symbol": key[1],
            "bar_open_time": key[2],
            "open": float(first["open"]),
            "high": float(pd.to_numeric(group["high"], errors="coerce").max()),
            "low": float(pd.to_numeric(group["low"], errors="coerce").min()),
            "close": float(last["close"]),
            **{col: float(pd.to_numeric(group[col], errors="coerce").sum()) for col in sum_cols},
            "taker_buy_ratio": buy_turnover / turnover if turnover else np.nan,
            "cvd_delta_volume": float(pd.to_numeric(group["buy_volume"], errors="coerce").sum() - pd.to_numeric(group["sell_volume"], errors="coerce").sum()),
            "cvd_delta_turnover": buy_turnover - sell_turnover,
            "buy_sell_imbalance": (buy_turnover - sell_turnover) / turnover if turnover else np.nan,
            "first_event_time": first["first_event_time"],
            "last_event_time": last["last_event_time"],
            "first_event_id": first["first_event_id"],
            "last_event_id": last["last_event_id"],
            "stream_session_id": "|".join(sorted(group["stream_session_id"].astype(str).unique())),
            "ingest_time": group["ingest_time"].max(),
            "event_lag_seconds": float(pd.to_numeric(group["event_lag_seconds"], errors="coerce").max()),
            "bar_complete": bool(group["bar_complete"].fillna(False).all()),
            "fragment_count": int(len(group)),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def coverage_summary(
    bars: pd.DataFrame,
    start: object,
    end: object,
    exchanges: tuple[str, ...] = EXCHANGES,
) -> pd.DataFrame:
    start_time = _utc(start).floor("1min")
    end_time = _utc(end).floor("1min")
    expected = max(int((end_time - start_time) / pd.Timedelta(minutes=1)), 0)
    if bars.empty or expected == 0:
        return pd.DataFrame()
    data = combine_bar_fragments(bars) if "fragment_count" not in bars.columns else bars.copy()
    data = data[data["bar_open_time"].ge(start_time) & data["bar_open_time"].lt(end_time)]
    rows = []
    symbols = sorted(data["symbol"].astype(str).unique())
    for symbol in symbols:
        local = data[data["symbol"].eq(symbol)]
        minute_sets = {
            exchange: set(local.loc[local["exchange"].eq(exchange), "bar_open_time"])
            for exchange in exchanges
        }
        synchronized = set.intersection(*(minute_sets[exchange] for exchange in exchanges))
        for exchange in exchanges:
            sample = local[local["exchange"].eq(exchange)].sort_values("bar_open_time")
            observed = sample["bar_open_time"].nunique()
            gaps = sample["bar_open_time"].drop_duplicates().diff().dt.total_seconds().div(60).sub(1)
            rows.append(
                {
                    "exchange": exchange,
                    "symbol": symbol,
                    "start": start_time,
                    "end": end_time,
                    "expected_minutes": expected,
                    "observed_minutes": observed,
                    "coverage_ratio": observed / expected,
                    "synchronized_minutes": len(synchronized),
                    "synchronized_ratio": len(synchronized) / expected,
                    "max_internal_gap_minutes": float(gaps.max()) if gaps.notna().any() else 0.0,
                    "stale_event_minutes": int(pd.to_numeric(sample.get("event_lag_seconds"), errors="coerce").gt(10).sum()),
                }
            )
    return pd.DataFrame(rows)


def write_coverage_report(
    root: Path = TAPE_ROOT,
    report_root: Path = REPORT_ROOT,
    lookback_hours: int = 24,
    now: object | None = None,
) -> dict[str, Path]:
    end = _utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    end = end.floor("1min")
    start = end - pd.Timedelta(hours=lookback_hours)
    fragments = read_bar_fragments(root, start, end)
    combined = combine_bar_fragments(fragments)
    coverage = coverage_summary(combined, start, end)
    target = ensure_dir(report_root)
    outputs = {
        "combined_bars": target / "latest_combined_bars.parquet",
        "coverage": target / "coverage.csv",
        "status": target / "status.md",
    }
    combined.to_parquet(outputs["combined_bars"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    min_sync = float(coverage["synchronized_ratio"].min()) if not coverage.empty else 0.0
    lines = [
        "# v9.6 Cross-Venue Tape Status",
        "",
        "Status: data collection only; no trading action is allowed.",
        f"- window: {start.isoformat()} -> {end.isoformat()}",
        f"- fragments: {len(fragments)}; combined bars: {len(combined)}",
        f"- symbols: {combined['symbol'].nunique() if not combined.empty else 0}",
        f"- minimum synchronized ratio: {min_sync:.2%}",
        f"- research-ready coverage: {bool(min_sync >= 0.95)}",
        "- alpha verdict remains DATA_ACCUMULATING until the preregistered 90-day gate.",
    ]
    outputs["status"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outputs


def select_common_symbols(
    feature_path: Path,
    binance_symbols: set[str],
    max_symbols: int = 20,
    core_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
) -> list[str]:
    if not feature_path.exists():
        return [symbol for symbol in core_symbols if symbol in binance_symbols][:max_symbols]
    features = pd.read_parquet(feature_path)
    if features.empty or "symbol" not in features.columns:
        return []
    time_col = "feature_time" if "feature_time" in features.columns else "bar_close_time"
    features[time_col] = pd.to_datetime(features[time_col], utc=True, errors="coerce")
    latest = features[time_col].max()
    features = features[features[time_col].ge(latest - pd.Timedelta(days=7))].copy()
    rank_col = "dynamic_all_rank" if "dynamic_all_rank" in features.columns else "turnover_rank_30d"
    features[rank_col] = pd.to_numeric(features.get(rank_col), errors="coerce")
    ranked = (
        features[features["symbol"].astype(str).isin(binance_symbols)]
        .groupby("symbol", as_index=False)[rank_col]
        .min()
        .sort_values([rank_col, "symbol"])
    )
    ordered = [symbol for symbol in core_symbols if symbol in binance_symbols]
    ordered.extend(ranked["symbol"].dropna().astype(str).tolist())
    return list(dict.fromkeys(ordered))[:max_symbols]
