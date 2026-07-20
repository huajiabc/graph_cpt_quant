from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pressure_graph.binance_um_book_depth_history import (
    BinanceBookDepthConfig,
    backfill_binance_book_depth,
)
from pressure_graph.binance_um_book_depth_hourly import (
    BinanceBookDepthHourlyConfig,
    build_hourly_book_depth_panel,
)
from pressure_graph.clients import BybitClient
from pressure_graph.config import load_config
from pressure_graph.io import ensure_dir, write_parquet
from pressure_graph.paper_live.q90 import (
    FROZEN_SYMBOLS,
    load_q90_live_config,
    next_q90_book_day,
    write_q90_ledgers,
)


def _merge_daily(
    existing: dict[str, pd.DataFrame], daily_root: Path
) -> None:
    for symbol, historical in existing.items():
        path = daily_root / f"{symbol}.parquet"
        current = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        frames = [frame for frame in (historical, current) if not frame.empty]
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged["source_day"] = pd.to_datetime(
            merged["source_day"], utc=True, errors="coerce"
        )
        merged = (
            merged.dropna(subset=["source_day"])
            .drop_duplicates("source_day", keep="last")
            .sort_values("source_day")
            .reset_index(drop=True)
        )
        write_parquet(merged, path)


def _ready_days(daily_root: Path) -> set[pd.Timestamp]:
    per_symbol: list[set[pd.Timestamp]] = []
    for symbol in FROZEN_SYMBOLS:
        path = daily_root / f"{symbol}.parquet"
        if not path.exists():
            return set()
        frame = pd.read_parquet(path, columns=["source_day"])
        per_symbol.append(
            set(
                pd.to_datetime(
                    frame["source_day"], utc=True, errors="coerce"
                )
                .dropna()
                .dt.normalize()
            )
        )
    return set.intersection(*per_symbol) if per_symbol else set()


def collect_next_day(cfg, observed: pd.Timestamp) -> dict[str, object]:
    book_root = cfg.forward_root / "book_depth"
    daily_root = ensure_dir(book_root / "daily_features")
    target = next_q90_book_day(cfg, observed)
    if target is None:
        return {"status": "no_publishable_day_due"}
    existing = {
        symbol: pd.read_parquet(path)
        for symbol in FROZEN_SYMBOLS
        if (path := daily_root / f"{symbol}.parquet").exists()
    }
    manifest = backfill_binance_book_depth(
        list(FROZEN_SYMBOLS),
        target.date(),
        target.date(),
        BinanceBookDepthConfig(
            output_root=book_root,
            download_workers=cfg.download_workers,
        ),
    )
    _merge_daily(existing, daily_root)
    hourly = build_hourly_book_depth_panel(
        list(FROZEN_SYMBOLS),
        BinanceBookDepthHourlyConfig(
            data_root=book_root,
            workers=cfg.download_workers,
        ),
    )
    ready = _ready_days(daily_root)
    day_ready = target.normalize() in ready
    attempts = ensure_dir(cfg.forward_root / "collection_attempts")
    attempt = {
        "attempted_at_utc": observed.isoformat(),
        "book_day": target.date().isoformat(),
        "day_ready": day_ready,
        "book_symbols_requested": len(FROZEN_SYMBOLS),
        "book_symbols_downloaded": int(manifest["downloaded_days"].gt(0).sum()),
        "book_symbols_ready": int(
            sum(
                target.normalize()
                in set(
                    pd.to_datetime(
                        pd.read_parquet(
                            daily_root / f"{symbol}.parquet",
                            columns=["source_day"],
                        )["source_day"],
                        utc=True,
                        errors="coerce",
                    ).dt.normalize()
                )
                for symbol in FROZEN_SYMBOLS
            )
        ),
        "book_symbols_missing": int(manifest["missing_days"].gt(0).sum()),
        "hourly_symbols_ready": int(hourly["rows"].gt(0).sum()),
        "ready_days": [day.isoformat() for day in sorted(ready)],
        "isolation_root": str(cfg.forward_root),
    }
    (attempts / f"{target.date().isoformat()}.json").write_text(
        json.dumps(attempt, indent=2), encoding="utf-8"
    )
    if day_ready:
        payload = {
            **attempt,
            "book_symbols_ready": len(FROZEN_SYMBOLS),
            "book_symbols_missing": 0,
            "hourly_symbols_ready": len(FROZEN_SYMBOLS),
        }
        (cfg.forward_root / "forward_collection_manifest.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return attempt


def _read_btc_cache(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def refresh_pending_btc_bars(cfg, observed: pd.Timestamp) -> pd.DataFrame:
    decisions_path = cfg.report_root / "forward" / "event_decisions.parquet"
    if not decisions_path.exists():
        return pd.DataFrame()
    decisions = pd.read_parquet(decisions_path)
    if decisions.empty:
        return pd.DataFrame()
    decisions["entry_time"] = pd.to_datetime(
        decisions["entry_time"], utc=True, errors="coerce"
    )
    start = decisions["entry_time"].min() - pd.Timedelta(hours=26)
    end = min(
        observed,
        decisions["entry_time"].max() + pd.Timedelta(hours=4, minutes=15),
    )
    base = load_config(cfg.base_config)
    client = BybitClient(
        str(base.exchanges.bybit.base_url), base.exchanges.bybit.category
    )
    try:
        bars = client.klines("BTCUSDT", start, end, "15m")
    finally:
        client.close()
    cache_path = cfg.forward_root / "btc_event_bars" / "BTCUSDT.parquet"
    existing = _read_btc_cache(cache_path)
    merged = pd.concat(
        [frame for frame in (existing, bars) if not frame.empty],
        ignore_index=True,
    )
    merged["bar_open_time"] = pd.to_datetime(
        merged["bar_open_time"], utc=True, errors="coerce"
    )
    merged = (
        merged.dropna(subset=["bar_open_time"])
        .drop_duplicates("bar_open_time", keep="last")
        .sort_values("bar_open_time")
        .reset_index(drop=True)
    )
    write_parquet(merged, cache_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the frozen q90 delayed forward-research shadow."
    )
    parser.add_argument(
        "--config", default="configs/v23_8_q90_forward_shadow.yaml"
    )
    parser.add_argument("--skip-collection", action="store_true")
    args = parser.parse_args()
    cfg = load_q90_live_config(args.config)
    observed = pd.Timestamp.now(tz="UTC")
    if not args.skip_collection:
        print(json.dumps(collect_next_day(cfg, observed), indent=2))
    outputs = write_q90_ledgers(cfg, observed_at=observed)
    btc = refresh_pending_btc_bars(cfg, observed)
    if not btc.empty:
        outputs = write_q90_ledgers(
            cfg, observed_at=observed, btc_bars=btc
        )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
