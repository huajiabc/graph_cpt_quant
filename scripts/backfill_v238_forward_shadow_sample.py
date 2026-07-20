from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import date
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
from pressure_graph.recent_perp_carry_history import (
    RecentPerpCarryConfig,
    download_recent_perp_symbol,
)
from pressure_graph.reports.v155_binance_one_percent_depth_imbalance import (
    FROZEN_SYMBOLS,
)


ROOT = Path("data/external/v238_forward_shadow")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect isolated forward data for the frozen v23.8 candidate."
    )
    parser.add_argument("--book-day", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--perp-start",
        type=pd.Timestamp,
        default=pd.Timestamp("2026-07-15 00:00:00+00:00"),
    )
    parser.add_argument("--perp-end", type=pd.Timestamp, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_collection_manifests(manifest: dict[str, object]) -> str:
    ROOT.mkdir(parents=True, exist_ok=True)
    attempts_root = ROOT / "collection_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    attempt_path = attempts_root / f"{manifest['book_day']}.json"
    payload = json.dumps(manifest, indent=2)
    attempt_path.write_text(payload, encoding="utf-8")
    (ROOT / "latest_collection_attempt.json").write_text(
        payload, encoding="utf-8"
    )

    complete = (
        int(manifest["book_symbols_ready"]) == len(FROZEN_SYMBOLS)
        and int(manifest["book_symbols_missing"]) == 0
        and int(manifest["hourly_symbols_ready"]) == len(FROZEN_SYMBOLS)
        and not manifest["perp"]["error"]
    )
    successful_path = ROOT / "forward_collection_manifest.json"
    if complete or not successful_path.exists():
        successful_path.write_text(payload, encoding="utf-8")
    return "success_manifest_updated" if complete else "failed_attempt_preserved"


def _count_ready_daily_symbols(book_root: Path, book_day: date) -> int:
    target = pd.Timestamp(book_day, tz="UTC")
    ready = 0
    for symbol in FROZEN_SYMBOLS:
        path = book_root / "daily_features" / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["source_day"])
        source_day = pd.to_datetime(frame["source_day"], utc=True, errors="coerce")
        ready += int(source_day.eq(target).any())
    return ready


def _merge_existing_daily_features(
    existing: dict[str, pd.DataFrame],
    book_root: Path,
) -> None:
    for symbol, historical in existing.items():
        path = book_root / "daily_features" / f"{symbol}.parquet"
        if not path.exists():
            _atomic_parquet(historical, path)
            continue
        current = pd.read_parquet(path)
        merged = pd.concat([historical, current], ignore_index=True)
        merged["source_day"] = pd.to_datetime(
            merged["source_day"], utc=True, errors="coerce"
        )
        merged = (
            merged.dropna(subset=["source_day"])
            .drop_duplicates("source_day", keep="last")
            .sort_values("source_day")
            .reset_index(drop=True)
        )
        _atomic_parquet(merged, path)


def main() -> None:
    args = _arguments()
    book_root = ROOT / "book_depth"
    perp_root = ROOT / "perp"
    existing_daily = {
        symbol: pd.read_parquet(path)
        for symbol in FROZEN_SYMBOLS
        if (path := book_root / "daily_features" / f"{symbol}.parquet").exists()
    }
    book_cfg = replace(
        BinanceBookDepthConfig(),
        output_root=book_root,
        download_workers=args.workers,
    )
    daily_manifest = backfill_binance_book_depth(
        list(FROZEN_SYMBOLS),
        args.book_day,
        args.book_day,
        book_cfg,
    )
    _merge_existing_daily_features(existing_daily, book_root)
    hourly_manifest = build_hourly_book_depth_panel(
        list(FROZEN_SYMBOLS),
        BinanceBookDepthHourlyConfig(
            data_root=book_root,
            workers=args.workers,
        ),
    )
    book_symbols_ready = _count_ready_daily_symbols(book_root, args.book_day)
    perp_end = pd.Timestamp(args.perp_end)
    perp_end = (
        perp_end.tz_convert("UTC")
        if perp_end.tzinfo
        else perp_end.tz_localize("UTC")
    )
    perp_start = pd.Timestamp(args.perp_start)
    perp_start = (
        perp_start.tz_convert("UTC")
        if perp_start.tzinfo
        else perp_start.tz_localize("UTC")
    )
    existing_btc_path = perp_root / "bybit_klines_15m" / "BTCUSDT.parquet"
    existing_btc = (
        pd.read_parquet(existing_btc_path) if existing_btc_path.exists() else None
    )
    perp_result = download_recent_perp_symbol(
        "BTCUSDT",
        perp_start,
        perp_end,
        RecentPerpCarryConfig(output_root=perp_root),
    )
    if existing_btc is not None and existing_btc_path.exists():
        current_btc = pd.read_parquet(existing_btc_path)
        merged_btc = pd.concat([existing_btc, current_btc], ignore_index=True)
        merged_btc["bar_open_time"] = pd.to_datetime(
            merged_btc["bar_open_time"], utc=True, errors="coerce"
        )
        merged_btc = (
            merged_btc.dropna(subset=["bar_open_time"])
            .drop_duplicates("bar_open_time", keep="last")
            .sort_values("bar_open_time")
            .reset_index(drop=True)
        )
        _atomic_parquet(merged_btc, existing_btc_path)
    manifest = {
        "book_day": args.book_day.isoformat(),
        "perp_start": perp_start.isoformat(),
        "perp_end": perp_end.isoformat(),
        "book_symbols_requested": len(FROZEN_SYMBOLS),
        "book_symbols_downloaded": int(daily_manifest["downloaded_days"].gt(0).sum()),
        "book_symbols_ready": book_symbols_ready,
        "book_symbols_missing": int(daily_manifest["missing_days"].gt(0).sum()),
        "hourly_symbols_ready": int(hourly_manifest["rows"].gt(0).sum()),
        "perp": asdict(perp_result),
        "isolation_root": str(ROOT),
    }
    manifest["manifest_write_status"] = _write_collection_manifests(manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
