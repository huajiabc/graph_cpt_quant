"""Driver for the continuous Binance UM CVD backfill (77.docx P1).

Iterates a symbol list × date range, calls
``binance_continuous_cvd.backfill_symbol_day`` per (symbol, day),
emits coverage + quality audit reports. Safe to re-run — existing
shards are merged in (de-duped by bar_open_time).

CLI:
    python scripts/v7s_continuous_cvd_backfill.py \\
        --symbols BTCUSDT ETHUSDT \\
        --start-day 2025-10-01 --end-day 2025-10-31 \\
        --download-missing

Defaults skip the download step (relies on the existing aggTrades
cache). Pass ``--download-missing`` on the A100 server where the
proxy is wired up.

The full top-30 × 12 months backfill is best run on the A100 box in
chunks of ~30 days × 5 symbols to keep memory + disk pressure low.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from pressure_graph.binance_continuous_cvd import (
    ContinuousCvdConfig,
    backfill_symbol_days,
    write_coverage_report,
    write_quality_audit,
)


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="vData Continuous Binance UM CVD backfill")
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Binance UM symbols (e.g. BTCUSDT ETHUSDT). Use Bybit→Binance aliases yourself if needed.",
    )
    parser.add_argument("--start-day", type=_parse_day, required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-day", type=_parse_day, required=True, help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download zip archives that are not in the local cache (server-only).",
    )
    parser.add_argument(
        "--bar-sizes",
        nargs="+",
        default=["1min", "5min", "15min"],
        help="Bar sizes to aggregate at (default 1min/5min/15min).",
    )
    parser.add_argument(
        "--continuous-root",
        type=Path,
        default=None,
        help="Override the continuous output root (debug only).",
    )
    parser.add_argument(
        "--skip-quality-audit",
        action="store_true",
        help="Skip walking the shards for the quality audit (saves IO on huge batches).",
    )
    args = parser.parse_args()

    cfg = ContinuousCvdConfig(
        bar_sizes=tuple(args.bar_sizes),
        download_if_missing=args.download_missing,
        **({"continuous_root": args.continuous_root} if args.continuous_root else {}),
    )
    stats = backfill_symbol_days(args.symbols, args.start_day, args.end_day, cfg)
    print(
        f"vData backfill complete: complete={stats.days_complete} empty={stats.days_empty} "
        f"missing={stats.days_missing} error={stats.days_error}",
        flush=True,
    )
    coverage_path = write_coverage_report(stats, cfg)
    print(f"coverage report: {coverage_path}", flush=True)
    if not args.skip_quality_audit:
        audit_path = write_quality_audit(args.symbols, cfg)
        print(f"quality audit: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
