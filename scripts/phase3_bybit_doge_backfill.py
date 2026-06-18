"""Backfill Bybit linear continuous CVD for DOGEUSDT across the Phase 3 months.

Phase 2C left only October 2025 on disk. The Phase 3A time-stability gate
needs ≥3 distinct months; this script fills 2025-08, 09, 11, 12, 2026-01,
02, 03. Run sequentially per the Phase 2C parquet-race fix.

Bybit SSL read timeouts have been observed on Aug-2025 day-zips ≥ 50 MB; the
driver wraps each per-day call with retry-and-backoff so a transient drop
does not kill the entire month.

Usage: ``python scripts/phase3_bybit_doge_backfill.py`` from the
``graph_cpt_quant`` directory.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import urllib.error
from datetime import date, timedelta
from pathlib import Path


def _calendar_month_days(year: int, month: int) -> tuple[date, date]:
    start = date(year, month, 1)
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    end_inclusive = date.fromordinal(next_month.toordinal() - 1)
    return start, end_inclusive


def main() -> int:
    parser = argparse.ArgumentParser(description="Bybit DOGE multi-month CVD backfill")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument(
        "--months",
        nargs="+",
        default=[
            "2025-08",
            "2025-09",
            "2025-11",
            "2025-12",
            "2026-01",
            "2026-02",
            "2026-03",
        ],
        help="Months in YYYY-MM form to backfill (2025-10 skipped — already on disk).",
    )
    parser.add_argument(
        "--history-root",
        default="data/orderflow_history/bybit_linear",
        help="Bybit history root; default matches the existing local layout.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pressure_graph.bybit_continuous_cvd import (
        BybitCvdConfig,
        backfill_symbol_day,
        raw_path,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/phase3_bybit_doge_backfill.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    # Larger timeout — 50 MB day-zips routinely take > 60s on a domestic line.
    cfg = BybitCvdConfig(history_root=Path(args.history_root), timeout_seconds=300)
    transient_errors = (
        TimeoutError,
        urllib.error.URLError,
        ConnectionResetError,
        OSError,
    )

    def _backfill_day_with_retry(day: date, max_attempts: int = 5) -> str:
        if raw_path(cfg, args.symbol, day).exists():
            try:
                backfill_symbol_day(cfg, args.symbol, day)
                return "from-cache"
            except Exception as exc:  # rebuild fail — re-download below
                logging.warning("rebuild fail for %s %s — refetch: %s", args.symbol, day, exc)
        for attempt in range(1, max_attempts + 1):
            try:
                backfill_symbol_day(cfg, args.symbol, day)
                return "built"
            except urllib.error.HTTPError as exc:
                # 404 → Bybit hasn't published that day. Stop retrying.
                logging.warning("HTTP %s for %s %s — skip", exc.code, args.symbol, day)
                return "http-skip"
            except transient_errors as exc:
                backoff = min(30, 2 ** attempt)
                logging.warning(
                    "transient %s on %s %s (attempt %d/%d) — sleep %ds",
                    type(exc).__name__,
                    args.symbol,
                    day,
                    attempt,
                    max_attempts,
                    backoff,
                )
                time.sleep(backoff)
        logging.error("giving up on %s %s after %d attempts", args.symbol, day, max_attempts)
        return "give-up"

    total_built = total_cached = total_skipped = total_giveup = 0
    started = time.time()
    for month_key in args.months:
        year, month = (int(p) for p in month_key.split("-"))
        start, end = _calendar_month_days(year, month)
        logging.info("backfill %s %s..%s", args.symbol, start, end)
        t0 = time.time()
        cur = start
        m_built = m_cached = m_skip = m_giveup = 0
        while cur <= end:
            outcome = _backfill_day_with_retry(cur)
            if outcome == "built":
                m_built += 1
            elif outcome == "from-cache":
                m_cached += 1
            elif outcome == "http-skip":
                m_skip += 1
            else:
                m_giveup += 1
            cur += timedelta(days=1)
        logging.info(
            "month %s: built=%d from_cache=%d http_skip=%d give_up=%d in %.1fs",
            month_key,
            m_built,
            m_cached,
            m_skip,
            m_giveup,
            time.time() - t0,
        )
        total_built += m_built
        total_cached += m_cached
        total_skipped += m_skip
        total_giveup += m_giveup
    logging.info(
        "TOTAL: built=%d from_cache=%d http_skip=%d give_up=%d in %.1fs",
        total_built,
        total_cached,
        total_skipped,
        total_giveup,
        time.time() - started,
    )
    return 0 if total_giveup == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
