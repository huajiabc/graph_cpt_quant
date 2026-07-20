from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date

from pressure_graph.binance_um_book_depth_history import (
    BinanceBookDepthConfig,
    backfill_binance_book_depth,
)


FROZEN_SYMBOLS = [
    "SOLUSDT",
    "DOGEUSDT",
    "1000PEPEUSDT",
    "WIFUSDT",
    "ETHUSDT",
    "ENAUSDT",
    "HBARUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "ONDOUSDT",
    "XRPUSDT",
    "XLMUSDT",
    "FARTCOINUSDT",
    "WLDUSDT",
    "SEIUSDT",
    "TIAUSDT",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    cfg = replace(BinanceBookDepthConfig(), download_workers=args.workers)
    manifest = backfill_binance_book_depth(FROZEN_SYMBOLS, args.start, args.end, cfg)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
