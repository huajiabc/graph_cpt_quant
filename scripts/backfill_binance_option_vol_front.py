from __future__ import annotations

import argparse
from datetime import date

from pressure_graph.binance_option_eoh_history import (
    BinanceOptionHistoryConfig,
    backfill_binance_option_vol_front,
)


DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "LTCUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "BCHUSDT",
    "MATICUSDT",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 5, 18))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2023, 10, 23))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--no-raw", action="store_true")
    args = parser.parse_args()
    cfg = BinanceOptionHistoryConfig(
        download_workers=max(1, args.workers),
        retain_raw_archives=not args.no_raw,
    )
    outputs = backfill_binance_option_vol_front(
        "BTCUSDT", DEFAULT_SYMBOLS, args.start, args.end, cfg
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
