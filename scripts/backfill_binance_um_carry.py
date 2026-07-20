from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from pressure_graph.binance_um_carry_history import (
    BinanceUmCarryConfig,
    backfill_binance_um_carry,
)


DEFAULT_MEMBERSHIP = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Binance USD-M one-hour klines and funding."
    )
    parser.add_argument("--kline-start", type=date.fromisoformat, default=date(2025, 8, 1))
    parser.add_argument("--funding-start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 4))
    parser.add_argument("--membership", type=Path, default=DEFAULT_MEMBERSHIP)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/binance_um_carry"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--symbols", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.symbols:
        symbols = args.symbols
    else:
        membership = pd.read_csv(args.membership, usecols=["symbol"])
        symbols = sorted(membership["symbol"].astype(str).unique())
    manifest = backfill_binance_um_carry(
        symbols,
        args.kline_start,
        args.funding_start,
        args.end,
        BinanceUmCarryConfig(
            output_root=args.output_root,
            download_workers=args.workers,
        ),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
