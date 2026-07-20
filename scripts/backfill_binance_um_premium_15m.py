from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pressure_graph.binance_um_premium_history import (
    BinancePremiumConfig,
    backfill_binance_premium,
)


METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
KLINE_ROOT = Path("data/raw/binance/klines")
EXCLUDED = {"XAUTUSDT"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill checksummed Binance USD-M 15m premium-index klines."
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 14))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/binance_um_premium_15m"),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--skip-checksums", action="store_true")
    return parser.parse_args()


def _default_symbols() -> list[str]:
    metric_symbols = {path.stem.upper() for path in METRICS_ROOT.glob("*.parquet")}
    kline_symbols = {path.stem.upper() for path in KLINE_ROOT.glob("*.parquet")}
    return sorted((metric_symbols & kline_symbols) - EXCLUDED)


def main() -> None:
    args = _arguments()
    symbols = args.symbols or _default_symbols()
    manifest = backfill_binance_premium(
        symbols,
        args.start,
        args.end,
        BinancePremiumConfig(
            output_root=args.output_root,
            download_workers=args.workers,
            verify_checksums=not args.skip_checksums,
        ),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
