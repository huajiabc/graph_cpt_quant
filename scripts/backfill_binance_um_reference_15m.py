from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from pressure_graph.binance_um_reference_history import (
    BinanceReferenceConfig,
    backfill_binance_reference,
)


METRICS_ROOT = Path("data/external/binance_um_metrics_5m")
KLINE_ROOT = Path("data/raw/binance/klines")
EXCLUDED = {"XAUTUSDT"}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill checksummed Binance USD-M 15m mark/index klines."
    )
    parser.add_argument(
        "dataset", choices=["markPriceKlines", "indexPriceKlines"]
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 4))
    parser.add_argument("--output-root", type=Path)
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
    default_root = Path(
        "data/external/binance_um_mark_price_15m"
        if args.dataset == "markPriceKlines"
        else "data/external/binance_um_index_price_15m"
    )
    manifest = backfill_binance_reference(
        symbols,
        args.start,
        args.end,
        BinanceReferenceConfig(
            dataset=args.dataset,
            output_root=args.output_root or default_root,
            download_workers=args.workers,
            verify_checksums=not args.skip_checksums,
        ),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
