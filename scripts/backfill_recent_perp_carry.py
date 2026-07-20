from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.recent_perp_carry_history import (
    RecentPerpCarryConfig,
    backfill_recent_perp_carry,
)


MEMBERSHIP_PATH = Path(
    "reports/v11_0_balanced_topology_break/monthly_balanced_membership.csv"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill recent Bybit/Binance carry data.")
    parser.add_argument(
        "--start", type=pd.Timestamp, default=pd.Timestamp("2026-06-03", tz="UTC")
    )
    parser.add_argument(
        "--end", type=pd.Timestamp, default=pd.Timestamp("2026-07-15 01:00", tz="UTC")
    )
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    membership = pd.read_csv(MEMBERSHIP_PATH, usecols=["symbol"])
    symbols = sorted(set(membership["symbol"].astype(str)) | {"BTCUSDT"})
    manifest = backfill_recent_perp_carry(
        symbols,
        args.start.tz_convert("UTC") if args.start.tzinfo else args.start.tz_localize("UTC"),
        args.end.tz_convert("UTC") if args.end.tzinfo else args.end.tz_localize("UTC"),
        RecentPerpCarryConfig(download_workers=args.workers),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
