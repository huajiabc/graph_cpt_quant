from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.recent_perp_carry_history import (
    RecentPerpCarryConfig,
    backfill_recent_perp_carry,
)
from pressure_graph.reports.v2334_okx_liquidation_forward_data_audit import (
    EXPECTED_SYMBOLS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=pd.Timestamp, required=True)
    parser.add_argument("--end", type=pd.Timestamp, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external/okx_liquidation_forward/mechanism_prices"),
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    start = pd.Timestamp(args.start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    end = pd.Timestamp(args.end)
    end = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")
    manifest = backfill_recent_perp_carry(
        list(EXPECTED_SYMBOLS),
        start,
        end,
        RecentPerpCarryConfig(
            output_root=args.output_root,
            download_workers=args.workers,
        ),
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
