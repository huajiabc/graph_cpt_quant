from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.paper_live.v07d2 import S2_PAPER_LIVE_ROOT, write_v07d2_s2_paper_live
from pressure_graph.reports.v10_short_mirror import PAPER_LIVE_ROOT, write_v10_short_paper_live


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v1.0 short mirror paper-live shadow refresh.")
    parser.add_argument(
        "--prepared-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
        help="Prepared live feature parquet produced by v0.7D.2.",
    )
    parser.add_argument("--signal-days", type=int, default=7)
    parser.add_argument(
        "--family",
        choices=("s1", "s2", "all"),
        default="all",
        help="Which short family to roll: s1 = bearish mirror (default historical), "
        "s2 = long-failure -> short (P0b paper-live shadow), all = both.",
    )
    args = parser.parse_args()

    prepared_path = Path(args.prepared_path)
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared live feature file not found: {prepared_path}")
    prepared = pd.read_parquet(prepared_path)

    if args.family in {"s1", "all"}:
        outputs = write_v10_short_paper_live(
            prepared, signal_days=args.signal_days, report_root=PAPER_LIVE_ROOT
        )
        for name, path in outputs.items():
            print(f"s1.{name}: {path}")
    if args.family in {"s2", "all"}:
        s2_outputs = write_v07d2_s2_paper_live(
            prepared, signal_days=args.signal_days, report_root=S2_PAPER_LIVE_ROOT
        )
        for name, path in s2_outputs.items():
            print(f"s2.{name}: {path}")


if __name__ == "__main__":
    main()
