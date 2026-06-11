from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pressure_graph.reports.v10_short_mirror import PAPER_LIVE_ROOT, write_v10_short_paper_live


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v1.0 short mirror paper-live shadow refresh.")
    parser.add_argument(
        "--prepared-path",
        default="data/live_v07d2/processed/v0_7d2_live_features.parquet",
        help="Prepared live feature parquet produced by v0.7D.2.",
    )
    parser.add_argument("--signal-days", type=int, default=7)
    args = parser.parse_args()

    prepared_path = Path(args.prepared_path)
    if not prepared_path.exists():
        raise FileNotFoundError(f"prepared live feature file not found: {prepared_path}")
    prepared = pd.read_parquet(prepared_path)
    outputs = write_v10_short_paper_live(prepared, signal_days=args.signal_days, report_root=PAPER_LIVE_ROOT)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

