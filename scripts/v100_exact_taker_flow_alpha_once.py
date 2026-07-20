from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v100_exact_taker_flow_alpha import (  # noqa: E402
    V100Config,
    write_v100_exact_taker_flow_alpha,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered v10.0 exact taker-flow alpha test."
    )
    parser.add_argument(
        "--event-path",
        type=Path,
        default=Path("reports/v0_1/entry_policy_1m_trades.csv"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/bybit/public_trading_parquet"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("data/processed/v100_exact_taker_flow_1m"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/v10_0_exact_taker_flow_alpha"),
    )
    parser.add_argument("--random-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--cache-workers", type=int, default=4)
    args = parser.parse_args()
    outputs = write_v100_exact_taker_flow_alpha(
        V100Config(
            event_path=args.event_path,
            raw_root=args.raw_root,
            cache_root=args.cache_root,
            report_root=args.report_root,
            random_iterations=args.random_iterations,
            bootstrap_iterations=args.bootstrap_iterations,
            cache_workers=args.cache_workers,
        )
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
