from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v101_exact_flow_persistence import (  # noqa: E402
    V101Config,
    write_v101_exact_flow_persistence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preregistered v10.1 exact-flow 240m persistence audit."
    )
    parser.add_argument("--random-iterations", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--btc-path",
        type=Path,
        default=Path("data/raw/bybit/klines_1m_execution_public/BTCUSDT.parquet"),
    )
    parser.add_argument("--btc-tolerance-minutes", type=int, default=2)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/v10_1_exact_flow_persistence"),
    )
    args = parser.parse_args()
    outputs = write_v101_exact_flow_persistence(
        V101Config(
            report_root=args.report_root,
            random_iterations=args.random_iterations,
            bootstrap_iterations=args.bootstrap_iterations,
            btc_path=args.btc_path,
            btc_tolerance_minutes=args.btc_tolerance_minutes,
        )
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
