from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v99_sparse_cost_leverage import (  # noqa: E402
    V99Config,
    write_v99_sparse_cost_leverage,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen v9.9 sparse residual and leverage stress evaluation."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("reports/v9_8_residual_hysteresis_4h"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/v9_9_sparse_cost_leverage_4h"),
    )
    parser.add_argument("--horizon-hours", type=int, choices=(4, 12), default=4)
    parser.add_argument("--random-iterations", type=int, default=500)
    args = parser.parse_args()
    outputs = write_v99_sparse_cost_leverage(
        V99Config(
            prediction_root=args.prediction_root,
            report_root=args.report_root,
            horizon_hours=args.horizon_hours,
            random_iterations=args.random_iterations,
        )
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
