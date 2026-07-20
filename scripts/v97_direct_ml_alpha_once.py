from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v97_direct_ml_alpha import V97Config, write_v97_direct_ml_alpha  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen v9.7 direct ML alpha evaluation.")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path("data/processed/v0_3/perp_pressure_features_all_eligible.parquet"),
    )
    parser.add_argument("--report-root", type=Path, default=Path("reports/v9_7_direct_ml_alpha"))
    parser.add_argument("--random-iterations", type=int, default=200)
    parser.add_argument("--horizon-hours", type=int, choices=(4, 12), default=4)
    args = parser.parse_args()
    outputs = write_v97_direct_ml_alpha(
        V97Config(
            feature_path=args.feature_path,
            report_root=args.report_root,
            random_iterations=args.random_iterations,
            horizon_hours=args.horizon_hours,
            label_col=f"future_ret_{args.horizon_hours}h",
        )
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
