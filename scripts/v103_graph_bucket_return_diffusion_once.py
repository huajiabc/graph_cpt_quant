from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pressure_graph.reports.v103_graph_bucket_return_diffusion import (  # noqa: E402
    V103Config,
    write_v103_graph_bucket_return_diffusion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preregistered graph bucket-return diffusion audit."
    )
    parser.add_argument("--random-iterations", type=int, default=50)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/v10_3_graph_bucket_return_diffusion"),
    )
    args = parser.parse_args()
    outputs = write_v103_graph_bucket_return_diffusion(
        V103Config(
            report_root=args.report_root,
            random_iterations=args.random_iterations,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
