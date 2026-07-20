import argparse
from pathlib import Path

from pressure_graph.reports.v107_flow_graph_readiness import (
    V107Config,
    write_v107_flow_graph_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the forward cross-venue flow-graph data gate."
    )
    parser.add_argument("--tape-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=None)
    args = parser.parse_args()
    defaults = V107Config()
    cfg = V107Config(
        tape_root=args.tape_root or defaults.tape_root,
        report_root=args.report_root or defaults.report_root,
    )
    for name, path in write_v107_flow_graph_readiness(cfg).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
