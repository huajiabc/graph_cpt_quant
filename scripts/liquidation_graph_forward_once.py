from __future__ import annotations

import argparse

from pressure_graph.paper_live.liquidation_graph import (
    load_liquidation_graph_live_config,
    write_liquidation_graph_live_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update the remote-started liquidation graph diagnostic."
    )
    parser.add_argument(
        "--config",
        default="configs/v23_42_liquidation_graph_live_diagnostic.yaml",
    )
    args = parser.parse_args()
    cfg = load_liquidation_graph_live_config(args.config)
    for name, path in write_liquidation_graph_live_diagnostic(cfg).items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
