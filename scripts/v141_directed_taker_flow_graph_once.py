from __future__ import annotations

from pressure_graph.reports.v141_directed_taker_flow_graph import (
    write_v141_directed_taker_flow_graph,
)


if __name__ == "__main__":
    for name, path in write_v141_directed_taker_flow_graph().items():
        print(f"{name}: {path}")
