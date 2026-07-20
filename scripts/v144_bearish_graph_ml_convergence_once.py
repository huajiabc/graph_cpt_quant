from __future__ import annotations

from pressure_graph.reports.v144_bearish_graph_ml_convergence import (
    write_v144_bearish_graph_ml_convergence,
)


if __name__ == "__main__":
    for name, path in write_v144_bearish_graph_ml_convergence().items():
        print(f"{name}: {path}")
