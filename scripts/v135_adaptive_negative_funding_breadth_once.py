from __future__ import annotations

from pressure_graph.reports.v135_adaptive_negative_funding_breadth import (
    write_v135_adaptive_negative_funding_breadth,
)


if __name__ == "__main__":
    for name, path in write_v135_adaptive_negative_funding_breadth().items():
        print(f"{name}: {path}")
