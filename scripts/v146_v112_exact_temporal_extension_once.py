from __future__ import annotations

from pressure_graph.reports.v146_v112_exact_temporal_extension import (
    write_v146_v112_exact_temporal_extension,
)


if __name__ == "__main__":
    for name, path in write_v146_v112_exact_temporal_extension().items():
        print(f"{name}: {path}")
