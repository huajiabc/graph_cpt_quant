from __future__ import annotations

from pressure_graph.reports.v145_bearish_convergence_horizon_extension import (
    write_v145_bearish_convergence_horizon_extension,
)


if __name__ == "__main__":
    for name, path in write_v145_bearish_convergence_horizon_extension().items():
        print(f"{name}: {path}")
