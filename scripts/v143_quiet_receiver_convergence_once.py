from __future__ import annotations

from pressure_graph.reports.v143_quiet_receiver_convergence import (
    write_v143_quiet_receiver_convergence,
)


if __name__ == "__main__":
    for name, path in write_v143_quiet_receiver_convergence().items():
        print(f"{name}: {path}")
