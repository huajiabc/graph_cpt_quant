from __future__ import annotations

from pressure_graph.reports.v139_short_crowding_pressure_rebound import (
    write_v139_short_crowding_pressure_rebound,
)


if __name__ == "__main__":
    for name, path in write_v139_short_crowding_pressure_rebound().items():
        print(f"{name}: {path}")
