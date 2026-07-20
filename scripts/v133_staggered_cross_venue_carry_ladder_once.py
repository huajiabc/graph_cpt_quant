from __future__ import annotations

from pressure_graph.reports.v133_staggered_cross_venue_carry_ladder import (
    write_v133_staggered_cross_venue_carry_ladder,
)


if __name__ == "__main__":
    for name, path in write_v133_staggered_cross_venue_carry_ladder().items():
        print(f"{name}: {path}")
