from __future__ import annotations

from pressure_graph.reports.v137_cross_venue_consensus_negative_funding import (
    write_v137_cross_venue_consensus_negative_funding,
)


if __name__ == "__main__":
    for name, path in write_v137_cross_venue_consensus_negative_funding().items():
        print(f"{name}: {path}")
