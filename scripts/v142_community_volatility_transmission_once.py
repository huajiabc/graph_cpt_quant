from __future__ import annotations

from pressure_graph.reports.v142_community_volatility_transmission import (
    write_v142_community_volatility_transmission,
)


if __name__ == "__main__":
    for name, path in write_v142_community_volatility_transmission().items():
        print(f"{name}: {path}")
