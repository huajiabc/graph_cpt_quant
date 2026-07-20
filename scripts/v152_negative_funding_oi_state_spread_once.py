from __future__ import annotations

from pressure_graph.reports.v152_negative_funding_oi_state_spread import (
    write_v152_negative_funding_oi_state_spread,
)


if __name__ == "__main__":
    for name, path in write_v152_negative_funding_oi_state_spread().items():
        print(f"{name}: {path}")
