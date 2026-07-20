from __future__ import annotations

from pressure_graph.reports.v138_negative_funding_oi_state import (
    write_v138_negative_funding_oi_state,
)


if __name__ == "__main__":
    for name, path in write_v138_negative_funding_oi_state().items():
        print(f"{name}: {path}")
