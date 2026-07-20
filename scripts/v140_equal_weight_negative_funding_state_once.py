from __future__ import annotations

from pressure_graph.reports.v140_equal_weight_negative_funding_state import (
    write_v140_equal_weight_negative_funding_state,
)


if __name__ == "__main__":
    for name, path in write_v140_equal_weight_negative_funding_state().items():
        print(f"{name}: {path}")
