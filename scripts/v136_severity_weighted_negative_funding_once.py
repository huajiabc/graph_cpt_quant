from __future__ import annotations

from pressure_graph.reports.v136_severity_weighted_negative_funding import (
    write_v136_severity_weighted_negative_funding,
)


if __name__ == "__main__":
    for name, path in write_v136_severity_weighted_negative_funding().items():
        print(f"{name}: {path}")
