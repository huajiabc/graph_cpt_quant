from __future__ import annotations

from pressure_graph.reports.v147_funding_sign_spread import (
    write_v147_funding_sign_spread,
)


if __name__ == "__main__":
    for name, path in write_v147_funding_sign_spread().items():
        print(f"{name}: {path}")
