from __future__ import annotations

from pressure_graph.reports.v148_funding_sign_hysteresis import (
    write_v148_funding_sign_hysteresis,
)


if __name__ == "__main__":
    for name, path in write_v148_funding_sign_hysteresis().items():
        print(f"{name}: {path}")
