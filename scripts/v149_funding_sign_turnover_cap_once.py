from __future__ import annotations

from pressure_graph.reports.v149_funding_sign_turnover_cap import (
    write_v149_funding_sign_turnover_cap,
)


if __name__ == "__main__":
    for name, path in write_v149_funding_sign_turnover_cap().items():
        print(f"{name}: {path}")
