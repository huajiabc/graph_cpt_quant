from __future__ import annotations

from pressure_graph.reports.v134_negative_funding_beta_neutral_rebound import (
    write_v134_negative_funding_beta_neutral_rebound,
)


if __name__ == "__main__":
    for name, path in write_v134_negative_funding_beta_neutral_rebound().items():
        print(f"{name}: {path}")
