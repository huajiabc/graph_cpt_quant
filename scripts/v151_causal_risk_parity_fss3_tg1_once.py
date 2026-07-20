from __future__ import annotations

from pressure_graph.reports.v151_causal_risk_parity_fss3_tg1 import (
    write_v151_causal_risk_parity_fss3_tg1,
)


if __name__ == "__main__":
    for name, path in write_v151_causal_risk_parity_fss3_tg1().items():
        print(f"{name}: {path}")
