from pressure_graph.reports.v129_causal_risk_parity_combination import (
    write_v129_causal_risk_parity_combination,
)


if __name__ == "__main__":
    for name, path in write_v129_causal_risk_parity_combination().items():
        print(f"{name}: {path}")
