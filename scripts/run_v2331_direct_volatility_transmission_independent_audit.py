from pressure_graph.reports.v2331_direct_volatility_transmission_independent_audit import (
    write_v2331_direct_volatility_transmission_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v2331_direct_volatility_transmission_independent_audit().items():
        print(f"{name}: {path}")
