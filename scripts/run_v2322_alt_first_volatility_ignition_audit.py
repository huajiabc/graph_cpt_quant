from pressure_graph.reports.v2322_alt_first_volatility_ignition_audit import (
    write_v2322_alt_first_volatility_ignition_audit,
)


if __name__ == "__main__":
    for name, path in write_v2322_alt_first_volatility_ignition_audit().items():
        print(f"{name}: {path}")
