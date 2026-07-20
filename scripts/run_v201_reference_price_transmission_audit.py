from pressure_graph.reports.v201_reference_price_transmission_audit import (
    write_v201_independent_audit,
)


if __name__ == "__main__":
    for name, path in write_v201_independent_audit().items():
        print(f"{name}: {path}")
