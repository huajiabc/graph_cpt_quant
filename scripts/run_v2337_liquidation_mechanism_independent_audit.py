from pressure_graph.reports.v2337_liquidation_mechanism_independent_audit import (
    write_v2337,
)


if __name__ == "__main__":
    for name, path in write_v2337().items():
        print(f"{name}: {path}")
