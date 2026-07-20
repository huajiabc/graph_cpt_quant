from pressure_graph.reports.v206_aggtrade_flow_exhaustion_audit import (
    write_v206_audit,
)


if __name__ == "__main__":
    for name, path in write_v206_audit().items():
        print(f"{name}: {path}")
