from pressure_graph.reports.v208_unhedged_flow_exhaustion_audit import (
    write_v208_audit,
)


if __name__ == "__main__":
    for name, path in write_v208_audit().items():
        print(f"{name}: {path}")
