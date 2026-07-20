from pressure_graph.reports.v214_dex_community_propagation_audit import (
    write_v214_audit,
)


if __name__ == "__main__":
    for name, path in write_v214_audit().items():
        print(f"{name}: {path}")
