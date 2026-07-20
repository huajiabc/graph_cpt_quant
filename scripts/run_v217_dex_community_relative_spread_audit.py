from pressure_graph.reports.v217_dex_community_relative_spread_audit import (
    write_v217_audit,
)


if __name__ == "__main__":
    for name, path in write_v217_audit().items():
        print(f"{name}: {path}")
