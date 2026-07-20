from pressure_graph.reports.v220_spot_perp_flow_inventory_audit import (
    write_v220_audit,
)


if __name__ == "__main__":
    for name, path in write_v220_audit().items():
        print(f"{name}: {path}")
