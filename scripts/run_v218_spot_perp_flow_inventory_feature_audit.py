from pressure_graph.reports.v218_spot_perp_flow_inventory_feature_audit import (
    write_v218_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v218_feature_audit().items():
        print(f"{name}: {path}")
