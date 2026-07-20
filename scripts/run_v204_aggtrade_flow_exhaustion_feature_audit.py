from pressure_graph.reports.v204_aggtrade_flow_exhaustion_feature_audit import (
    write_v204_feature_audit,
)


if __name__ == "__main__":
    for name, path in write_v204_feature_audit().items():
        print(f"{name}: {path}")
